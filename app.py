from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import supervision as sv
import torch
from flask import Flask, Response, jsonify, render_template, request
from ultralytics import YOLO
from werkzeug.utils import secure_filename

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

MODEL_PATH = os.getenv("POTHOLE_MODEL_PATH", str(BASE_DIR / "best.pt"))
DEFAULT_VIDEO = str(BASE_DIR / "demo2.mp4")
INFER_IMGSZ = int(os.getenv("INFER_IMGSZ", "512"))
MAX_CAPTURE_WIDTH = int(os.getenv("MAX_CAPTURE_WIDTH", "800"))
MAX_PHONE_WIDTH = int(os.getenv("MAX_PHONE_WIDTH", "720"))
STREAM_JPEG_QUALITY = int(os.getenv("STREAM_JPEG_QUALITY", "72"))
TARGET_LATENCY_MS = float(os.getenv("TARGET_LATENCY_MS", "450"))
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def depth_class(depth_cm: float) -> str:
    if depth_cm < 5.0:
        return "shallow"
    if depth_cm < 12.0:
        return "medium"
    return "severe"


class DetectionEngine:
    def __init__(self) -> None:
        self.model = YOLO(MODEL_PATH)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.use_half = self.device.startswith("cuda")
        self.infer_imgsz = max(320, INFER_IMGSZ)
        self.max_capture_width = max(480, MAX_CAPTURE_WIDTH)
        self.max_phone_width = max(480, MAX_PHONE_WIDTH)
        self.jpeg_quality = int(np.clip(STREAM_JPEG_QUALITY, 50, 90))
        self.target_latency_ms = max(150.0, TARGET_LATENCY_MS)

        self.box_annotator = sv.BoxAnnotator(thickness=2, color=sv.Color.from_hex("#f4b400"))
        self.label_annotator = sv.LabelAnnotator(
            text_scale=0.52,
            text_thickness=1,
            text_color=sv.Color.BLACK,
            color=sv.Color.from_hex("#f4b400"),
            text_padding=8,
        )

        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.capture_thread: Optional[threading.Thread] = None
        self.infer_thread: Optional[threading.Thread] = None
        self.capture: Optional[cv2.VideoCapture] = None

        self.mode = "upload"
        self.source_path = DEFAULT_VIDEO
        self.source_label = Path(DEFAULT_VIDEO).name
        self.stream_source_url = ""
        self.phone_last_frame_ts = 0.0

        self.reference_cm: Optional[float] = None
        self.reference_px: Optional[float] = None

        self.latest_frame: Optional[np.ndarray] = None
        self.latest_frame_id = 0
        self.latest_capture_ts = 0.0
        self.last_processed_id = -1

        self.latest_jpeg = self._make_status_frame("Idle. Select source and press Start.")
        self.latest_jpeg_id = 0

        self.infer_ms_ema = 70.0
        self.capture_fps_ema = 0.0
        self.infer_fps_ema = 0.0
        self.stream_fps_ema = 0.0
        self.prev_capture_ts = 0.0
        self.prev_stream_ts = 0.0

        self.running = False
        self.status = "idle"
        self.status_message = "Ready"

        self.stats: Dict[str, object] = {
            "detections": 0,
            "avg_confidence": 0.0,
            "avg_width_cm": 0.0,
            "avg_area_m2": 0.0,
            "avg_depth_cm": 0.0,
            "depth_class": "none",
            "estimation_confidence": 0.0,
            "capture_fps": 0.0,
            "infer_fps": 0.0,
            "stream_fps": 0.0,
            "latency_ms": 0.0,
        }

    @staticmethod
    def _ema(current: float, sample: float, alpha: float = 0.15) -> float:
        if current <= 0.0:
            return sample
        return (1.0 - alpha) * current + alpha * sample

    def _make_status_frame(self, text: str) -> bytes:
        frame = np.full((480, 854, 3), 246, dtype=np.uint8)
        cv2.putText(frame, text, (30, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (45, 45, 45), 2)
        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return buffer.tobytes() if ok else b""

    def _estimate_metrics(
        self,
        frame_gray: np.ndarray,
        xyxy: np.ndarray,
        frame_shape: Tuple[int, int, int],
    ) -> Tuple[float, float, float, float, str, float]:
        x1, y1, x2, y2 = xyxy.astype(int)
        h, w = frame_shape[:2]

        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))

        bbox_w_px = max(1, x2 - x1)
        bbox_h_px = max(1, y2 - y1)

        if self.reference_cm and self.reference_px and self.reference_px > 0:
            cm_per_px = self.reference_cm / self.reference_px
            estimate_conf = 0.9
        else:
            y_center_norm = ((y1 + y2) / 2.0) / max(h, 1)
            perspective_scale = 1.55 - (0.9 * y_center_norm)
            base_cm_per_px = 0.4 * (720.0 / max(float(h), 1.0))
            cm_per_px = base_cm_per_px * max(0.5, perspective_scale)
            estimate_conf = 0.62

        width_cm = bbox_w_px * cm_per_px
        length_cm = bbox_h_px * cm_per_px
        area_m2 = (width_cm * length_cm) / 10000.0

        roi = frame_gray[y1:y2, x1:x2]
        if roi.size == 0:
            lap_var = 0.0
            dark_ratio = 0.0
        else:
            lap_var = float(cv2.Laplacian(roi, cv2.CV_64F).var())
            dark_ratio = float((roi < 70).mean())

        depth_cm = 1.4 + (0.18 * length_cm) + (0.013 * np.sqrt(max(lap_var, 0.0))) + (6.0 * dark_ratio)
        depth_cm = float(np.clip(depth_cm, 2.0, 30.0))
        d_class = depth_class(depth_cm)

        area_px = bbox_w_px * bbox_h_px
        if area_px > 12000:
            estimate_conf = min(0.94, estimate_conf + 0.06)

        return width_cm, length_cm, area_m2, depth_cm, d_class, estimate_conf

    def _annotate_and_stats(self, frame: np.ndarray) -> np.ndarray:
        infer_start = time.perf_counter()

        proc_frame = frame
        h, w = proc_frame.shape[:2]
        if w > self.infer_imgsz:
            scale = self.infer_imgsz / float(w)
            proc_frame = cv2.resize(proc_frame, (self.infer_imgsz, int(h * scale)), interpolation=cv2.INTER_AREA)

        result = self.model.predict(
            source=proc_frame,
            conf=0.45,
            iou=0.5,
            imgsz=self.infer_imgsz,
            device=self.device,
            half=self.use_half,
            verbose=False,
        )[0]

        detections = sv.Detections.from_ultralytics(result)
        if len(detections) > 0:
            widths = detections.xyxy[:, 2] - detections.xyxy[:, 0]
            heights = detections.xyxy[:, 3] - detections.xyxy[:, 1]
            keep_mask = (widths * heights) > 480
            detections = detections[keep_mask]

        annotated = self.box_annotator.annotate(scene=proc_frame.copy(), detections=detections)

        gray = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY)
        confs = detections.confidence if detections.confidence is not None else []

        labels: List[str] = []
        width_values: List[float] = []
        area_values: List[float] = []
        depth_values: List[float] = []
        conf_values: List[float] = []
        estimate_conf_values: List[float] = []

        for idx, xyxy in enumerate(detections.xyxy):
            width_cm, _length_cm, area_m2, depth_cm, d_class, est_conf = self._estimate_metrics(gray, xyxy, proc_frame.shape)
            conf = float(confs[idx]) if len(confs) > idx else 0.0

            labels.append(
                f"conf {conf:.2f} | w {width_cm:.1f}cm | area {area_m2:.2f}m2 | depth {depth_cm:.1f}cm ({d_class})"
            )

            width_values.append(width_cm)
            area_values.append(area_m2)
            depth_values.append(depth_cm)
            conf_values.append(conf)
            estimate_conf_values.append(est_conf)

        annotated = self.label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

        infer_ms = (time.perf_counter() - infer_start) * 1000.0
        self.infer_ms_ema = self._ema(self.infer_ms_ema, infer_ms, 0.2)

        infer_fps = 1000.0 / max(infer_ms, 1e-6)
        self.infer_fps_ema = self._ema(self.infer_fps_ema, infer_fps, 0.2)

        avg_depth = float(np.mean(depth_values)) if depth_values else 0.0

        with self.lock:
            self.stats.update(
                {
                    "detections": int(len(detections)),
                    "avg_confidence": float(np.mean(conf_values)) if conf_values else 0.0,
                    "avg_width_cm": float(np.mean(width_values)) if width_values else 0.0,
                    "avg_area_m2": float(np.mean(area_values)) if area_values else 0.0,
                    "avg_depth_cm": avg_depth,
                    "depth_class": depth_class(avg_depth) if depth_values else "none",
                    "estimation_confidence": float(np.mean(estimate_conf_values)) if estimate_conf_values else 0.0,
                    "infer_fps": self.infer_fps_ema,
                }
            )

        return annotated

    def _open_capture(self, mode: str, source: str) -> Optional[cv2.VideoCapture]:
        if mode == "stream":
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        else:
            cap = cv2.VideoCapture(source)

        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 30)
        return cap if cap.isOpened() else None

    def start(self) -> Tuple[bool, str]:
        self.stop()

        with self.lock:
            mode = self.mode
            source = self.source_path
            stream_url = self.stream_source_url

        cap: Optional[cv2.VideoCapture] = None
        if mode in {"upload", "stream"}:
            target_source = stream_url if mode == "stream" else source
            if mode == "stream" and not target_source:
                return False, "No RTSP/IP stream URL configured."

            cap = self._open_capture(mode, target_source)
            if cap is None:
                return False, "Could not open selected stream source."

        with self.lock:
            self.capture = cap
            self.running = True
            self.status = "running"
            self.status_message = (
                "Phone stream active. Open /video on your phone and tap Start Camera."
                if mode == "phone"
                else "Low-latency streaming and inference active"
            )
            self.stop_event.clear()
            self.latest_frame = None
            self.latest_frame_id = 0
            self.last_processed_id = -1
            self.prev_capture_ts = 0.0
            self.prev_stream_ts = 0.0
            self.phone_last_frame_ts = 0.0
            self.latest_jpeg_id = 0

        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True) if mode in {"upload", "stream"} else None
        self.infer_thread = threading.Thread(target=self._infer_loop, daemon=True)
        if self.capture_thread is not None:
            self.capture_thread.start()
        self.infer_thread.start()

        return True, "Detection started"

    def stop(self) -> None:
        with self.lock:
            was_running = self.running
            self.running = False
            self.status = "stopped" if was_running else self.status
            self.status_message = "Stopped" if was_running else self.status_message
            self.stop_event.set()

        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.2)
        if self.infer_thread and self.infer_thread.is_alive():
            self.infer_thread.join(timeout=1.2)

        with self.lock:
            if self.capture is not None:
                self.capture.release()
            self.capture = None
            self.capture_thread = None
            self.infer_thread = None

    def _capture_loop(self) -> None:
        while not self.stop_event.is_set():
            with self.lock:
                cap = self.capture
                mode = self.mode

            if cap is None:
                break

            if mode == "stream":
                # Drain stale frames quickly for live sources so inference always sees the newest frame.
                for _ in range(2):
                    cap.grab()

            ok, frame = cap.read()
            now = time.perf_counter()

            if not ok:
                if mode == "upload":
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                time.sleep(0.01)
                continue

            max_width = self.max_capture_width
            h, w = frame.shape[:2]
            if w > max_width:
                scale = max_width / float(w)
                frame = cv2.resize(frame, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)

            if self.prev_capture_ts > 0:
                capture_fps = 1.0 / max(now - self.prev_capture_ts, 1e-6)
                self.capture_fps_ema = self._ema(self.capture_fps_ema, capture_fps, 0.15)
            self.prev_capture_ts = now

            with self.lock:
                self.latest_frame = frame
                self.latest_capture_ts = now
                self.latest_frame_id += 1
                self.stats["capture_fps"] = self.capture_fps_ema

    def _infer_loop(self) -> None:
        while not self.stop_event.is_set():
            with self.lock:
                frame = None if self.latest_frame is None else self.latest_frame.copy()
                frame_id = self.latest_frame_id
                capture_ts = self.latest_capture_ts
                running = self.running

            if not running:
                time.sleep(0.02)
                continue

            if frame is None or frame_id == self.last_processed_id:
                time.sleep(0.004)
                continue

            target_gap = max(1, int(self.infer_ms_ema / 24.0))
            if frame_id - self.last_processed_id < target_gap:
                time.sleep(0.003)
                continue

            annotated = self._annotate_and_stats(frame)
            ok, buffer = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
            if not ok:
                continue

            now = time.perf_counter()
            if self.prev_stream_ts > 0:
                stream_fps = 1.0 / max(now - self.prev_stream_ts, 1e-6)
                self.stream_fps_ema = self._ema(self.stream_fps_ema, stream_fps, 0.2)
            self.prev_stream_ts = now

            latency_ms = (now - capture_ts) * 1000.0

            # Skip publishing stale frames when pipeline delay exceeds target budget.
            if latency_ms > (self.target_latency_ms * 2.0):
                self.last_processed_id = frame_id
                continue

            with self.lock:
                self.latest_jpeg = buffer.tobytes()
                self.latest_jpeg_id += 1
                self.last_processed_id = frame_id
                self.stats["stream_fps"] = self.stream_fps_ema
                self.stats["latency_ms"] = latency_ms

    def generate_stream(self):
        last_jpeg_id = -1
        while True:
            with self.lock:
                payload = self.latest_jpeg
                payload_id = self.latest_jpeg_id

            if payload_id == last_jpeg_id:
                time.sleep(0.004)
                continue

            last_jpeg_id = payload_id

            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + payload + b"\r\n"

    def get_snapshot(self) -> Dict[str, object]:
        with self.lock:
            if self.running and self.mode == "phone":
                idle_for = time.perf_counter() - self.phone_last_frame_ts if self.phone_last_frame_ts > 0 else 999.0
                if idle_for > 2.0:
                    self.status = "warning"
                    self.status_message = "Waiting for phone camera frames from /video"
            snapshot = {
                "running": self.running,
                "mode": self.mode,
                "status": self.status,
                "status_message": self.status_message,
                "source_label": self.source_label,
                "phone_endpoint": "/video",
                "device": "GPU (CUDA)" if self.device.startswith("cuda") else "CPU",
                "calibrated": bool(self.reference_cm and self.reference_px),
            }
            snapshot.update(self.stats)
            return snapshot

    def set_mode(self, mode: str) -> None:
        if mode not in {"upload", "phone", "stream"}:
            raise ValueError("Invalid mode")

        with self.lock:
            self.mode = mode
            if mode == "upload":
                self.source_label = Path(self.source_path).name
                self.status_message = "Upload mode selected"
            elif mode == "phone":
                self.source_label = "Phone Browser Stream (/video)"
                self.status_message = "Phone mode selected. Open /video on your phone."
            else:
                self.source_label = self.stream_source_url or "RTSP/IP Stream"
                self.status_message = "RTSP/IP stream mode selected"

    def set_stream_source(self, stream_url: str) -> None:
        with self.lock:
            self.stream_source_url = stream_url
            self.mode = "stream"
            self.source_label = stream_url
            self.status_message = "Selected RTSP/IP stream source"

    def set_upload(self, file_path: str, display_name: str) -> None:
        with self.lock:
            self.source_path = file_path
            self.mode = "upload"
            self.source_label = display_name
            self.status_message = f"Selected file: {display_name}"

    def push_phone_frame(self, frame: np.ndarray) -> None:
        now = time.perf_counter()
        max_width = self.max_phone_width
        h, w = frame.shape[:2]
        if w > max_width:
            scale = max_width / float(w)
            frame = cv2.resize(frame, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)

        with self.lock:
            if self.prev_capture_ts > 0:
                capture_fps = 1.0 / max(now - self.prev_capture_ts, 1e-6)
                self.capture_fps_ema = self._ema(self.capture_fps_ema, capture_fps, 0.15)
            self.prev_capture_ts = now
            self.phone_last_frame_ts = now
            self.latest_frame = frame
            self.latest_capture_ts = now
            self.latest_frame_id += 1
            self.stats["capture_fps"] = self.capture_fps_ema
            if self.running and self.mode == "phone":
                self.status = "running"
                self.status_message = "Streaming from phone camera"

    def set_calibration(self, ref_cm: Optional[float], ref_px: Optional[float]) -> None:
        with self.lock:
            self.reference_cm = ref_cm
            self.reference_px = ref_px


engine = DetectionEngine()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video")
def phone_video():
    return render_template("video.html")


@app.route("/upload_video", methods=["POST"])
def upload_video():
    file = request.files.get("video")
    if not file or file.filename == "":
        return jsonify({"ok": False, "error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"ok": False, "error": "Unsupported format. Use mp4/avi/mov/mkv."}), 400

    safe_name = secure_filename(file.filename)
    file_path = UPLOAD_DIR / f"{int(time.time())}_{safe_name}"
    file.save(str(file_path))

    engine.stop()
    engine.set_upload(str(file_path), safe_name)

    return jsonify({"ok": True, "filename": safe_name})


@app.route("/video_frame", methods=["POST"])
def video_frame():
    file = request.files.get("frame")
    frame_bytes = file.read() if file is not None else request.get_data(cache=False)
    if not frame_bytes:
        return jsonify({"ok": False, "error": "Empty frame."}), 400

    np_data = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"ok": False, "error": "Could not decode frame."}), 400

    engine.push_phone_frame(frame)
    return ("", 204)


@app.route("/select_mode", methods=["POST"])
def select_mode():
    payload = request.get_json(silent=True) or {}
    mode = str(payload.get("mode", "upload"))

    if mode not in {"upload", "phone", "stream"}:
        return jsonify({"ok": False, "error": "Invalid mode."}), 400

    engine.stop()
    engine.set_mode(mode)
    return jsonify({"ok": True, "mode": mode})


@app.route("/set_stream_source", methods=["POST"])
def set_stream_source():
    payload = request.get_json(silent=True) or {}
    stream_url = str(payload.get("stream_url", "")).strip()

    if not stream_url:
        return jsonify({"ok": False, "error": "Stream URL is required."}), 400

    if not stream_url.startswith(("rtsp://", "http://", "https://")):
        return jsonify({"ok": False, "error": "Use an rtsp:// or http(s):// stream URL."}), 400

    engine.stop()
    engine.set_stream_source(stream_url)
    return jsonify({"ok": True, "stream_url": stream_url})


@app.route("/set_calibration", methods=["POST"])
def set_calibration():
    payload = request.get_json(silent=True) or {}
    ref_cm_raw = payload.get("reference_cm")
    ref_px_raw = payload.get("reference_px")

    if ref_cm_raw in (None, "") or ref_px_raw in (None, ""):
        engine.set_calibration(None, None)
        return jsonify({"ok": True, "message": "Calibration reset to heuristic mode."})

    try:
        ref_cm = float(ref_cm_raw)
        ref_px = float(ref_px_raw)
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Calibration values must be numeric."}), 400

    if ref_cm <= 0 or ref_px <= 0:
        return jsonify({"ok": False, "error": "Calibration values must be > 0."}), 400

    engine.set_calibration(ref_cm, ref_px)
    return jsonify({"ok": True, "message": "Calibration enabled."})


@app.route("/start_detection", methods=["POST"])
def start_detection():
    ok, message = engine.start()
    if not ok:
        return jsonify({"ok": False, "error": message}), 400
    return jsonify({"ok": True, "message": message, "video_url": "/video_feed"})


@app.route("/stop_detection", methods=["POST"])
def stop_detection():
    engine.stop()
    return jsonify({"ok": True})


@app.route("/video_feed")
def video_feed():
    return Response(engine.generate_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/detection_stats")
def detection_stats():
    return jsonify(engine.get_snapshot())


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True, use_reloader=False)
