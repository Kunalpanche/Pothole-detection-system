from __future__ import annotations

import io
import os
import queue
import threading
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse
from functools import wraps

import cv2
import numpy as np
import supervision as sv
import torch
from flask import Flask, Response, jsonify, render_template, request, send_from_directory, session, redirect, url_for
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
from ultralytics import YOLO
from ultralytics.nn.modules.block import AAttn
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "pothole_detection_secret_key_2024")

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
SCREENSHOT_DIR = BASE_DIR / "screenshots"
SCREENSHOT_DIR.mkdir(exist_ok=True)

MODEL_PATH = os.getenv("POTHOLE_MODEL_PATH", str(BASE_DIR / "best.pt"))
DEFAULT_VIDEO = str(BASE_DIR / "demo2.mp4")
INFER_IMGSZ = int(os.getenv("INFER_IMGSZ", "512"))
MAX_CAPTURE_WIDTH = int(os.getenv("MAX_CAPTURE_WIDTH", "800"))
MAX_PHONE_WIDTH = int(os.getenv("MAX_PHONE_WIDTH", "720"))
STREAM_JPEG_QUALITY = int(os.getenv("STREAM_JPEG_QUALITY", "72"))
TARGET_LATENCY_MS = float(os.getenv("TARGET_LATENCY_MS", "450"))
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "512"))

app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

# Demo credentials
DEMO_CREDENTIALS = {
    "email": "admin@acubeai",
    "password": "admin123"
}

SAMPLE_VIDEOS_DIR = BASE_DIR / "sample_videos"

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function


def _patch_aattn_compat() -> None:
    # Some checkpoints contain legacy AAttn modules with qk/v attributes instead of qkv.
    # Newer Ultralytics expects qkv, which crashes during inference unless we handle both.
    if getattr(AAttn, "_compat_patched", False):
        return

    def compat_forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W

        if hasattr(self, "qkv"):
            qkv = self.qkv(x).flatten(2).transpose(1, 2)
            if self.area > 1:
                qkv = qkv.reshape(B * self.area, N // self.area, C * 3)
                B, N, _ = qkv.shape
            q, k, v = (
                qkv.view(B, N, self.num_heads, self.head_dim * 3)
                .permute(0, 2, 3, 1)
                .split([self.head_dim, self.head_dim, self.head_dim], dim=2)
            )
        else:
            qk = self.qk(x).flatten(2).transpose(1, 2)
            v_raw = self.v(x).flatten(2).transpose(1, 2)
            if self.area > 1:
                qk = qk.reshape(B * self.area, N // self.area, C * 2)
                v_raw = v_raw.reshape(B * self.area, N // self.area, C)
                B, N, _ = qk.shape
            q, k = (
                qk.view(B, N, self.num_heads, self.head_dim * 2)
                .permute(0, 2, 3, 1)
                .split([self.head_dim, self.head_dim], dim=2)
            )
            v = v_raw.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 3, 1)

        attn = (q.transpose(-2, -1) @ k) * (self.head_dim**-0.5)
        attn = attn.softmax(dim=-1)
        out = v @ attn.transpose(-2, -1)
        out = out.permute(0, 3, 1, 2)
        v_map = v.permute(0, 3, 1, 2)

        if self.area > 1:
            out = out.reshape(B // self.area, N * self.area, C)
            v_map = v_map.reshape(B // self.area, N * self.area, C)
            B, N, _ = out.shape

        out = out.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        v_map = v_map.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        out = out + self.pe(v_map)
        return self.proj(out)

    AAttn.forward = compat_forward
    AAttn._compat_patched = True


_patch_aattn_compat()


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
        self.frame_queue: queue.Queue[Tuple[np.ndarray, float, float]] = queue.Queue(maxsize=4)

        self.mode = "upload"
        self.source_path = DEFAULT_VIDEO
        self.source_label = Path(DEFAULT_VIDEO).name
        self.stream_source_url = ""
        self.phone_last_frame_ts = 0.0
        self.upload_paused = False
        self.upload_speed = 1.0
        self.upload_source_fps = 0.0

        self.reference_cm: Optional[float] = None
        self.reference_px: Optional[float] = None

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
        self.last_error = ""
        self.session_started_ts = time.time()
        self.session_first_detection_ts: Optional[float] = None
        self.total_potholes = 0
        self.detection_log: List[Dict[str, object]] = []
        self.max_log_rows = 5000
        self.log_seq = 0

        # IOU-based deduplication tracker
        # Each entry: {"xyxy": np.ndarray, "last_seen": float, "pothole_id": int}
        self._tracked_potholes: List[Dict] = []
        self._track_iou_threshold = 0.5
        self._track_expiry_seconds = 2.0  # forget a pothole after 2s without re-detection
        self._next_pothole_uid = 0

        self.stats: Dict[str, object] = {
            "detections": 0,
            "total_potholes": 0,
            "detections_per_min": 0.0,
            "avg_confidence": 0.0,
            "avg_length_cm": 0.0,
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

    @staticmethod
    def _format_timeline(seconds: float) -> str:
        total_seconds = max(0, int(round(seconds)))
        hh = total_seconds // 3600
        mm = (total_seconds % 3600) // 60
        ss = total_seconds % 60
        return f"{hh:02d}:{mm:02d}:{ss:02d}"

    def _make_status_frame(self, text: str) -> bytes:
        frame = np.full((480, 854, 3), 246, dtype=np.uint8)
        cv2.putText(frame, text, (30, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (45, 45, 45), 2)
        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return buffer.tobytes() if ok else b""

    def _reset_frame_queue(self) -> None:
        while True:
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

    def _enqueue_frame(self, frame: np.ndarray, timeline_s: float, capture_ts: float) -> None:
        payload = (frame, timeline_s, capture_ts)
        try:
            self.frame_queue.put(payload, block=True, timeout=0.5)
        except queue.Full:
            # Only drop the oldest frame if the queue is truly stuck
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.frame_queue.put(payload, block=False)
            except queue.Full:
                pass

    def _save_screenshot(self, frame: np.ndarray, total_count: int) -> str:
        timestamp_ms = int(time.time() * 1000)
        filename = f"pothole_{total_count}_{timestamp_ms}.jpg"
        path = SCREENSHOT_DIR / filename
        ok = cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        return filename if ok else ""

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

    @staticmethod
    def _compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
        """Compute Intersection over Union between two [x1,y1,x2,y2] boxes."""
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
        area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _is_duplicate_pothole(self, xyxy: np.ndarray, now: float) -> bool:
        """Check if this detection matches any tracked pothole (IOU > threshold).
        If matched, update the tracked entry's last_seen. If not, register as new.
        Returns True if duplicate (already tracked), False if new.
        """
        # Expire old tracked potholes
        self._tracked_potholes = [
            t for t in self._tracked_potholes
            if (now - t["last_seen"]) < self._track_expiry_seconds
        ]

        # Check against existing tracked potholes
        for tracked in self._tracked_potholes:
            iou = self._compute_iou(xyxy, tracked["xyxy"])
            if iou > self._track_iou_threshold:
                # Same pothole — update position and timestamp
                tracked["xyxy"] = xyxy.copy()
                tracked["last_seen"] = now
                return True

        # New pothole — register it
        self._next_pothole_uid += 1
        self._tracked_potholes.append({
            "xyxy": xyxy.copy(),
            "last_seen": now,
            "pothole_id": self._next_pothole_uid,
        })
        return False

    def _annotate_and_stats(self, frame: np.ndarray, timeline_s: float) -> np.ndarray:
        infer_start = time.perf_counter()

        proc_frame = frame
        h, w = proc_frame.shape[:2]
        if w > self.infer_imgsz:
            scale = self.infer_imgsz / float(w)
            proc_frame = cv2.resize(proc_frame, (self.infer_imgsz, int(h * scale)), interpolation=cv2.INTER_AREA)

        with torch.no_grad():
            result = self.model.predict(
                source=proc_frame,
                conf=0.35,
                iou=0.45,
                imgsz=self.infer_imgsz,
                device=self.device,
                half=self.use_half,
                verbose=False,
            )[0]

        detections = sv.Detections.from_ultralytics(result)
        if len(detections) > 0:
            widths = detections.xyxy[:, 2] - detections.xyxy[:, 0]
            heights = detections.xyxy[:, 3] - detections.xyxy[:, 1]
            keep_mask = (widths * heights) > 200
            detections = detections[keep_mask]

        annotated = self.box_annotator.annotate(scene=proc_frame.copy(), detections=detections)

        gray = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY)
        confs = detections.confidence if detections.confidence is not None else []

        labels: List[str] = []
        length_values: List[float] = []
        width_values: List[float] = []
        area_values: List[float] = []
        depth_values: List[float] = []
        conf_values: List[float] = []
        estimate_conf_values: List[float] = []
        log_rows: List[Dict[str, object]] = []

        timeline_display = self._format_timeline(timeline_s)

        for idx, xyxy in enumerate(detections.xyxy):
            width_cm, length_cm, area_m2, depth_cm, d_class, est_conf = self._estimate_metrics(gray, xyxy, proc_frame.shape)
            conf = float(confs[idx]) if len(confs) > idx else 0.0

            labels.append(
                f"conf {conf:.2f} | l {length_cm:.1f}cm | w {width_cm:.1f}cm | depth {depth_cm:.1f}cm ({d_class})"
            )

            length_values.append(length_cm)
            width_values.append(width_cm)
            area_values.append(area_m2)
            depth_values.append(depth_cm)
            conf_values.append(conf)
            estimate_conf_values.append(est_conf)

            log_rows.append(
                {
                    "video_time": timeline_display,
                    "video_time_seconds": round(float(timeline_s), 3),
                    "pothole_detected": "Yes",
                    "length_cm": round(length_cm, 2),
                    "width_cm": round(width_cm, 2),
                    "depth_cm": round(depth_cm, 2),
                    "depth_class": d_class,
                    "confidence": round(conf, 3),
                }
            )

        annotated = self.label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

        infer_ms = (time.perf_counter() - infer_start) * 1000.0
        self.infer_ms_ema = self._ema(self.infer_ms_ema, infer_ms, 0.2)

        infer_fps = 1000.0 / max(infer_ms, 1e-6)
        self.infer_fps_ema = self._ema(self.infer_fps_ema, infer_fps, 0.2)

        avg_depth = float(np.mean(depth_values)) if depth_values else 0.0
        avg_length = float(np.mean(length_values)) if length_values else 0.0
        now_epoch = time.time()

        with self.lock:
            detections_in_frame = int(len(detections))
            if detections_in_frame > 0 and self.session_first_detection_ts is None:
                self.session_first_detection_ts = now_epoch

            # Deduplicate: only log truly new potholes
            new_log_rows: List[Dict[str, object]] = []
            if detections_in_frame > 0:
                for idx_row, row in enumerate(log_rows):
                    xyxy_for_row = detections.xyxy[idx_row]
                    is_dup = self._is_duplicate_pothole(xyxy_for_row, now_epoch)
                    if not is_dup:
                        self.total_potholes += 1
                        self.log_seq += 1
                        row["id"] = self.log_seq
                        row["total_count"] = self.total_potholes
                        try:
                            shot_name = self._save_screenshot(annotated, self.total_potholes)
                        except Exception:
                            shot_name = ""
                        row["screenshot"] = f"/screenshots/{shot_name}" if shot_name else ""
                        new_log_rows.append(row)

            for row in new_log_rows:
                self.detection_log.append(row)

            if len(self.detection_log) > self.max_log_rows:
                self.detection_log = self.detection_log[-self.max_log_rows :]

            elapsed_minutes = max((now_epoch - self.session_started_ts) / 60.0, 1e-6)
            dpm = self.total_potholes / elapsed_minutes

            self.stats.update(
                {
                    "detections": detections_in_frame,
                    "total_potholes": self.total_potholes,
                    "detections_per_min": dpm,
                    "avg_confidence": float(np.mean(conf_values)) if conf_values else 0.0,
                    "avg_length_cm": avg_length,
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
        backends: List[int] = []
        if mode == "stream":
            backends.extend([cv2.CAP_FFMPEG, cv2.CAP_ANY])
            if hasattr(cv2, "CAP_MSMF"):
                backends.append(int(cv2.CAP_MSMF))
            if hasattr(cv2, "CAP_DSHOW"):
                backends.append(int(cv2.CAP_DSHOW))
        else:
            backends.extend([cv2.CAP_ANY, cv2.CAP_FFMPEG])

        tried = set()
        for backend in backends:
            if backend in tried:
                continue
            tried.add(backend)
            cap = cv2.VideoCapture(source, backend)
            if not cap.isOpened():
                cap.release()
                continue

            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if mode != "upload":
                cap.set(cv2.CAP_PROP_FPS, 30)
            return cap

        return None

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

            if mode == "upload" and (not target_source or not Path(target_source).exists()):
                return False, "Selected upload file was not found. Upload the video again."

            cap = self._open_capture(mode, target_source)
            if cap is None:
                if mode == "stream":
                    return False, "Could not open stream URL. Check URL, credentials, and network connectivity."
                return False, "Could not open uploaded video file. Verify file integrity and supported format."

        with self.lock:
            self.capture = cap
            self.running = True
            self.status = "running"
            self.status_message = (
                "Phone stream active. Open /video on your phone and tap Start Camera."
                if mode == "phone"
                else "Low-latency streaming and inference active"
            )
            self.last_error = ""
            self.stop_event.clear()
            self.prev_capture_ts = 0.0
            self.prev_stream_ts = 0.0
            self.phone_last_frame_ts = 0.0
            self.latest_jpeg_id = 0
            self.upload_paused = False
            self.upload_source_fps = 0.0
            self.session_started_ts = time.time()
            self.session_first_detection_ts = None
            self.total_potholes = 0
            self.log_seq = 0
            self.detection_log = []
            self._tracked_potholes = []
            self._next_pothole_uid = 0
            self.stats["detections"] = 0
            self.stats["total_potholes"] = 0
            self.stats["detections_per_min"] = 0.0
            self._reset_frame_queue()

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
        consecutive_failures = 0
        last_reconnect_ts = 0.0

        # Read source FPS for upload mode to sync playback
        source_fps = 0.0
        with self.lock:
            cap_init = self.capture
            mode_init = self.mode
        if cap_init is not None and mode_init == "upload":
            source_fps = float(cap_init.get(cv2.CAP_PROP_FPS) or 0.0)
            if source_fps <= 0 or source_fps > 120:
                source_fps = 25.0  # sensible fallback
            with self.lock:
                self.upload_source_fps = source_fps

        prev_frame_time = time.perf_counter()

        while not self.stop_event.is_set():
            with self.lock:
                cap = self.capture
                mode = self.mode
                paused = self.upload_paused
                speed = self.upload_speed
                stream_url = self.stream_source_url

            if cap is None:
                break

            if mode == "upload" and paused:
                time.sleep(0.03)
                prev_frame_time = time.perf_counter()
                continue

            if mode == "stream":
                # Drain stale frames quickly for live sources so inference always sees the newest frame.
                for _ in range(2):
                    cap.grab()

            ok, frame = cap.read()
            now = time.perf_counter()
            timeline_s = max(0.0, time.time() - self.session_started_ts)

            if not ok:
                if mode == "upload":
                    with self.lock:
                        self.running = False
                        self.status = "stopped"
                        self.status_message = "Upload playback completed"
                    break

                if mode == "stream":
                    consecutive_failures += 1
                    with self.lock:
                        self.status = "warning"
                        self.status_message = "Stream interrupted. Reconnecting..."
                        self.last_error = "Stream frame read failed"

                    if consecutive_failures >= 10 and (now - last_reconnect_ts) > 1.2:
                        last_reconnect_ts = now
                        replacement = self._open_capture("stream", stream_url)
                        if replacement is not None:
                            with self.lock:
                                if self.capture is not None:
                                    self.capture.release()
                                self.capture = replacement
                                self.status = "running"
                                self.status_message = "Stream reconnected"
                                self.last_error = ""
                            consecutive_failures = 0
                            continue
                time.sleep(0.01)
                continue

            consecutive_failures = 0

            max_width = self.max_capture_width
            h, w = frame.shape[:2]
            if w > max_width:
                scale = max_width / float(w)
                frame = cv2.resize(frame, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)

            if mode == "upload":
                pos_msec = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
                if pos_msec > 0:
                    timeline_s = pos_msec / 1000.0

            if self.prev_capture_ts > 0:
                capture_fps = 1.0 / max(now - self.prev_capture_ts, 1e-6)
                self.capture_fps_ema = self._ema(self.capture_fps_ema, capture_fps, 0.15)
            self.prev_capture_ts = now

            with self.lock:
                self.stats["capture_fps"] = self.capture_fps_ema

            self._enqueue_frame(frame, timeline_s, now)

            # FPS-synced sleep for upload mode so video plays at correct speed
            if mode == "upload" and source_fps > 0:
                target_interval = 1.0 / (source_fps * max(speed, 0.1))
                elapsed = time.perf_counter() - prev_frame_time
                sleep_time = target_interval - elapsed
                if sleep_time > 0.001:
                    time.sleep(sleep_time)
                prev_frame_time = time.perf_counter()
                continue

    def _infer_loop(self) -> None:
        while not self.stop_event.is_set():
            with self.lock:
                running = self.running

            if not running:
                time.sleep(0.02)
                continue

            try:
                frame, timeline_s, capture_ts = self.frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                annotated = self._annotate_and_stats(frame, timeline_s)
            except Exception as exc:
                annotated = frame
                with self.lock:
                    self.status = "warning"
                    self.status_message = "Inference degraded. Streaming raw frames."
                    self.last_error = f"Inference error: {type(exc).__name__}"

            ok, buffer = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
            if not ok:
                continue

            now = time.perf_counter()
            if self.prev_stream_ts > 0:
                stream_fps = 1.0 / max(now - self.prev_stream_ts, 1e-6)
                self.stream_fps_ema = self._ema(self.stream_fps_ema, stream_fps, 0.2)
            self.prev_stream_ts = now

            latency_ms = (now - capture_ts) * 1000.0

            with self.lock:
                self.latest_jpeg = buffer.tobytes()
                self.latest_jpeg_id += 1
                self.stats["stream_fps"] = self.stream_fps_ema
                self.stats["latency_ms"] = latency_ms

    def generate_stream(self):
        last_jpeg_id = -1
        while True:
            with self.lock:
                payload = self.latest_jpeg
                payload_id = self.latest_jpeg_id

            if payload_id == last_jpeg_id:
                time.sleep(0.008)
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
                    self.last_error = "No phone camera frames received"
            snapshot = {
                "running": self.running,
                "mode": self.mode,
                "status": self.status,
                "status_message": self.status_message,
                "last_error": self.last_error,
                "source_label": self.source_label,
                "phone_endpoint": "/video",
                "device": "GPU (CUDA)" if self.device.startswith("cuda") else "CPU",
                "calibrated": bool(self.reference_cm and self.reference_px),
                "upload_paused": self.upload_paused,
                "upload_speed": self.upload_speed,
            }
            snapshot.update(self.stats)
            return snapshot

    def set_upload_controls(self, paused: Optional[bool], speed: Optional[float]) -> None:
        with self.lock:
            if paused is not None:
                self.upload_paused = bool(paused)
                if self.mode == "upload" and self.running:
                    self.status_message = "Upload paused" if self.upload_paused else "Upload playing"

            if speed is not None:
                self.upload_speed = float(np.clip(speed, 0.1, 2.0))

    def get_log_slice(self, since_id: int = 0, limit: int = 200) -> Dict[str, object]:
        with self.lock:
            rows = [row for row in self.detection_log if int(row.get("id", 0)) > since_id]
            if len(rows) > limit:
                rows = rows[-limit:]
            last_id = int(self.detection_log[-1]["id"]) if self.detection_log else 0
            return {"rows": rows, "last_id": last_id, "total_rows": len(self.detection_log)}

    def build_report_data(self) -> Dict[str, object]:
        with self.lock:
            yes_rows = [r for r in self.detection_log if r.get("pothole_detected") == "Yes"]
            depth_counts = {"shallow": 0, "medium": 0, "severe": 0}
            for row in yes_rows:
                d_class = str(row.get("depth_class", "none")).lower()
                if d_class in depth_counts:
                    depth_counts[d_class] += 1

            avg_length = float(np.mean([float(r.get("length_cm", 0.0)) for r in yes_rows])) if yes_rows else 0.0
            avg_width = float(np.mean([float(r.get("width_cm", 0.0)) for r in yes_rows])) if yes_rows else 0.0
            avg_depth = float(np.mean([float(r.get("depth_cm", 0.0)) for r in yes_rows])) if yes_rows else 0.0

            timestamps = [str(r.get("video_time", "")) for r in yes_rows]
            first_ts = timestamps[0] if timestamps else "N/A"
            last_ts = timestamps[-1] if timestamps else "N/A"

            return {
                "rows": list(self.detection_log),
                "yes_rows": yes_rows,
                "total_potholes": self.total_potholes,
                "avg_length": avg_length,
                "avg_width": avg_width,
                "avg_depth": avg_depth,
                "depth_counts": depth_counts,
                "first_ts": first_ts,
                "last_ts": last_ts,
                "source_label": self.source_label,
            }

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
            self.last_error = ""

    def set_stream_source(self, stream_url: str) -> None:
        with self.lock:
            self.stream_source_url = stream_url
            self.mode = "stream"
            self.source_label = stream_url
            self.status_message = "Selected RTSP/IP stream source"
            self.last_error = ""

    def set_upload(self, file_path: str, display_name: str) -> None:
        with self.lock:
            self.source_path = file_path
            self.mode = "upload"
            self.source_label = display_name
            self.status_message = f"Selected file: {display_name}"
            self.last_error = ""

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
            self.stats["capture_fps"] = self.capture_fps_ema
            if self.running and self.mode == "phone":
                self.status = "running"
                self.status_message = "Streaming from phone camera"
                self.last_error = ""

        timeline_s = max(0.0, time.time() - self.session_started_ts)
        self._enqueue_frame(frame, timeline_s, now)

    def set_calibration(self, ref_cm: Optional[float], ref_px: Optional[float]) -> None:
        with self.lock:
            self.reference_cm = ref_cm
            self.reference_px = ref_px


engine = DetectionEngine()


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()
        
        if email == DEMO_CREDENTIALS["email"] and password == DEMO_CREDENTIALS["password"]:
            session["user"] = email
            return redirect(url_for("index"))
        else:
            return render_template("login.html", error="Invalid email or password")
    
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
def index():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("index.html")


@app.route("/video")
@login_required
def phone_video():
    return render_template("video.html")


@app.route("/screenshots/<path:filename>")
@login_required
def screenshot_file(filename: str):
    return send_from_directory(str(SCREENSHOT_DIR), filename)


@app.route("/upload_video", methods=["POST"])
@login_required
def upload_video():
    file = request.files.get("video")
    if not file or file.filename == "":
        return jsonify({"ok": False, "error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"ok": False, "error": "Unsupported format. Use mp4/avi/mov/mkv."}), 400

    safe_name = secure_filename(file.filename)
    file_path = UPLOAD_DIR / f"{int(time.time())}_{safe_name}"
    try:
        file.save(str(file_path))
    except OSError:
        return jsonify({"ok": False, "error": "Unable to save upload. Check disk space and permissions."}), 500

    verifier = cv2.VideoCapture(str(file_path))
    valid_video = verifier.isOpened()
    verifier.release()
    if not valid_video:
        try:
            file_path.unlink(missing_ok=True)
        except OSError:
            pass
        return jsonify({"ok": False, "error": "Uploaded file is not a readable video."}), 400

    engine.stop()
    engine.set_upload(str(file_path), safe_name)

    return jsonify({"ok": True, "filename": safe_name})


@app.route("/video_frame", methods=["POST"])
@login_required
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
@login_required
def select_mode():
    payload = request.get_json(silent=True) or {}
    mode = str(payload.get("mode", "upload"))

    if mode not in {"upload", "phone", "stream"}:
        return jsonify({"ok": False, "error": "Invalid mode."}), 400

    engine.stop()
    engine.set_mode(mode)
    return jsonify({"ok": True, "mode": mode})


@app.route("/set_stream_source", methods=["POST"])
@login_required
def set_stream_source():
    payload = request.get_json(silent=True) or {}
    stream_url = str(payload.get("stream_url", "")).strip()

    if not stream_url:
        return jsonify({"ok": False, "error": "Stream URL is required."}), 400

    parsed = urlparse(stream_url)
    if parsed.scheme not in {"rtsp", "http", "https"} or not parsed.netloc:
        return jsonify({"ok": False, "error": "Use an rtsp:// or http(s):// stream URL."}), 400

    engine.stop()
    engine.set_stream_source(stream_url)
    return jsonify({"ok": True, "stream_url": stream_url})


@app.route("/set_calibration", methods=["POST"])
@login_required
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


@app.route("/set_upload_playback", methods=["POST"])
@login_required
def set_upload_playback():
    payload = request.get_json(silent=True) or {}

    paused_raw = payload.get("paused")
    speed_raw = payload.get("speed")

    paused: Optional[bool] = None
    speed: Optional[float] = None

    if paused_raw is not None:
        paused = bool(paused_raw)

    if speed_raw is not None:
        try:
            speed = float(speed_raw)
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "Speed must be numeric."}), 400

    engine.set_upload_controls(paused=paused, speed=speed)
    return jsonify({"ok": True})


@app.route("/detection_log")
@login_required
def detection_log():
    since_id_raw = request.args.get("since_id", "0")
    try:
        since_id = int(since_id_raw)
    except ValueError:
        since_id = 0

    payload = engine.get_log_slice(since_id=since_id, limit=300)
    return jsonify({"ok": True, **payload})


def _draw_report_pdf(report: Dict[str, object]) -> bytes:
    rows = report["rows"]
    depth_counts = report["depth_counts"]

    pdf_buffer = io.BytesIO()
    pdf = canvas.Canvas(pdf_buffer, pagesize=A4)
    page_w, page_h = A4

    def draw_header(y: float) -> float:
        pdf.setFillColor(colors.HexColor("#1f2937"))
        pdf.setFont("Helvetica-Bold", 16)
        pdf.drawString(2 * cm, y, "Pothole Detection Report")
        y -= 0.5 * cm
        pdf.setFont("Helvetica", 10)
        pdf.setFillColor(colors.HexColor("#4b5563"))
        pdf.drawString(2 * cm, y, f"Source: {report['source_label']}")
        return y - 0.6 * cm

    y = draw_header(page_h - 2 * cm)

    pdf.setFont("Helvetica-Bold", 11)
    pdf.setFillColor(colors.black)
    pdf.drawString(2 * cm, y, "Key KPIs")
    y -= 0.45 * cm

    pdf.setFont("Helvetica", 10)
    pdf.drawString(2 * cm, y, f"Total potholes detected: {report['total_potholes']}")
    y -= 0.38 * cm
    pdf.drawString(
        2 * cm,
        y,
        f"Average size (L x W x D): {report['avg_length']:.2f} x {report['avg_width']:.2f} x {report['avg_depth']:.2f} cm",
    )
    y -= 0.38 * cm
    pdf.drawString(
        2 * cm,
        y,
        f"Depth distribution: shallow={depth_counts['shallow']} medium={depth_counts['medium']} severe={depth_counts['severe']}",
    )
    y -= 0.38 * cm
    pdf.drawString(2 * cm, y, f"Detection window: {report['first_ts']} to {report['last_ts']}")
    y -= 0.7 * cm

    headers = ["Time", "Detected", "Total", "L(cm)", "W(cm)", "D(cm)", "Class"]
    col_widths = [4.1 * cm, 2.0 * cm, 1.7 * cm, 1.8 * cm, 1.8 * cm, 1.8 * cm, 2.2 * cm]

    def draw_table_header(y_pos: float) -> float:
        x = 2 * cm
        pdf.setFillColor(colors.HexColor("#f4b400"))
        pdf.rect(x, y_pos - 0.35 * cm, sum(col_widths), 0.45 * cm, fill=1, stroke=0)
        pdf.setFillColor(colors.HexColor("#1f2937"))
        pdf.setFont("Helvetica-Bold", 9)
        for idx, head in enumerate(headers):
            pdf.drawString(x + 0.08 * cm, y_pos - 0.2 * cm, head)
            x += col_widths[idx]
        return y_pos - 0.5 * cm

    y = draw_table_header(y)

    pdf.setFont("Helvetica", 8.5)
    max_rows = min(len(rows), 800)
    rows_to_render = rows[:max_rows]
    for idx, row in enumerate(rows_to_render):
        if y < 1.8 * cm:
            pdf.showPage()
            y = draw_header(page_h - 1.5 * cm)
            y = draw_table_header(y)
            pdf.setFont("Helvetica", 8.5)

        x = 2 * cm
        if idx % 2 == 0:
            pdf.setFillColor(colors.HexColor("#f9fafb"))
            pdf.rect(x, y - 0.28 * cm, sum(col_widths), 0.36 * cm, fill=1, stroke=0)

        pdf.setFillColor(colors.HexColor("#111827"))
        values = [
            str(row.get("video_time", "")),
            str(row.get("pothole_detected", "")),
            str(row.get("total_count", "")),
            f"{float(row.get('length_cm', 0.0)):.1f}",
            f"{float(row.get('width_cm', 0.0)):.1f}",
            f"{float(row.get('depth_cm', 0.0)):.1f}",
            str(row.get("depth_class", "")),
        ]
        for col_idx, value in enumerate(values):
            pdf.drawString(x + 0.08 * cm, y - 0.16 * cm, value)
            x += col_widths[col_idx]

        y -= 0.36 * cm

    if len(rows) > max_rows:
        y -= 0.2 * cm
        pdf.setFont("Helvetica-Oblique", 8.5)
        pdf.setFillColor(colors.HexColor("#6b7280"))
        pdf.drawString(2 * cm, y, f"Note: Showing first {max_rows} rows of {len(rows)} total log entries.")

    pdf.save()
    return pdf_buffer.getvalue()


@app.route("/download_report")
@login_required
def download_report():
    report = engine.build_report_data()
    pdf_bytes = _draw_report_pdf(report)
    filename = f"pothole_report_{int(time.time())}.pdf"
    return Response(
        pdf_bytes,
        mimetype="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.route("/start_detection", methods=["POST"])
@login_required
def start_detection():
    ok, message = engine.start()
    if not ok:
        return jsonify({"ok": False, "error": message}), 400
    return jsonify({"ok": True, "message": message, "video_url": "/video_feed"})


@app.route("/stop_detection", methods=["POST"])
@login_required
def stop_detection():
    engine.stop()
    return jsonify({"ok": True})


@app.route("/video_feed")
@login_required
def video_feed():
    return Response(engine.generate_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/detection_stats")
@login_required
def detection_stats():
    return jsonify(engine.get_snapshot())


@app.route("/download_samples")
@login_required
def download_samples():
    """Download sample videos as a ZIP file"""
    try:
        # Create a ZIP file in memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # List of sample video files to include
            sample_videos = ["demo.mp4", "demo1.mp4", "demo2.mp4"]
            
            for video_name in sample_videos:
                video_path = SAMPLE_VIDEOS_DIR / video_name
                if video_path.exists():
                    # Add file to ZIP with just the filename (not full path)
                    zip_file.write(str(video_path), arcname=video_name)
        
        zip_buffer.seek(0)
        filename = f"sample_videos_{int(time.time())}.zip"
        return Response(
            zip_buffer.getvalue(),
            mimetype="application/zip",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.errorhandler(413)
def request_too_large(_err):
    return jsonify(
        {
            "ok": False,
            "error": f"Upload exceeds {MAX_UPLOAD_MB} MB limit. Compress the video or increase MAX_UPLOAD_MB.",
        }
    ), 413


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True, use_reloader=False)
