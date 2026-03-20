# Pothole Detection System (Flask + YOLO + Supervision)

This project provides a real-time pothole detection dashboard with dual video source support, live analytics, and approximate size/depth estimation.

## Stack
- Backend: Flask (`app.py`)
- Detection model: Ultralytics YOLO with custom weights (`best.pt`)
- Annotation: Roboflow Supervision
- Frontend: HTML/CSS/JS dashboard (`templates/index.html`)

## Features
- Dual source support:
  - Uploaded local video
  - Phone browser camera uplink via `/video`
  - External RTSP/IP stream URL mode
- Threaded low-latency pipeline:
  - Separate capture and inference loops
  - Capture buffer trimming for live streams (latest-frame priority)
  - Adaptive frame skipping based on inference time
  - Pre-inference frame resizing for speed
  - Adaptive JPEG quality + frame pacing for phone uplink
- GPU acceleration when available (`CUDA`) with automatic CPU fallback
- Start/Stop detection controls from UI
- Clear mode identification in UI:
  - `Video Upload Mode`
  - `Phone Browser Mode`
  - Active mode badge + disabled irrelevant controls
- Real-time stats polling (`/detection_stats`) for:
  - Capture FPS, inference FPS, stream FPS, latency
  - Detection confidence and pothole dimension estimates
- Estimated pothole metrics:
  - Width (cm)
  - Area (m2)
  - Depth (cm)
  - Severity class: `shallow` / `medium` / `severe`
- Optional calibration inputs for better real-world scaling:
  - Reference size in cm
  - Reference size in pixels

## Run
```bash
python app.py
```
Open: `http://127.0.0.1:5000`

## Low-Latency Tuning
You can tune runtime performance using environment variables:

- `POTHOLE_MODEL_PATH` (default: `best.pt`)
- `INFER_IMGSZ` (default: `512`)
- `MAX_CAPTURE_WIDTH` (default: `800`)
- `MAX_PHONE_WIDTH` (default: `720`)
- `STREAM_JPEG_QUALITY` (default: `72`)
- `TARGET_LATENCY_MS` (default: `450`)

Example (PowerShell):

```powershell
$env:INFER_IMGSZ = "448"
$env:MAX_CAPTURE_WIDTH = "720"
$env:STREAM_JPEG_QUALITY = "68"
python app.py
```

Notes:
- For fastest response, prefer `RTSP/IP Mode` on the dashboard over browser uplink.
- Keep both devices on the same 5 GHz Wi-Fi network.
- Use GPU/CUDA when available.

## API Endpoints
- `POST /upload_video`: upload video file (`video` form field)
- `POST /select_mode`: `{ "mode": "upload" | "phone" | "stream" }`
- `POST /set_stream_source`: `{ "stream_url": "rtsp://..." }`
- `GET /video`: phone camera capture page
- `POST /video_frame`: receives JPEG frames from phone page
- `POST /start_detection`: starts stream and returns stream URL
- `POST /stop_detection`: stops current session
- `POST /set_calibration`: `{ "reference_cm": number, "reference_px": number }` or reset with nulls
- `GET /video_feed`: MJPEG stream
- `GET /detection_stats`: live detection + estimation stats

## Accuracy Notes
- Without calibration, size/depth values are heuristic estimates based on perspective and texture cues.
- With calibration values, size estimation reliability improves significantly.
- Reaching 80-90% physical measurement accuracy generally requires controlled camera calibration and/or depth hardware.
