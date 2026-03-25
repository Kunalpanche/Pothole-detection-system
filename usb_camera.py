"""USB Camera utility module for DroidCam / IVCam wired phone camera support.

This module probes available camera devices and helps the DetectionEngine
identify which camera index corresponds to the phone connected via USB.
"""

from __future__ import annotations

import platform
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def _preferred_backend() -> int:
    """Return the best OpenCV backend for the current OS."""
    if platform.system() == "Windows" and hasattr(cv2, "CAP_DSHOW"):
        return int(cv2.CAP_DSHOW)
    return int(cv2.CAP_ANY)


def list_available_cameras(max_index: int = 6) -> List[Dict[str, object]]:
    """Probe camera indices 0..max_index and return info on each working camera.

    Returns a list of dicts:
        [{"index": 0, "name": "Camera 0", "width": 640, "height": 480}, ...]
    """
    backend = _preferred_backend()
    cameras: List[Dict[str, object]] = []

    for idx in range(max_index):
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened():
            cap.release()
            continue

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        # Try to grab one frame to confirm it's truly usable
        ok, _ = cap.read()
        cap.release()

        if ok and w > 0 and h > 0:
            cameras.append({
                "index": idx,
                "name": f"Camera {idx}",
                "width": w,
                "height": h,
                "label": f"Camera {idx} ({w}x{h})",
            })

    return cameras


def test_camera(index: int, timeout_ms: int = 3000) -> Tuple[bool, Optional[np.ndarray], str]:
    """Test whether a specific camera index is accessible.

    Returns:
        (success, sample_frame_or_None, status_message)
    """
    backend = _preferred_backend()
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        return False, None, f"Camera index {index} could not be opened."

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Attempt a few reads (first frames can sometimes be blank)
    frame = None
    for _ in range(5):
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            break

    cap.release()

    if frame is None or not ok:
        return False, None, f"Camera {index} opened but returned no frames."

    h, w = frame.shape[:2]
    return True, frame, f"Camera {index} OK — {w}x{h}"


def find_usb_camera_index(skip_builtin: bool = True, max_index: int = 6) -> Optional[int]:
    """Auto-detect the USB webcam camera index.

    Heuristic: if skip_builtin is True, skip index 0 (usually the built-in
    laptop webcam) and return the first working index >= 1.
    """
    start = 1 if skip_builtin else 0
    cameras = list_available_cameras(max_index)
    for cam in cameras:
        if cam["index"] >= start:
            return cam["index"]
    return None


def open_usb_capture(index: int) -> Optional[cv2.VideoCapture]:
    """Open a USB camera capture at the given index with optimal settings."""
    backend = _preferred_backend()
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        return None

    # Optimise for low latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Try to set a reasonable resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Verify it actually works
    ok, _ = cap.read()
    if not ok:
        cap.release()
        return None

    return cap


def get_usb_status() -> Dict[str, object]:
    """Return a status dict suitable for the /usb_status API endpoint."""
    cameras = list_available_cameras()
    auto_index = find_usb_camera_index()

    return {
        "cameras": cameras,
        "camera_count": len(cameras),
        "suggested_index": auto_index,
        "usb_camera_detected": auto_index is not None,
        "instructions": (
            "Install DroidCam or IVCam on your phone and PC. "
            "Connect your phone via USB cable and start the app. "
            "The phone camera will appear as a standard webcam."
        ),
    }
