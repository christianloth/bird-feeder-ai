"""
RTSP camera capture module.

Handles connecting to the SV3C 4K PTZ camera via RTSP, reading frames,
and providing them to the detection pipeline. Includes automatic reconnection
on stream failure.
"""

import logging
import os
import time
import threading
from dataclasses import dataclass

import cv2
import numpy as np

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class FrameResult:
    """A captured frame with metadata."""
    frame: np.ndarray
    timestamp: float
    frame_number: int
    width: int
    height: int


class RTSPCamera:
    """
    Captures frames from an RTSP stream using OpenCV.

    Uses a background thread to continuously grab frames so the main
    pipeline always gets the latest frame without buffering delay.
    """

    def __init__(
        self,
        rtsp_url: str | None = None,
        reconnect_delay: float = 5.0,
        max_reconnect_attempts: int = 0,
    ):
        """
        Args:
            rtsp_url: RTSP stream URL. Defaults to config value.
            reconnect_delay: Seconds to wait before reconnecting on failure.
            max_reconnect_attempts: Max reconnection attempts (0 = unlimited).
        """
        self.rtsp_url = rtsp_url or settings.rtsp_url
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts

        self._cap: cv2.VideoCapture | None = None
        self._frame: np.ndarray | None = None
        self._frame_lock = threading.Lock()
        self._frame_number = 0
        self._running = False
        self._thread: threading.Thread | None = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def frame_number(self) -> int:
        return self._frame_number

    def connect(self) -> bool:
        """Open the RTSP stream. Returns True if successful."""
        if self._cap is not None:
            self._cap.release()

        # Use FFMPEG backend for RTSP — more reliable than GStreamer for capture
        # Force TCP transport to avoid UDP packet loss and h264 decode errors
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        self._cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)

        # Set buffer size to 1 to always get the latest frame
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if self._cap.isOpened():
            width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self._cap.get(cv2.CAP_PROP_FPS)
            logger.info(f"Connected to RTSP stream: {width}x{height} @ {fps:.1f} FPS")
            logger.debug(f"RTSP URL: {self.rtsp_url}")
            self._connected = True
            return True

        logger.error(f"Failed to open RTSP stream: {self.rtsp_url}")
        self._connected = False
        return False

    def _grab_loop(self):
        """Background thread that continuously grabs frames."""
        reconnect_attempts = 0

        while self._running:
            if not self._connected:
                if self.max_reconnect_attempts > 0 and reconnect_attempts >= self.max_reconnect_attempts:
                    logger.critical(
                        f"Camera unreachable after {reconnect_attempts} attempts. "
                        "Stopping capture thread."
                    )
                    self._running = False
                    break

                reconnect_attempts += 1
                logger.warning(
                    f"Reconnecting in {self.reconnect_delay}s "
                    f"(attempt {reconnect_attempts}/"
                    f"{'unlimited' if self.max_reconnect_attempts == 0 else self.max_reconnect_attempts})"
                )
                time.sleep(self.reconnect_delay)

                if self.connect():
                    reconnect_attempts = 0
                    logger.info("Reconnected to RTSP stream")
                continue

            ret, frame = self._cap.read()

            if not ret or frame is None:
                logger.warning("Failed to read frame, connection may be lost")
                self._connected = False
                continue

            with self._frame_lock:
                self._frame = frame
                self._frame_number += 1

        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def start(self) -> bool:
        """Start capturing frames in a background thread."""
        if self._running:
            logger.warning("Camera is already running.")
            return True

        if not self.connect():
            return False

        self._running = True
        self._thread = threading.Thread(target=self._grab_loop, daemon=True, name="rtsp-capture")
        self._thread.start()
        logger.info("RTSP capture thread started.")
        return True

    def stop(self):
        """Stop capturing and release resources."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._connected = False
        logger.info("RTSP capture stopped.")

    def get_frame(self) -> FrameResult | None:
        """
        Get the most recent frame from the camera.

        Returns None if no frame is available yet.
        """
        with self._frame_lock:
            if self._frame is None:
                return None
            frame = self._frame.copy()
            number = self._frame_number

        h, w = frame.shape[:2]
        return FrameResult(
            frame=frame,
            timestamp=time.time(),
            frame_number=number,
            width=w,
            height=h,
        )

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class FrameSkipper:
    """
    Utility to process only every Nth frame.

    For a bird feeder, processing every frame is wasteful.
    This lets you process e.g. every 5th frame (6 FPS from a 30 FPS stream).
    """

    def __init__(self, process_every_n: int = 5):
        self.process_every_n = process_every_n
        self._last_processed = -1

    def should_process(self, frame_number: int) -> bool:
        if frame_number - self._last_processed >= self.process_every_n:
            self._last_processed = frame_number
            return True
        return False
