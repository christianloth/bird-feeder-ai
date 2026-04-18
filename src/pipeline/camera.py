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


def _build_rotate_fn(degrees: float):
    """Return a function that rotates a BGR frame by `degrees`.

    Positive = counterclockwise, negative = clockwise. Returns None when
    no rotation is needed so the caller can skip the work entirely.

    90/180/270 multiples use cv2.rotate (lossless, no interpolation);
    arbitrary angles use a full affine warp that expands the canvas so
    nothing is clipped (corners are filled with black).
    """
    angle = float(degrees) % 360.0
    if angle == 0.0:
        return None

    # Fast paths for cardinal rotations. cv2.ROTATE_90_COUNTERCLOCKWISE
    # matches "positive degrees = CCW" per the config convention.
    if angle == 90.0:
        return lambda f: cv2.rotate(f, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if angle == 180.0:
        return lambda f: cv2.rotate(f, cv2.ROTATE_180)
    if angle == 270.0:
        return lambda f: cv2.rotate(f, cv2.ROTATE_90_CLOCKWISE)

    # Arbitrary angle — build an affine matrix lazily per frame size so
    # different-sized inputs still work. Cache by (h, w) to avoid rebuilding.
    import math
    cache: dict[tuple[int, int], tuple[np.ndarray, tuple[int, int]]] = {}

    def rotate(frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        cached = cache.get((h, w))
        if cached is None:
            # cv2.getRotationMatrix2D: positive angle is CCW, matches our convention.
            center = (w / 2.0, h / 2.0)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            cos = abs(matrix[0, 0])
            sin = abs(matrix[0, 1])
            new_w = int(math.ceil(h * sin + w * cos))
            new_h = int(math.ceil(h * cos + w * sin))
            matrix[0, 2] += (new_w / 2.0) - center[0]
            matrix[1, 2] += (new_h / 2.0) - center[1]
            cached = (matrix, (new_w, new_h))
            cache[(h, w)] = cached
        matrix, size = cached
        return cv2.warpAffine(frame, matrix, size, flags=cv2.INTER_LINEAR)

    return rotate


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
        rotation_degrees: float | None = None,
    ):
        """
        Args:
            rtsp_url: RTSP stream URL. Defaults to config value.
            reconnect_delay: Seconds to wait before reconnecting on failure.
            max_reconnect_attempts: Max reconnection attempts (0 = unlimited).
            rotation_degrees: Rotate frames by this angle. Positive values are
                counterclockwise, negative are clockwise. Defaults to config
                value. 0 disables rotation.
        """
        self.rtsp_url = rtsp_url or settings.rtsp_url
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts
        rotation = (
            rotation_degrees
            if rotation_degrees is not None
            else settings.rotation_degrees
        )
        self._rotate_fn = _build_rotate_fn(rotation)
        if self._rotate_fn is not None:
            logger.info(f"Frame rotation enabled: {rotation}°")

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

    def _build_gstreamer_pipeline(self) -> str:
        """Build a GStreamer pipeline string for H.265 RTSP via software decode.

        The v4l2slh265dec hardware decoder outputs tiled NV12_128C8 via DMABuf
        and the glupload/gldownload GL path silently produced all-black frames
        on this setup. Falling back to avdec_h265 (libav software decode) gives
        correct pixel data; the tradeoff is higher CPU use than the broken HW
        path, but still benefits from GStreamer's threading vs CAP_FFMPEG.
        """
        return (
            f'rtspsrc location="{self.rtsp_url}" latency=200 protocols=tcp ! '
            "rtph265depay ! h265parse ! avdec_h265 ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=true max-buffers=1 sync=false"
        )

    def connect(self) -> bool:
        """Open the RTSP stream. Returns True if successful."""
        if self._cap is not None:
            self._cap.release()

        if settings.camera_codec == "h265":
            # H.265 via GStreamer (software decode via avdec_h265).
            # HW decode path via v4l2slh265dec silently produced black frames.
            pipeline = self._build_gstreamer_pipeline()
            logger.info("Using H.265 software decode (GStreamer + avdec_h265)")
            logger.debug(f"GStreamer pipeline: {pipeline}")
            self._cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        else:
            # H.264 via FFMPEG — software decode on CPU
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            logger.info("Using H.264 software decode (FFMPEG)")
            self._cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if self._cap.isOpened():
            # GStreamer may not report dimensions until first frame is read
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

            if self._rotate_fn is not None:
                frame = self._rotate_fn(frame)

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
            # Safe to return the reference directly because _grab_loop always
            # rebinds self._frame to a fresh ndarray (cv2.read + cv2.rotate
            # both return new arrays) and never mutates the previous one.
            # Downstream consumers MUST NOT mutate this frame in place.
            frame = self._frame
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
