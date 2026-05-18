"""
Object detection using YOLO models.

BirdDetector: YOLO detecting birds (COCO class 14) in camera frames.

Supports:
- Ultralytics PyTorch (MPS/CUDA/CPU) — for development
- Hailo NPU — production deployment on Raspberry Pi
"""

import logging
import time
from pathlib import Path

import numpy as np

from config.settings import settings

SLOW_INFERENCE_MS = 500  # Log a warning if inference takes longer than this

logger = logging.getLogger(__name__)


class BirdDetector:
    """
    Detects birds in camera frames using YOLO (model set in config.yaml).

    Filters detections to only COCO class 14 (bird) and applies
    a confidence threshold.

    Usage:
        detector = BirdDetector.from_ultralytics()
        bboxes, confidences = detector.detect(frame)
    """

    def __init__(
        self,
        backend: str,
        model=None,
        confidence_threshold: float | None = None,
        bird_class_id: int | None = None,
    ):
        self.backend = backend
        self._model = model
        self.confidence_threshold = confidence_threshold or settings.detection_confidence_threshold
        self.bird_class_id = bird_class_id or settings.bird_class_id

    @classmethod
    def from_ultralytics(
        cls,
        model_path: str | Path | None = None,
        device: str | None = None,
        confidence_threshold: float | None = None,
    ) -> "BirdDetector":
        """
        Load a YOLO model using the Ultralytics library.

        Works on MPS (Mac), CUDA (NVIDIA), or CPU.
        If model_path is not provided, uses detection.model from config.yaml.
        """
        from ultralytics import YOLO
        from config.settings import get_device

        model_path = model_path or settings.detection_model
        if not model_path:
            raise ValueError(
                "No detection model configured. Set detection.model in config/config.yaml"
            )
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Detection model not found: {model_path}")
        device = get_device(device)
        logger.debug(f"Loading YOLO detector from {model_path}")
        model = YOLO(str(model_path))
        model.to(device)

        logger.info(f"Loaded YOLO detector from {model_path} (device={device})")
        return cls(
            backend="ultralytics",
            model=model,
            confidence_threshold=confidence_threshold,
        )

    @classmethod
    def from_hailo(
        cls,
        hef_path: str | Path | None = None,
        confidence_threshold: float | None = None,
        vdevice=None,
    ) -> "BirdDetector":
        """Load YOLO HEF model for Hailo NPU inference."""
        from hailo_platform import VDevice, FormatType

        hef_path = hef_path or settings.detection_model_path
        if not Path(hef_path).exists():
            logger.critical(f"Detection model not found: {hef_path}")
            raise FileNotFoundError(f"Detection HEF model not found: {hef_path}")

        logger.debug(f"Loading Hailo YOLO detector from {hef_path}")
        if vdevice is None:
            vdevice = VDevice()
        infer_model = vdevice.create_infer_model(str(hef_path))
        infer_model.input().set_format_type(FormatType.UINT8)
        nms_threshold = confidence_threshold or settings.detection_confidence_threshold
        for output in infer_model.outputs:
            output.set_format_type(FormatType.FLOAT32)
            if output.is_nms:
                output.set_nms_score_threshold(nms_threshold)
                output.set_nms_iou_threshold(0.45)
        configured_model = infer_model.configure()

        input_shape = infer_model.input().shape  # (H, W, C)
        logger.info(
            f"Loaded Hailo YOLO detector from {hef_path} "
            f"(input={input_shape}, outputs={infer_model.output_names})"
        )
        return cls(
            backend="hailo",
            model={
                "vdevice": vdevice,
                "infer_model": infer_model,
                "configured": configured_model,
            },
            confidence_threshold=confidence_threshold,
        )

    def _detect_ultralytics(
        self, frame: np.ndarray,
    ) -> tuple[list[tuple[int, int, int, int]], list[float]]:
        """Run detection with Ultralytics backend."""
        results = self._model(
            frame,
            conf=self.confidence_threshold,
            classes=[self.bird_class_id],
            verbose=False,
        )

        bboxes = []
        confidences = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                bboxes.append((int(x1), int(y1), int(x2), int(y2)))
                confidences.append(conf)

        return bboxes, confidences

    def _detect_hailo(
        self, frame: np.ndarray,
    ) -> tuple[list[tuple[int, int, int, int]], list[float]]:
        """Run detection with Hailo NPU backend (InferModel API)."""
        import cv2

        infer_model = self._model["infer_model"]
        configured = self._model["configured"]
        input_shape = infer_model.input().shape  # (H, W, C)
        h, w = input_shape[0], input_shape[1]

        # Preprocess: resize to model input size, keep as uint8
        orig_h, orig_w = frame.shape[:2]
        resized = cv2.resize(frame, (w, h)).astype(np.uint8)

        # Create bindings and run inference
        bindings = configured.create_bindings()
        bindings.input().set_buffer(np.ascontiguousarray(resized))
        for name in infer_model.output_names:
            out_shape = infer_model.output(name).shape
            bindings.output(name).set_buffer(np.empty(out_shape, dtype=np.float32))
        configured.run([bindings], timeout=5000)

        # NMS output: list of 80 arrays (one per COCO class)
        # Each detection: [y1, x1, y2, x2, confidence] with normalized coords
        output_name = infer_model.output_names[0]
        nms_results = bindings.output(output_name).get_buffer()

        bboxes = []
        confidences = []

        # Only look at the bird class
        if self.bird_class_id < len(nms_results):
            for det in nms_results[self.bird_class_id]:
                conf = float(det[4])
                if conf < self.confidence_threshold:
                    continue
                # Normalized [y1, x1, y2, x2] -> pixel [x1, y1, x2, y2]
                y1 = int(det[0] * orig_h)
                x1 = int(det[1] * orig_w)
                y2 = int(det[2] * orig_h)
                x2 = int(det[3] * orig_w)
                bboxes.append((x1, y1, x2, y2))
                confidences.append(conf)

        return bboxes, confidences

    def detect(
        self, frame: np.ndarray,
    ) -> tuple[list[tuple[int, int, int, int]], list[float]]:
        """
        Detect birds in a frame.

        Args:
            frame: BGR numpy array from camera.

        Returns:
            (bboxes, confidences) — list of (x1, y1, x2, y2) boxes and their scores.
            Only birds (COCO class 14) above the confidence threshold are returned.
        """
        t0 = time.perf_counter()

        if self.backend == "ultralytics":
            bboxes, confidences = self._detect_ultralytics(frame)
        elif self.backend == "hailo":
            bboxes, confidences = self._detect_hailo(frame)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug(f"Bird detect: {len(bboxes)} found in {elapsed_ms:.1f}ms")
        if elapsed_ms > SLOW_INFERENCE_MS:
            logger.warning(f"Slow bird detection: {elapsed_ms:.0f}ms (threshold={SLOW_INFERENCE_MS}ms)")

        return bboxes, confidences

