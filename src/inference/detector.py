"""
Object detection using YOLO models.

BirdDetector: YOLOv8n detecting birds (COCO class 14) in camera frames.
WildlifeDetector: Custom YOLO detecting 11 wildlife classes for nighttime.

Both support:
- Ultralytics PyTorch (MPS/CUDA/CPU) — for development
- Hailo NPU — production deployment on Raspberry Pi
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config.settings import settings

logger = logging.getLogger(__name__)


class BirdDetector:
    """
    Detects birds in camera frames using YOLOv8n.

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
        Load YOLOv8n using the Ultralytics library.

        Works on MPS (Mac), CUDA (NVIDIA), or CPU.
        If model_path is not provided, downloads yolov8n.pt automatically.
        """
        from ultralytics import YOLO

        model_path = model_path or "yolov8n.pt"
        model = YOLO(str(model_path))

        if device:
            model.to(device)

        logger.info(f"Loaded YOLO detector from {model_path}")
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
    ) -> "BirdDetector":
        """Load YOLOv8n HEF model for Hailo NPU inference."""
        from hailo_platform import HEF, VDevice, ConfigureParams

        hef_path = hef_path or settings.detection_model_path
        hef = HEF(str(hef_path))
        vdevice = VDevice()
        infer_model = vdevice.configure(hef, ConfigureParams.create_from_hef(hef))

        logger.info(f"Loaded Hailo YOLO detector from {hef_path}")
        return cls(
            backend="hailo",
            model={"hef": hef, "vdevice": vdevice, "infer_model": infer_model},
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
        """Run detection with Hailo NPU backend."""
        import cv2
        from hailo_platform import InferVStreams

        infer_model = self._model["infer_model"]
        input_info = infer_model.input_vstream_infos[0]
        h, w = input_info.shape[1], input_info.shape[2]

        # Preprocess: resize to model input size
        orig_h, orig_w = frame.shape[:2]
        resized = cv2.resize(frame, (w, h))
        input_data = np.expand_dims(resized, axis=0).astype(np.float32) / 255.0

        with InferVStreams(infer_model) as pipeline:
            results = pipeline.infer({input_info.name: input_data})

        # Parse YOLO output (format depends on Hailo post-processing config)
        output_name = infer_model.output_vstream_infos[0].name
        detections = results[output_name][0]

        bboxes = []
        confidences = []

        for det in detections:
            class_id = int(det[5]) if len(det) > 5 else int(det[4])
            conf = float(det[4]) if len(det) > 5 else float(det[5])

            if class_id != self.bird_class_id or conf < self.confidence_threshold:
                continue

            # Scale bounding box back to original frame size
            x1 = int(det[0] * orig_w / w)
            y1 = int(det[1] * orig_h / h)
            x2 = int(det[2] * orig_w / w)
            y2 = int(det[3] * orig_h / h)
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
        if self.backend == "ultralytics":
            return self._detect_ultralytics(frame)
        elif self.backend == "hailo":
            return self._detect_hailo(frame)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")


@dataclass
class WildlifeDetection:
    """A single wildlife detection with class information."""
    bbox: tuple[int, int, int, int]
    confidence: float
    class_name: str
    class_id: int


class WildlifeDetector:
    """
    Detects wildlife using a custom-trained YOLO model (11 classes).

    Used for nighttime inference when the camera switches to IR mode.
    Unlike BirdDetector, this returns all detected classes with their names
    rather than filtering to a single COCO class.

    Usage:
        detector = WildlifeDetector.from_ultralytics()
        detections = detector.detect(frame)
        for det in detections:
            print(f"{det.class_name} ({det.confidence:.2f}) at {det.bbox}")
    """

    def __init__(
        self,
        backend: str,
        model=None,
        confidence_threshold: float | None = None,
        class_names: dict[int, str] | None = None,
    ):
        self.backend = backend
        self._model = model
        self.confidence_threshold = (
            confidence_threshold or settings.wildlife_confidence_threshold
        )
        self.class_names = class_names or settings.wildlife_class_names

    @classmethod
    def from_ultralytics(
        cls,
        model_path: str | Path | None = None,
        device: str | None = None,
        confidence_threshold: float | None = None,
    ) -> "WildlifeDetector":
        """
        Load the wildlife YOLO model using Ultralytics.

        If model_path is not provided, uses the default from settings
        (models/wildlife/yolo11n-wildlife-equal/weights/best.pt).
        """
        from ultralytics import YOLO

        model_path = model_path or settings.wildlife_model_path
        model = YOLO(str(model_path))

        if device:
            model.to(device)

        logger.info(f"Loaded wildlife detector from {model_path}")
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
    ) -> "WildlifeDetector":
        """Load wildlife YOLO HEF model for Hailo NPU inference."""
        from hailo_platform import HEF, VDevice, ConfigureParams

        hef_path = hef_path or settings.wildlife_model_path
        hef = HEF(str(hef_path))
        vdevice = VDevice()
        infer_model = vdevice.configure(hef, ConfigureParams.create_from_hef(hef))

        logger.info(f"Loaded Hailo wildlife detector from {hef_path}")
        return cls(
            backend="hailo",
            model={"hef": hef, "vdevice": vdevice, "infer_model": infer_model},
            confidence_threshold=confidence_threshold,
        )

    def detect(self, frame: np.ndarray) -> list[WildlifeDetection]:
        """
        Detect wildlife in a frame.

        Args:
            frame: BGR numpy array from camera.

        Returns:
            List of WildlifeDetection objects with bbox, confidence, and class info.
        """
        if self.backend == "ultralytics":
            return self._detect_ultralytics(frame)
        elif self.backend == "hailo":
            return self._detect_hailo(frame)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _detect_ultralytics(self, frame: np.ndarray) -> list[WildlifeDetection]:
        """Run detection with Ultralytics backend."""
        results = self._model(
            frame,
            conf=self.confidence_threshold,
            verbose=False,
        )

        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self.class_names.get(class_id, f"class_{class_id}")
                detections.append(WildlifeDetection(
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    confidence=conf,
                    class_name=class_name,
                    class_id=class_id,
                ))

        return detections

    def _detect_hailo(self, frame: np.ndarray) -> list[WildlifeDetection]:
        """Run detection with Hailo NPU backend."""
        import cv2
        from hailo_platform import InferVStreams

        infer_model = self._model["infer_model"]
        input_info = infer_model.input_vstream_infos[0]
        h, w = input_info.shape[1], input_info.shape[2]

        orig_h, orig_w = frame.shape[:2]
        resized = cv2.resize(frame, (w, h))
        input_data = np.expand_dims(resized, axis=0).astype(np.float32) / 255.0

        with InferVStreams(infer_model) as pipeline:
            results = pipeline.infer({input_info.name: input_data})

        output_name = infer_model.output_vstream_infos[0].name
        raw_detections = results[output_name][0]

        detections = []
        for det in raw_detections:
            class_id = int(det[5]) if len(det) > 5 else int(det[4])
            conf = float(det[4]) if len(det) > 5 else float(det[5])

            if conf < self.confidence_threshold:
                continue

            x1 = int(det[0] * orig_w / w)
            y1 = int(det[1] * orig_h / h)
            x2 = int(det[2] * orig_w / w)
            y2 = int(det[3] * orig_h / h)

            class_name = self.class_names.get(class_id, f"class_{class_id}")
            detections.append(WildlifeDetection(
                bbox=(x1, y1, x2, y2),
                confidence=conf,
                class_name=class_name,
                class_id=class_id,
            ))

        return detections
