"""
Main detection pipeline.

Ties everything together: camera → detector → tracker → classifier → storage → database.

Supports multiple deployment modes:
- Development (Mac/PC): Ultralytics YOLO + PyTorch MobileNetV2 on MPS/CUDA/CPU
- Production (Raspberry Pi): Hailo NPU for both detection and classification
- Demo mode: Process a single image or video file instead of live camera
"""

import logging
import signal
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from config.settings import settings
from src.backend.database import (
    create_tables,
    get_session_factory,
    Detection,
    Species,
    load_species_from_dataset,
)
from src.backend.storage import ImageStorage
from src.inference.classifier import BirdClassifier
from src.inference.tracker import BirdTracker, crop_bird_roi
from src.pipeline.camera import RTSPCamera, FrameSkipper
from src.inference.detector import BirdDetector

logger = logging.getLogger(__name__)


class BirdPipeline:
    """
    Main pipeline orchestrating the full detection → classification flow.

    The pipeline loop:
    1. Grab frame from camera
    2. Run YOLO to find bird bounding boxes
    3. Update tracker (deduplicates birds sitting on feeder)
    4. For new tracks: crop bird, classify species, save image, log to DB
    """

    def __init__(
        self,
        detector: BirdDetector,
        classifier: BirdClassifier,
        camera: RTSPCamera | None = None,
        tracker: BirdTracker | None = None,
        storage: ImageStorage | None = None,
        process_every_n: int = 5,
    ):
        self.detector = detector
        self.classifier = classifier
        self.camera = camera
        self.tracker = tracker or BirdTracker()
        self.storage = storage or ImageStorage()
        self.skipper = FrameSkipper(process_every_n=process_every_n)

        self._running = False
        self._total_detections = 0
        self._session_factory = None

    def _init_database(self):
        """Initialize database tables and species lookup."""
        engine = create_tables()
        self._session_factory = get_session_factory(engine)

        # Load species from NABirds if not already in DB
        classes_file = settings.data_dir / "nabirds" / "classes.txt"
        if classes_file.exists():
            with self._session_factory() as session:
                load_species_from_dataset(session, classes_file)

    def _save_detection(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        species_name: str,
        confidence: float,
        timestamp: datetime,
    ):
        """Save a detection to storage and database."""
        # Save crop and thumbnail
        image_path, thumbnail_path = self.storage.save_detection(
            frame=frame,
            bbox=bbox,
            species_name=species_name,
            confidence=confidence,
            timestamp=timestamp,
        )

        # Find species in database
        with self._session_factory() as session:
            species = session.query(Species).filter(
                Species.common_name == species_name
            ).first()

            x1, y1, x2, y2 = bbox
            detection = Detection(
                timestamp=timestamp,
                species_id=species.id if species else None,
                confidence=confidence,
                detection_model="yolov8n",
                classifier_model="mobilenetv2_nabirds",
                bbox_x1=float(x1),
                bbox_y1=float(y1),
                bbox_x2=float(x2),
                bbox_y2=float(y2),
                image_path=image_path,
                thumbnail_path=thumbnail_path,
            )
            session.add(detection)
            session.commit()

        self._total_detections += 1
        logger.info(f"Detection #{self._total_detections}: {species_name} ({confidence:.2f})")

    def process_frame(self, frame: np.ndarray, timestamp: float | None = None) -> list[dict]:
        """
        Process a single frame through the full pipeline.

        Args:
            frame: BGR numpy array.
            timestamp: Frame timestamp (defaults to now).

        Returns:
            List of new detections as dicts with species, confidence, bbox.
        """
        timestamp = timestamp or time.time()
        new_detections = []

        # 1. Detect birds
        bboxes, confidences = self.detector.detect(frame)

        # 2. Update tracker
        new_tracks = self.tracker.update(bboxes, timestamp)

        # 3. Classify new tracks
        for track in new_tracks:
            crop = crop_bird_roi(frame, track.bbox)
            if crop.size == 0:
                continue

            species_name, conf = self.classifier.predict(crop)
            if species_name is None:
                continue

            # Update tracker with classification result
            self.tracker.mark_classified(track.track_id, species_name, conf)

            # Save to storage and database
            dt = datetime.fromtimestamp(timestamp)
            if self._session_factory and settings.save_crops:
                self._save_detection(frame, track.bbox, species_name, conf, dt)

            new_detections.append({
                "species": species_name,
                "confidence": conf,
                "bbox": track.bbox,
                "track_id": track.track_id,
            })

        return new_detections

    def run(self):
        """
        Run the live camera pipeline loop.

        Press Ctrl+C to stop gracefully.
        """
        if self.camera is None:
            raise RuntimeError("No camera configured. Use run_on_image() for single images.")

        self._init_database()
        self._running = True

        # Graceful shutdown on Ctrl+C
        def signal_handler(sig, frame):
            logger.info("Shutting down pipeline...")
            self._running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        logger.info("Starting bird detection pipeline...")
        logger.info(f"  Detector: {self.detector.backend}")
        logger.info(f"  Classifier: {self.classifier.backend}")
        logger.info(f"  Processing every {self.skipper.process_every_n} frames")

        self.camera.start()
        frames_processed = 0

        try:
            while self._running:
                frame_result = self.camera.get_frame()
                if frame_result is None:
                    time.sleep(0.01)
                    continue

                if not self.skipper.should_process(frame_result.frame_number):
                    continue

                detections = self.process_frame(frame_result.frame, frame_result.timestamp)
                frames_processed += 1

                if detections:
                    for det in detections:
                        print(f"  Bird: {det['species']} ({det['confidence']:.2f})")

        finally:
            self.camera.stop()
            logger.info(
                f"Pipeline stopped. Processed {frames_processed} frames, "
                f"{self._total_detections} birds detected."
            )

    def run_on_image(self, image_path: str | Path) -> list[dict]:
        """
        Run the pipeline on a single image file. Useful for testing.

        Args:
            image_path: Path to an image file.

        Returns:
            List of detections.
        """
        import cv2

        frame = cv2.imread(str(image_path))
        if frame is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")

        self._init_database()

        # Use min_frames=1 so detections are immediate
        self.tracker = BirdTracker(min_frames_for_detection=1)
        return self.process_frame(frame)

    @property
    def total_detections(self) -> int:
        return self._total_detections

    @property
    def is_running(self) -> bool:
        return self._running


def create_pipeline_dev(
    checkpoint_path: str | Path = "models/checkpoints/efficientnet_b2/best_model.pth",
    class_names: dict[int, str] | None = None,
    rtsp_url: str | None = None,
    device: str | None = None,
) -> BirdPipeline:
    """
    Create a pipeline for local development (Mac/PC).

    Uses Ultralytics YOLO + PyTorch classifier on MPS/CUDA/CPU.
    """
    detector = BirdDetector.from_ultralytics(device=device)

    # Load class names from dataset if not provided
    if class_names is None:
        class_names = _load_class_names()

    classifier = BirdClassifier.from_pytorch(
        checkpoint_path=checkpoint_path,
        class_names=class_names,
        device=device,
    )

    camera = RTSPCamera(rtsp_url=rtsp_url) if rtsp_url else None

    return BirdPipeline(detector=detector, classifier=classifier, camera=camera)


def create_pipeline_hailo(
    detection_hef: str | Path | None = None,
    classifier_hef: str | Path | None = None,
    class_names: dict[int, str] | None = None,
    rtsp_url: str | None = None,
) -> BirdPipeline:
    """
    Create a pipeline for Raspberry Pi with Hailo NPU.

    Uses Hailo HEF models for both detection and classification.
    """
    detector = BirdDetector.from_hailo(hef_path=detection_hef)

    if class_names is None:
        class_names = _load_class_names()

    classifier = BirdClassifier.from_hailo(
        hef_path=classifier_hef or settings.classifier_model_path,
        class_names=class_names,
    )

    camera = RTSPCamera(rtsp_url=rtsp_url)

    return BirdPipeline(detector=detector, classifier=classifier, camera=camera)


def _load_class_names() -> dict[int, str]:
    """Load class names from the NABirds dataset for prediction display."""
    from src.training.dataset import NABirdsDataset
    from src.training.transforms import get_val_transforms

    data_dir = settings.data_dir / "nabirds"
    if not data_dir.exists():
        logger.warning(f"NABirds data not found at {data_dir}, class names unavailable")
        return {}

    dataset = NABirdsDataset(data_dir, split="train", transform=get_val_transforms())
    return dataset.class_to_species


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Bird Feeder AI Pipeline")
    parser.add_argument(
        "--mode", choices=["dev", "hailo"], default="dev",
        help="Deployment mode: 'dev' (Mac/PC) or 'hailo' (Raspberry Pi)",
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Run on a single image instead of live camera",
    )
    parser.add_argument(
        "--checkpoint", type=str, default="models/checkpoints/efficientnet_b2/best_model.pth",
        help="Path to trained model checkpoint (dev mode)",
    )
    parser.add_argument(
        "--rtsp-url", type=str, default=None,
        help="RTSP camera URL (overrides config)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Force device: 'mps', 'cuda', or 'cpu'",
    )
    args = parser.parse_args()

    if args.mode == "dev":
        pipeline = create_pipeline_dev(
            checkpoint_path=args.checkpoint,
            rtsp_url=args.rtsp_url,
            device=args.device,
        )
    else:
        pipeline = create_pipeline_hailo(rtsp_url=args.rtsp_url)

    if args.image:
        results = pipeline.run_on_image(args.image)
        if results:
            for r in results:
                print(f"{r['species']} ({r['confidence']:.2f}) at {r['bbox']}")
        else:
            print("No birds detected.")
    else:
        pipeline.run()
