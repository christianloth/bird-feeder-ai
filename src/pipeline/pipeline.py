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
from src.inference.detector import BirdDetector, WildlifeDetector
from src.inference.tracker import BirdTracker, crop_bird_roi
from src.pipeline.camera import RTSPCamera, FrameSkipper
from src.backend.weather import WeatherService
from src.pipeline.mode_manager import DayNightManager

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
        wildlife_detector: WildlifeDetector | None = None,
        mode_manager: DayNightManager | None = None,
    ):
        self.detector = detector
        self.classifier = classifier
        self.wildlife_detector = wildlife_detector
        self.mode_manager = mode_manager
        self.camera = camera
        self.tracker = tracker or BirdTracker()
        self.storage = storage or ImageStorage()
        self.skipper = FrameSkipper(process_every_n=process_every_n)

        self._running = False
        self._total_detections = 0
        self._session_factory = None

    def _init_database(self):
        """Initialize database tables and species lookup."""
        logger.debug("Initializing database connection")
        engine = create_tables()
        self._session_factory = get_session_factory(engine)

        # Load species from NABirds if not already in DB
        classes_file = settings.data_dir / "nabirds" / "classes.txt"
        if classes_file.exists():
            with self._session_factory() as session:
                load_species_from_dataset(session, classes_file)
            logger.debug("Species lookup table loaded")
        else:
            logger.warning(f"NABirds classes file not found at {classes_file}")

    def _save_detection(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        species_name: str,
        confidence: float,
        timestamp: datetime,
        detection_model: str = "yolov8n",
        classifier_model: str = "mobilenetv2_nabirds",
    ):
        """Save a detection to storage and database."""
        try:
            image_path, thumbnail_path = self.storage.save_detection(
                frame=frame,
                bbox=bbox,
                species_name=species_name,
                confidence=confidence,
                timestamp=timestamp,
            )
        except OSError as e:
            logger.error(f"Failed to save detection image for {species_name}: {e}")
            return

        try:
            with self._session_factory() as session:
                species = session.query(Species).filter(
                    Species.common_name == species_name
                ).first()

                if species is None:
                    logger.warning(
                        f"Species '{species_name}' not found in database, "
                        "detection will have no species link"
                    )

                x1, y1, x2, y2 = bbox
                detection = Detection(
                    timestamp=timestamp,
                    species_id=species.id if species else None,
                    confidence=confidence,
                    detection_model=detection_model,
                    classifier_model=classifier_model,
                    bbox_x1=float(x1),
                    bbox_y1=float(y1),
                    bbox_x2=float(x2),
                    bbox_y2=float(y2),
                    image_path=image_path,
                    thumbnail_path=thumbnail_path,
                )
                session.add(detection)
                session.commit()
        except Exception as e:
            logger.error(f"Failed to save detection to database: {e}")
            return

        self._total_detections += 1
        logger.info(f"Detection #{self._total_detections}: {species_name} ({confidence:.2f})")

    def process_frame(self, frame: np.ndarray, timestamp: float | None = None) -> list[dict]:
        """
        Process a single frame through the full pipeline.

        In DAY mode: detect birds (COCO YOLO) -> track -> classify species.
        In NIGHT mode: detect wildlife (custom YOLO, 11 classes) -> track.

        Args:
            frame: BGR numpy array.
            timestamp: Frame timestamp (defaults to now).

        Returns:
            List of new detections as dicts with species, confidence, bbox.
        """
        timestamp = timestamp or time.time()

        # Check for day/night mode transition
        if self.mode_manager:
            mode_changed = self.mode_manager.update()
            if mode_changed:
                self.tracker.reset()
                logger.info(f"Tracker reset after mode switch to {self.mode_manager.mode.value}")

        if self._is_night_mode:
            return self._process_frame_night(frame, timestamp)
        return self._process_frame_day(frame, timestamp)

    @property
    def _is_night_mode(self) -> bool:
        """Check if we should use nighttime wildlife detection."""
        return (
            self.mode_manager is not None
            and self.wildlife_detector is not None
            and self.mode_manager.is_night
        )

    def _process_frame_day(
        self, frame: np.ndarray, timestamp: float,
    ) -> list[dict]:
        """Daytime path: bird detection + species classification."""
        new_detections = []

        # 1. Detect birds
        t0 = time.perf_counter()
        bboxes, confidences = self.detector.detect(frame)
        det_ms = (time.perf_counter() - t0) * 1000
        logger.debug(f"Detection: {len(bboxes)} birds in {det_ms:.1f}ms")

        # 2. Update tracker
        new_tracks = self.tracker.update(bboxes, timestamp)

        # 3. Classify new tracks
        for track in new_tracks:
            crop = crop_bird_roi(frame, track.bbox)
            if crop.size == 0:
                logger.warning(f"Empty crop for track {track.track_id}, skipping")
                continue

            t0 = time.perf_counter()
            species_name, conf = self.classifier.predict(crop)
            cls_ms = (time.perf_counter() - t0) * 1000
            logger.debug(
                f"Classification: {species_name} ({conf:.2f}) in {cls_ms:.1f}ms"
            )

            if species_name is None:
                continue

            self.tracker.mark_classified(track.track_id, species_name, conf)

            dt = datetime.fromtimestamp(timestamp)
            if self._session_factory and settings.save_crops:
                self._save_detection(
                    frame, track.bbox, species_name, conf, dt,
                    detection_model="yolov8n",
                    classifier_model="mobilenetv2_nabirds",
                )

            new_detections.append({
                "species": species_name,
                "confidence": conf,
                "bbox": track.bbox,
                "track_id": track.track_id,
            })

        return new_detections

    def _process_frame_night(
        self, frame: np.ndarray, timestamp: float,
    ) -> list[dict]:
        """Nighttime path: wildlife detection (class comes from YOLO directly)."""
        new_detections = []

        # 1. Detect wildlife -- returns class names directly
        t0 = time.perf_counter()
        wildlife_dets = self.wildlife_detector.detect(frame)
        det_ms = (time.perf_counter() - t0) * 1000
        logger.debug(f"Wildlife detection: {len(wildlife_dets)} animals in {det_ms:.1f}ms")

        bboxes = [d.bbox for d in wildlife_dets]

        # 2. Update tracker
        new_tracks = self.tracker.update(bboxes, timestamp)

        # 3. Match new tracks to their wildlife detections
        for track in new_tracks:
            matched_det = None
            for wd in wildlife_dets:
                if wd.bbox == track.bbox:
                    matched_det = wd
                    break

            if matched_det is None:
                logger.debug(f"No wildlife detection match for track {track.track_id}")
                continue

            species_name = matched_det.class_name
            conf = matched_det.confidence

            self.tracker.mark_classified(track.track_id, species_name, conf)

            dt = datetime.fromtimestamp(timestamp)
            if self._session_factory and settings.save_crops:
                self._save_detection(
                    frame, track.bbox, species_name, conf, dt,
                    detection_model="yolo11n-wildlife",
                    classifier_model="yolo11n-wildlife",
                )

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
            logger.critical("No camera configured. Cannot start pipeline.")
            raise RuntimeError("No camera configured. Use run_on_image() for single images.")

        self._init_database()
        self._running = True

        # Graceful shutdown on Ctrl+C
        def signal_handler(sig, frame):
            logger.info("Received shutdown signal (SIGINT/SIGTERM)")
            self._running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        logger.info("Starting bird detection pipeline...")
        logger.info(f"  Detector: {self.detector.backend}")
        logger.info(f"  Classifier: {self.classifier.backend}")
        if self.wildlife_detector:
            logger.info(f"  Wildlife detector: {self.wildlife_detector.backend}")
        if self.mode_manager:
            logger.info("  Day/night mode: sun-time switching enabled")
            self.mode_manager.check_now()
            logger.info(f"  Initial mode: {self.mode_manager.mode.value}")
        else:
            logger.info("  Day/night mode: disabled (daytime only)")
        logger.info(f"  Processing every {self.skipper.process_every_n} frames")

        if not self.camera.start():
            logger.critical("Failed to connect to camera. Pipeline cannot start.")
            return

        frames_processed = 0
        last_status_time = time.time()
        status_interval = 300  # Log a status summary every 5 minutes

        try:
            while self._running:
                frame_result = self.camera.get_frame()
                if frame_result is None:
                    time.sleep(0.01)
                    continue

                if not self.skipper.should_process(frame_result.frame_number):
                    continue

                try:
                    detections = self.process_frame(
                        frame_result.frame, frame_result.timestamp,
                    )
                except Exception as e:
                    logger.error(f"Frame processing failed: {e}", exc_info=True)
                    continue

                frames_processed += 1

                if detections:
                    for det in detections:
                        logger.info(
                            f"  {det['species']} ({det['confidence']:.2f}) "
                            f"track={det['track_id']}"
                        )

                # Periodic status summary
                now = time.time()
                if now - last_status_time >= status_interval:
                    mode_str = (
                        f", mode={self.mode_manager.mode.value}"
                        if self.mode_manager else ""
                    )
                    logger.info(
                        f"Status: {frames_processed} frames processed, "
                        f"{self._total_detections} total detections, "
                        f"{self.tracker.active_count} active tracks{mode_str}"
                    )
                    last_status_time = now

        finally:
            self.camera.stop()
            logger.info(
                f"Pipeline stopped. Processed {frames_processed} frames, "
                f"{self._total_detections} detections total."
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


def _create_night_mode_components(
    device: str | None = None,
    backend: str = "ultralytics",
) -> tuple[WildlifeDetector | None, DayNightManager | None]:
    """
    Create wildlife detector and day/night manager using sunrise/sunset times.

    Returns (None, None) if sun times can't be fetched, so the pipeline
    gracefully falls back to daytime-only mode.
    """
    weather = WeatherService()
    sun_times = weather.get_daily_sun_times()

    if sun_times is None:
        logger.warning(
            "Could not fetch sunrise/sunset times -- day/night mode switching "
            "disabled. Pipeline will run in daytime-only mode."
        )
        weather.close()
        return None, None

    logger.info(
        f"Sun times loaded: sunrise {sun_times['sunrise'].strftime('%H:%M')}, "
        f"sunset {sun_times['sunset'].strftime('%H:%M')}"
    )

    if backend == "ultralytics":
        wildlife_detector = WildlifeDetector.from_ultralytics(device=device)
    else:
        wildlife_detector = WildlifeDetector.from_hailo()

    mode_manager = DayNightManager(weather)

    return wildlife_detector, mode_manager


def create_pipeline_dev(
    checkpoint_path: str | Path = "models/bird-classifier/efficientnet_b2/best_model.pth",
    class_names: dict[int, str] | None = None,
    rtsp_url: str | None = None,
    device: str | None = None,
    enable_night_mode: bool = True,
) -> BirdPipeline:
    """
    Create a pipeline for local development (Mac/PC).

    Uses Ultralytics YOLO + PyTorch classifier on MPS/CUDA/CPU.
    If enable_night_mode is True, also loads the wildlife detector and
    sets up camera IR mode polling for automatic day/night switching.
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

    wildlife_detector = None
    mode_manager = None
    if enable_night_mode:
        wildlife_detector, mode_manager = _create_night_mode_components(
            device=device, backend="ultralytics",
        )

    return BirdPipeline(
        detector=detector,
        classifier=classifier,
        camera=camera,
        wildlife_detector=wildlife_detector,
        mode_manager=mode_manager,
    )


def create_pipeline_hailo(
    detection_hef: str | Path | None = None,
    classifier_hef: str | Path | None = None,
    class_names: dict[int, str] | None = None,
    rtsp_url: str | None = None,
    enable_night_mode: bool = True,
) -> BirdPipeline:
    """
    Create a pipeline for Raspberry Pi with Hailo NPU.

    Uses Hailo HEF models for both detection and classification.
    If enable_night_mode is True, also loads the wildlife detector and
    sets up camera IR mode polling for automatic day/night switching.
    """
    detector = BirdDetector.from_hailo(hef_path=detection_hef)

    if class_names is None:
        class_names = _load_class_names()

    classifier = BirdClassifier.from_hailo(
        hef_path=classifier_hef or settings.classifier_model_path,
        class_names=class_names,
    )

    camera = RTSPCamera(rtsp_url=rtsp_url)

    wildlife_detector = None
    mode_manager = None
    if enable_night_mode:
        wildlife_detector, mode_manager = _create_night_mode_components(
            backend="hailo",
        )

    return BirdPipeline(
        detector=detector,
        classifier=classifier,
        camera=camera,
        wildlife_detector=wildlife_detector,
        mode_manager=mode_manager,
    )


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
        "--checkpoint", type=str, default="models/bird-classifier/efficientnet_b2/best_model.pth",
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
    parser.add_argument(
        "--no-night", action="store_true",
        help="Disable night mode (wildlife detection) switching",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=None,
        help="Set logging verbosity (default: from .env or INFO)",
    )
    args = parser.parse_args()

    log_level = args.log_level or settings.log_level
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.mode == "dev":
        pipeline = create_pipeline_dev(
            checkpoint_path=args.checkpoint,
            rtsp_url=args.rtsp_url,
            device=args.device,
            enable_night_mode=not args.no_night,
        )
    else:
        pipeline = create_pipeline_hailo(
            rtsp_url=args.rtsp_url,
            enable_night_mode=not args.no_night,
        )

    if args.image:
        results = pipeline.run_on_image(args.image)
        if results:
            for r in results:
                print(f"{r['species']} ({r['confidence']:.2f}) at {r['bbox']}")
        else:
            print("No birds detected.")
    else:
        pipeline.run()
