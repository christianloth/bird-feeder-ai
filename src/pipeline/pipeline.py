"""
Main detection pipeline.

Ties everything together: camera → detector → tracker → classifier → storage → database.

Supports multiple deployment modes:
- Development (Mac/PC): Ultralytics YOLO + PyTorch ViT-Small on MPS/CUDA/CPU
- Production (Raspberry Pi): Hailo NPU for both detection and classification
- Demo mode: Process a single image or video file instead of live camera
"""

import logging
import signal
import sys
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
    load_wildlife_species,
    migrate_species_category,
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
        save_enabled: bool = True,
    ):
        self.detector = detector
        self.classifier = classifier
        self.wildlife_detector = wildlife_detector
        self.mode_manager = mode_manager
        self.camera = camera
        self.tracker = tracker or BirdTracker()
        self.storage = storage or ImageStorage()
        self.skipper = FrameSkipper(process_every_n=process_every_n)
        self.save_enabled = save_enabled
        self._source = "rtsp"  # Updated by run_on_image/run_on_video
        self._running = False
        self._total_detections = 0
        self._session_factory = None
        # Species cooldown: suppress duplicate DB saves of the same species
        # within N seconds. One bird visit = one database record.
        self._species_cooldown: dict[str, float] = {}  # species → last save timestamp

    def _log_pipeline_config(self) -> None:
        """Log pipeline configuration at startup."""
        from config.settings import get_device
        device = get_device()
        logger.info("Pipeline configuration:")
        logger.info(f"  Device: {device}")
        logger.info(f"  Detector: {settings.detection_model}")
        logger.info(f"  Classifier: {self.classifier.backend} ({self.classifier.device})")
        if self.wildlife_detector:
            logger.info(f"  Wildlife: {settings.wildlife_model}")
        if self.mode_manager:
            logger.info(f"  Day/night: enabled")
        else:
            logger.info(f"  Day/night: disabled (daytime only)")
        if self.save_enabled:
            logger.info("  Storage: enabled (database + detections/rtsp/)")
        elif self._source in ("image", "video"):
            logger.info("  Storage: off (use --output for annotated output)")
        else:
            logger.info("  Storage: DISABLED (--no-save)")

    def _init_database(self):
        """Initialize database tables and species lookup."""
        logger.debug("Initializing database connection")
        engine = create_tables()
        migrate_species_category(engine)
        self._session_factory = get_session_factory(engine)

        # Load species from NABirds if not already in DB
        nabirds_dir = settings.data_dir / "nabirds"
        if nabirds_dir.exists():
            with self._session_factory() as session:
                load_species_from_dataset(session, nabirds_dir)
            logger.debug("Species lookup table loaded")
        else:
            logger.warning(f"NABirds data not found at {nabirds_dir}")

        # Load wildlife species for night-mode detection
        with self._session_factory() as session:
            load_wildlife_species(session)
        logger.debug("Wildlife species lookup table loaded")

    def _save_detection(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        species_name: str,
        confidence: float,
        timestamp: datetime,
        detection_model: str = "yolov8n",
        classifier_model: str = "vit_small_nabirds",
        source: str = "rtsp",
    ):
        """Save a detection to storage and database, with species cooldown."""
        # Species cooldown: skip if we recently saved the same species.
        # This prevents one bird sitting on the feeder from generating 40
        # database records due to tracker fragmentation.
        cooldown = settings.species_cooldown_seconds
        ts_float = timestamp.timestamp()
        if species_name in self._species_cooldown:
            elapsed = ts_float - self._species_cooldown[species_name]
            if elapsed < cooldown:
                logger.debug(
                    f"Cooldown active for {species_name} "
                    f"({elapsed:.0f}s < {cooldown}s), skipping save"
                )
                return
        self._species_cooldown[species_name] = ts_float

        try:
            paths = self.storage.save_detection(
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
                    frame_path=paths["frame_path"],
                    source=source,
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
        """
        Daytime path: bird detection + species classification with logit averaging.

        Instead of classifying a bird once and dropping it if confidence is low,
        we classify on every processed frame and accumulate raw logits. Averaging
        logits before softmax produces better predictions than any single frame
        because consistent signal accumulates while per-frame noise cancels out.

        The track is reclassified each frame until confidence crosses the threshold,
        at which point it's locked in and saved to the database (RTSP mode only).
        """
        new_detections = []

        # 1. Detect birds (YOLO — "is there a bird?")
        t0 = time.perf_counter()
        bboxes, confidences = self.detector.detect(frame)
        det_ms = (time.perf_counter() - t0) * 1000
        if bboxes:
            confs_str = ", ".join(f"{c:.2f}" for c in confidences)
            logger.debug(f"Detection: {len(bboxes)} birds in {det_ms:.1f}ms (YOLO conf: [{confs_str}])")
        else:
            logger.debug(f"Detection: 0 birds in {det_ms:.1f}ms")

        # 2. Update tracker — returns all active tracks not yet confidently
        #    classified (both new tracks and existing ones being retried)
        tracks_to_classify = self.tracker.update(bboxes, confidences, timestamp)

        # 3. Classify each track using logit averaging
        for track in tracks_to_classify:
            crop = crop_bird_roi(frame, track.bbox)
            if crop.size == 0:
                logger.warning(f"Empty crop for track {track.track_id}, skipping")
                continue

            # Get raw logits (pre-softmax scores) for this frame's crop
            t0 = time.perf_counter()
            logits = self.classifier.predict_logits(crop)
            cls_ms = (time.perf_counter() - t0) * 1000

            # Average logits across all frames seen so far, then softmax.
            # This is more accurate than taking the max probability from any
            # single frame (Dussert et al. 2025).
            if track.logit_sum is not None:
                avg_logits = (track.logit_sum + logits) / (track.classify_count + 1)
            else:
                avg_logits = logits
            exp_logits = np.exp(avg_logits - np.max(avg_logits))
            probs = exp_logits / exp_logits.sum()
            class_idx = int(np.argmax(probs))
            conf = float(probs[class_idx])
            species_name = self.classifier.class_names.get(class_idx, f"class_{class_idx}")

            # Add this frame's logits to the track's running sum
            self.tracker.accumulate_logits(
                track.track_id, logits, species_name, conf,
            )

            logger.debug(
                f"Classification: {species_name} ({conf:.2f}) in {cls_ms:.1f}ms "
                f"[attempt {track.classify_count}, track seen {track.frame_count}x]"
            )

            # Check if we've crossed the confidence threshold
            if conf >= settings.classification_confidence_threshold:
                self.tracker.mark_classified(track.track_id, species_name, conf)
                logger.debug(
                    f"Track {track.track_id} confidently classified: "
                    f"{species_name} ({conf:.2f}) after {track.classify_count} frames"
                )

                dt = datetime.fromtimestamp(timestamp)
                if self.save_enabled and self._session_factory and settings.save_crops:
                    self._save_detection(
                        frame, track.bbox, species_name, conf, dt,
                        detection_model=Path(settings.detection_model).stem,
                        classifier_model=settings.classifier_model_path.stem,
                        source=self._source,
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
        """Nighttime path: wildlife detection with best-of-N retry."""
        new_detections = []

        # 1. Detect wildlife -- returns class names directly
        t0 = time.perf_counter()
        wildlife_dets = self.wildlife_detector.detect(frame)
        det_ms = (time.perf_counter() - t0) * 1000
        logger.debug(f"Wildlife detection: {len(wildlife_dets)} animals in {det_ms:.1f}ms")

        bboxes = [d.bbox for d in wildlife_dets]
        wildlife_confs = [d.confidence for d in wildlife_dets]

        # 2. Update tracker — returns all unclassified tracks
        tracks_to_classify = self.tracker.update(bboxes, wildlife_confs, timestamp)

        # 3. Match tracks to their wildlife detections (best-of-N: keep highest conf)
        for track in tracks_to_classify:
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

            # Keep the best confidence seen across frames
            if conf > track.confidence:
                track.species = species_name
                track.confidence = conf
            track.classify_count += 1

            # Check if we've crossed the threshold
            if track.confidence >= settings.wildlife_confidence_threshold:
                if not track.classified:
                    self.tracker.mark_classified(
                        track.track_id, track.species, track.confidence,
                    )

                    dt = datetime.fromtimestamp(timestamp)
                    if self.save_enabled and self._session_factory and settings.save_crops:
                        self._save_detection(
                            frame, track.bbox, track.species, track.confidence, dt,
                            detection_model=Path(settings.wildlife_model).parent.parent.name,
                            classifier_model=Path(settings.wildlife_model).parent.parent.name,
                            source=self._source,
                        )

                    new_detections.append({
                        "species": track.species,
                        "confidence": track.confidence,
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

        if self.save_enabled:
            self._init_database()
        self._running = True

        # Graceful shutdown on Ctrl+C
        def signal_handler(sig, frame):
            logger.info("Received shutdown signal (SIGINT/SIGTERM)")
            self._running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        self._log_pipeline_config()
        if self.mode_manager:
            self.mode_manager.check_now()
            logger.info(f"  Initial mode: {self.mode_manager.mode.value}")
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

    def run_on_image(
        self,
        image_path: str | Path,
        output: str | Path | None = None,
    ) -> list[dict]:
        """
        Run the pipeline on a single image file. Useful for testing.

        Args:
            image_path: Path to an image file.
            output: If provided, save an annotated copy with bounding boxes.

        Returns:
            List of detections.
        """
        import cv2

        frame = cv2.imread(str(image_path))
        if frame is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")

        self._source = "image"
        self._log_pipeline_config()

        # Use min_frames=1 so detections are immediate
        self.tracker = BirdTracker(min_frames_for_detection=1)
        logger.info(f"Processing image: {image_path}")
        detections = self.process_frame(frame)
        for det in detections:
            logger.info(
                f"  {det['species']} ({det['confidence']:.2f}) track={det['track_id']}"
            )
        if not detections:
            logger.info("  No detections")

        # Save annotated image if requested
        if output and detections:
            output = Path(output)
            output.parent.mkdir(parents=True, exist_ok=True)
            annotated = frame.copy()
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                conf = det["confidence"]
                label = f"{det['species']} {conf:.2f}"

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1,
                )
                cv2.rectangle(
                    annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1),
                    (0, 0, 255), -1,
                )
                cv2.putText(
                    annotated, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                    cv2.LINE_AA,
                )
            cv2.imwrite(str(output), annotated)
            logger.info(f"Annotated image saved: {output}")

        return detections

    def run_on_video(
        self,
        video_path: str | Path,
        process_every_n: int | None = None,
        output: str | Path | None = None,
        virtual_rtsp: bool = False,
    ) -> list[dict]:
        """
        Run the pipeline on a video file.

        Processes frames through the full pipeline (detect → track → classify)
        and logs detections. Use --output to save an annotated video.

        With virtual_rtsp=True, the video is treated exactly like an RTSP
        stream: saves to database, uses min_frames=3, applies species cooldown.

        Args:
            video_path: Path to a video file (mp4, avi, etc.).
            process_every_n: Process every Nth frame. If None, uses the
                pipeline's configured FrameSkipper value.
            output: If set, write an annotated video with bounding boxes,
                class labels, and confidence scores to this path.
            virtual_rtsp: If True, treat as RTSP (save to DB, min_frames=3).

        Returns:
            List of all detections across the video.
        """
        import cv2

        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps

        self._source = "rtsp" if virtual_rtsp else "video"
        self._log_pipeline_config()
        if virtual_rtsp:
            logger.info(f"Processing video (virtual RTSP): {video_path.name}")
        else:
            logger.info(f"Processing video: {video_path.name}")
        logger.info(f"  Resolution: {width}x{height}, FPS: {fps:.0f}, "
                     f"Duration: {duration:.1f}s, Frames: {total_frames}")

        # Set up output video writer if requested
        video_writer = None
        if output:
            output = Path(output)
            output.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(
                str(output), fourcc, fps, (width, height),
            )
            logger.info(f"  Output video: {output}")

        if virtual_rtsp:
            # Use RTSP defaults: require 3 frames before accepting a track
            if self.save_enabled:
                self._init_database()
            self.tracker = BirdTracker(min_frames_for_detection=3)
        else:
            self.tracker = BirdTracker(min_frames_for_detection=1)

        skip = process_every_n or self.skipper.process_every_n
        all_detections = []
        frame_num = 0
        frames_processed = 0
        t_start = time.time()

        while cap.isOpened():
            should_process = frame_num % skip == 0

            # Use grab()/retrieve() to avoid decoding skipped frames.
            # grab() advances the codec without producing a BGR image;
            # retrieve() does the full decode only when we need the pixels.
            if video_writer or should_process:
                ret, frame = cap.read()
            else:
                ret = cap.grab()
                frame = None

            if not ret:
                break

            if should_process:
                timestamp = frame_num / fps

                try:
                    detections = self.process_frame(frame, timestamp)
                except Exception as e:
                    logger.error(f"Frame {frame_num} failed: {e}")
                    frame_num += 1
                    continue

                for det in detections:
                    det["frame_num"] = frame_num
                    det["video_time"] = f"{timestamp:.1f}s"
                    all_detections.append(det)
                    logger.info(
                        f"  [{timestamp:.1f}s] {det['species']} "
                        f"({det['confidence']:.2f}) track={det['track_id']}"
                    )

                frames_processed += 1

            # Draw bounding boxes on every frame for smooth output video.
            # Use Kalman-predicted positions so boxes move smoothly even on
            # frames where YOLO didn't detect the bird.
            if video_writer:
                active_boxes = self.tracker.get_predicted_boxes()
                annotated = frame.copy()
                for det in active_boxes:
                    x1, y1, x2, y2 = det["bbox"]
                    conf = det["confidence"]
                    label = f"{det['species']} {conf:.2f}"

                    # Draw box
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)

                    # Draw label background
                    (tw, th), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1,
                    )
                    cv2.rectangle(
                        annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1),
                        (0, 0, 255), -1,
                    )
                    # Draw label text
                    cv2.putText(
                        annotated, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                        cv2.LINE_AA,
                    )
                video_writer.write(annotated)

            frame_num += 1

        cap.release()
        if video_writer:
            video_writer.release()
            logger.info(f"Output video saved: {output}")

        elapsed = time.time() - t_start

        logger.info(
            f"Video complete: {frames_processed} frames processed in {elapsed:.1f}s "
            f"({frames_processed / max(elapsed, 0.01):.1f} fps), "
            f"{len(all_detections)} detections"
        )

        return all_detections

    @property
    def total_detections(self) -> int:
        return self._total_detections

    @property
    def is_running(self) -> bool:
        return self._running


def _create_night_mode_components(
    device: str | None = None,
    backend: str = "ultralytics",
    vdevice=None,
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
        wildlife_detector = WildlifeDetector.from_hailo(vdevice=vdevice)

    mode_manager = DayNightManager(weather)

    return wildlife_detector, mode_manager


def create_pipeline_dev(
    checkpoint_path: str | Path = "models/bird-classifier/vit_small/best_model.pth",
    class_names: dict[int, str] | None = None,
    rtsp_url: str | None = None,
    device: str | None = None,
    enable_night_mode: bool = True,
    save_enabled: bool = True,
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

    camera = RTSPCamera(rtsp_url=rtsp_url)

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
        save_enabled=save_enabled,
    )


def create_pipeline_hailo(
    detection_hef: str | Path | None = None,
    classifier_hef: str | Path | None = None,
    class_names: dict[int, str] | None = None,
    rtsp_url: str | None = None,
    enable_night_mode: bool = True,
    save_enabled: bool = True,
) -> BirdPipeline:
    """
    Create a pipeline for Raspberry Pi with Hailo NPU.

    Uses Hailo HEF models for both detection and classification.
    All models share a single VDevice (Hailo-10H has one physical device).
    If enable_night_mode is True, also loads the wildlife detector and
    sets up camera IR mode polling for automatic day/night switching.
    """
    from hailo_platform import VDevice

    vdevice = VDevice()

    detector = BirdDetector.from_hailo(hef_path=detection_hef, vdevice=vdevice)

    if class_names is None:
        class_names = _load_class_names()

    classifier = BirdClassifier.from_hailo(
        hef_path=classifier_hef or settings.classifier_model_path,
        class_names=class_names,
        vdevice=vdevice,
    )

    camera = RTSPCamera(rtsp_url=rtsp_url)

    wildlife_detector = None
    mode_manager = None
    if enable_night_mode:
        wildlife_detector, mode_manager = _create_night_mode_components(
            backend="hailo",
            vdevice=vdevice,
        )

    return BirdPipeline(
        detector=detector,
        classifier=classifier,
        camera=camera,
        wildlife_detector=wildlife_detector,
        mode_manager=mode_manager,
        save_enabled=save_enabled,
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
        "--image", type=str, nargs="+", default=None,
        help="Run on one or more image files (supports shell globs)",
    )
    parser.add_argument(
        "--video", type=str, nargs="+", default=None,
        help="Run on one or more video files (supports shell globs)",
    )
    parser.add_argument(
        "--process-every-n", type=int, default=None,
        help="Process every Nth frame in video mode (default: from config)",
    )
    parser.add_argument(
        "--checkpoint", type=str, default="models/bird-classifier/vit_small/best_model.pth",
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
        help="Disable night mode entirely (no wildlife detector loaded)",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--day", action="store_true",
        help="Force daytime mode (bird detection + species classification). "
             "Default for --image/--video. Can override RTSP auto-detection.",
    )
    mode_group.add_argument(
        "--night", action="store_true",
        help="Force nighttime mode (wildlife detection). "
             "Can override RTSP auto-detection or image/video default.",
    )
    parser.add_argument(
        "--virtual-rtsp", action="store_true",
        help="Treat --video as if it were an RTSP stream: save to database, "
             "use min_frames=3, apply species cooldown. For testing the full "
             "RTSP pipeline with a video file.",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Disable saving detections to database and disk (RTSP only)",
    )
    parser.add_argument(
        "--output", nargs="?", const="auto", default=None,
        help="Save annotated output (with --video or --image). "
             "Optionally specify a path; defaults to detections/video/ or detections/image/",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=None,
        help="Set logging verbosity (default: from .env or INFO)",
    )
    args = parser.parse_args()

    # Expand globs that the shell didn't expand (e.g., running from PyCharm/uv)
    import glob as _glob
    for attr in ("video", "image"):
        paths = getattr(args, attr, None)
        if paths:
            expanded = []
            for p in paths:
                if "*" in p or "?" in p:
                    expanded.extend(sorted(_glob.glob(p)))
                else:
                    expanded.append(p)
            setattr(args, attr, expanded or None)

    log_level = args.log_level or settings.log_level
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    # Only RTSP mode (and --virtual-rtsp) saves to database/storage
    is_offline = args.image or args.video
    virtual_rtsp = args.virtual_rtsp
    if virtual_rtsp and not args.video:
        parser.error("--virtual-rtsp requires --video")
    if args.no_save and is_offline and not virtual_rtsp:
        parser.error("--no-save is only for RTSP mode (image/video never save to database)")
    if virtual_rtsp:
        save_enabled = not args.no_save
    elif is_offline:
        save_enabled = False
    else:
        save_enabled = not args.no_save

    # --day implies --no-night (no point loading wildlife model if forcing daytime)
    if args.day:
        args.no_night = True

    if args.mode == "dev":
        pipeline = create_pipeline_dev(
            checkpoint_path=args.checkpoint,
            rtsp_url=args.rtsp_url,
            device=args.device,
            enable_night_mode=not args.no_night,
            save_enabled=save_enabled,
        )
    else:
        pipeline = create_pipeline_hailo(
            rtsp_url=args.rtsp_url,
            enable_night_mode=not args.no_night,
            save_enabled=save_enabled,
        )

    # Apply day/night mode override.
    # Offline (image/video) defaults to day mode unless --night is specified.
    # RTSP uses auto-detection from sunrise/sunset unless overridden.
    if args.day:
        # Force daytime: disable the mode manager so _is_night_mode is always False
        pipeline.mode_manager = None
    elif args.night:
        # Force nighttime: set mode manager to night and disable auto-updates
        if pipeline.mode_manager and pipeline.wildlife_detector:
            from src.pipeline.mode_manager import PipelineMode
            pipeline.mode_manager.force_mode(PipelineMode.NIGHT)
            pipeline.mode_manager.update = lambda: False  # disable time-based switching
        else:
            parser.error("--night requires night mode components (use without --no-night)")
    elif is_offline and not virtual_rtsp:
        # Default to daytime for image/video inference
        # (virtual-rtsp uses auto-detection like real RTSP)
        pipeline.mode_manager = None

    if args.output and not is_offline:
        parser.error("--output requires --video or --image")

    # Custom --output path only makes sense with a single input file
    if args.output and args.output != "auto":
        file_count = len(args.video or args.image or [])
        if file_count > 1:
            parser.error("--output with a custom path requires a single input file")

    if args.image:
        for image_file in args.image:
            # Resolve auto output path per file
            output_path = None
            if args.output == "auto":
                output_dir = settings.detections_dir / "image"
                input_path = Path(image_file)
                output_path = str(
                    output_dir / f"{input_path.stem}_annotated{input_path.suffix}"
                )
            elif args.output:
                output_path = args.output

            results = pipeline.run_on_image(image_file, output=output_path)
            if results:
                for r in results:
                    print(f"{r['species']} ({r['confidence']:.2f}) at {r['bbox']}")
            else:
                print(f"No birds detected in {Path(image_file).name}.")
    elif args.video:
        for video_file in args.video:
            # Resolve auto output path per file
            output_path = None
            if args.output == "auto":
                output_dir = settings.detections_dir / "video"
                input_stem = Path(video_file).stem
                output_path = str(output_dir / f"{input_stem}_annotated.mp4")
            elif args.output:
                output_path = args.output

            results = pipeline.run_on_video(
                video_file,
                process_every_n=args.process_every_n,
                output=output_path,
                virtual_rtsp=virtual_rtsp,
            )
            if results:
                print(f"\n{'='*60}")
                print(f"Video inference complete: {len(results)} detections")
                print(f"{'='*60}")
                for r in results:
                    print(f"  [{r['video_time']}] {r['species']} ({r['confidence']:.2f})")
            else:
                print(f"No detections in {Path(video_file).name}.")
    else:
        pipeline.run()
