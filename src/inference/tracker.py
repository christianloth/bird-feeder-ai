"""
Bird tracking, ROI cropping, and multi-frame classification support.

Uses Norfair's Kalman-filter + IOU-based tracker instead of simple centroid
distance matching. This solves the "tracker fragmentation" problem where a bird
moving on a swaying feeder jumps >80 pixels between processed frames and gets
assigned a new track each time.

The Kalman filter predicts where each bird should be on the next frame based on
its velocity, so even when processing every 5th frame (167ms gaps at 30fps),
the predicted bounding box still overlaps with the actual detection. This is
what Frigate NVR uses via Norfair for its bird feeder integrations.

Logit averaging is maintained per track: raw classifier logits accumulate across
frames and are averaged before softmax for more accurate species predictions.
(See Dussert et al. 2025: logit averaging outperforms max-pooling, majority
voting, and probability averaging.)

A track stays "unclassified" until the averaged confidence crosses the threshold,
allowing the classifier to retry on each new frame with a fresh crop.
"""

import logging
import time
from dataclasses import dataclass, field

import numpy as np
from norfair import Detection as NorfairDetection
from norfair import Tracker as NorfairTracker

logger = logging.getLogger(__name__)


@dataclass
class BirdTrack:
    """Tracks a single bird across multiple frames."""
    track_id: int
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    centroid: tuple[float, float]
    first_seen: float
    last_seen: float
    # Classification state — supports multi-frame logit averaging:
    # The classifier runs on each processed frame until confidence crosses
    # the threshold. Raw logits are accumulated in logit_sum and averaged
    # before softmax, which produces better predictions than any single frame.
    classified: bool = False  # True = passed confidence threshold, stop retrying
    species: str | None = None  # Current best species from averaged logits
    confidence: float = 0.0  # Confidence from averaged logits (not single-frame)
    frame_count: int = 0  # Number of frames this track has been detected
    logit_sum: np.ndarray | None = None  # Running sum of raw logits across frames
    classify_count: int = 0  # Number of frames classified so far


class BirdTracker:
    """
    IOU + Kalman filter tracker for birds at a feeder, powered by Norfair.

    Instead of simple centroid distance (which fails when a bird shifts >80px
    between processed frames due to feeder sway or fast movement), this tracker:

    1. Predicts where each bird SHOULD be using a Kalman velocity model
    2. Matches detections to predictions using bounding box IOU overlap
    3. Maintains tracks through brief occlusions (hit_counter_max frames)
    4. Requires initialization_delay frames before accepting a new track

    This is the same approach used by Frigate NVR (the most popular open-source
    NVR for bird feeder projects like WhosAtMyFeeder).
    """

    def __init__(
        self,
        min_frames_for_detection: int = 3,
        hit_counter_max: int = 30,
    ):
        """
        Args:
            min_frames_for_detection: Require this many consecutive detections
                before accepting a track (filters transient false positives).
                Use 1 for video/image mode, 3 for RTSP.
            hit_counter_max: Maximum number of update() calls a track survives
                without a matching detection. At ~6 processed fps (every 5th
                frame of 30fps), 30 = ~5 seconds of grace period.
        """
        self.min_frames_for_detection = min_frames_for_detection
        self._hit_counter_max = hit_counter_max

        # Norfair's initialization_delay is 0-indexed:
        # delay=0 → accepted on first detection, delay=2 → needs 3 detections
        self._init_delay = max(0, min_frames_for_detection - 1)

        self._norfair = NorfairTracker(
            distance_function="iou",
            distance_threshold=0.7,  # 1-IOU: match if IOU > 0.3
            hit_counter_max=hit_counter_max,
            initialization_delay=self._init_delay,
        )

        # Our classification state, keyed by Norfair track ID
        self._tracks: dict[int, BirdTrack] = {}
        # Frame counter to distinguish current-frame detections from stale ones
        self._frame_counter = 0

    @staticmethod
    def _centroid(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
        """Calculate the center point of a bounding box."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def update(
        self,
        bboxes: list[tuple[int, int, int, int]],
        confidences: list[float] | None = None,
        timestamp: float | None = None,
    ) -> list[BirdTrack]:
        """
        Update tracks with new detection bounding boxes.

        Args:
            bboxes: List of bounding boxes [(x1, y1, x2, y2), ...]
            confidences: Detection confidence for each bbox (from YOLO).
                If None, defaults to 1.0 for all.
            timestamp: Current time (defaults to time.time())

        Returns:
            List of active tracks that need classification: all tracks that
            were detected in the current frame and haven't been confidently
            classified yet.
        """
        now = timestamp or time.time()
        self._frame_counter += 1

        if confidences is None:
            confidences = [1.0] * len(bboxes)

        # Convert YOLO detections to Norfair format.
        # Norfair expects 2-point bboxes: [[x1,y1], [x2,y2]] with scores per point.
        norfair_dets = []
        for bbox, conf in zip(bboxes, confidences):
            x1, y1, x2, y2 = bbox
            det = NorfairDetection(
                points=np.array([[x1, y1], [x2, y2]], dtype=float),
                scores=np.array([conf, conf]),
                data={"bbox": (x1, y1, x2, y2), "frame": self._frame_counter},
            )
            norfair_dets.append(det)

        # Run Norfair's Kalman + IOU tracker
        tracked_objects = self._norfair.update(detections=norfair_dets)

        active_ids = set()
        tracks_to_classify = []

        for obj in tracked_objects:
            tid = obj.id
            active_ids.add(tid)

            # Check if this object was matched to a detection in the current frame
            # (vs being predicted by the Kalman filter with no matching detection)
            detected_this_frame = (
                obj.last_detection is not None
                and obj.last_detection.data is not None
                and obj.last_detection.data.get("frame") == self._frame_counter
            )

            # Get the bbox from the actual detection (not Kalman estimate)
            if detected_this_frame:
                bbox = obj.last_detection.data["bbox"]
            elif tid in self._tracks:
                bbox = self._tracks[tid].bbox  # keep last known position
            else:
                # Newly predicted object with no prior state — use Kalman estimate
                est = obj.estimate.flatten().astype(int)
                bbox = (int(est[0]), int(est[1]), int(est[2]), int(est[3]))

            # Create or update our classification state
            if tid not in self._tracks:
                self._tracks[tid] = BirdTrack(
                    track_id=tid,
                    bbox=bbox,
                    centroid=self._centroid(bbox),
                    first_seen=now,
                    last_seen=now,
                    frame_count=0,
                )
                logger.debug(
                    f"New track {tid} at centroid "
                    f"({self._centroid(bbox)[0]:.0f}, {self._centroid(bbox)[1]:.0f})"
                )

            track = self._tracks[tid]
            track.bbox = bbox
            track.centroid = self._centroid(bbox)
            track.last_seen = now
            if detected_this_frame:
                track.frame_count += 1

            # Only classify tracks with a fresh detection (new crop to work with)
            # that haven't been confidently classified yet
            if detected_this_frame and not track.classified:
                tracks_to_classify.append(track)

        # Clean up tracks that Norfair has removed (expired / lost)
        expired_ids = [tid for tid in self._tracks if tid not in active_ids]
        for tid in expired_ids:
            track = self._tracks[tid]
            if track.species and not track.classified:
                logger.debug(
                    f"Track {tid} expired with best-effort: "
                    f"{track.species} ({track.confidence:.2f}, "
                    f"{track.classify_count} attempts)"
                )
            else:
                logger.debug(
                    f"Track {tid} expired "
                    f"({track.species or 'unclassified'})"
                )
            del self._tracks[tid]

        return tracks_to_classify

    def accumulate_logits(
        self,
        track_id: int,
        logits: np.ndarray,
        species: str,
        confidence: float,
    ) -> None:
        """
        Add a frame's logits to a track's running sum.

        Updates the track's best prediction from the averaged logits.
        Does NOT mark as classified — call mark_classified separately
        when the averaged confidence passes the threshold.
        """
        if track_id not in self._tracks:
            return
        track = self._tracks[track_id]
        if track.logit_sum is None:
            track.logit_sum = logits.copy()
        else:
            track.logit_sum += logits
        track.classify_count += 1
        track.species = species
        track.confidence = confidence

    def mark_classified(
        self,
        track_id: int,
        species: str,
        confidence: float,
    ) -> None:
        """Mark a track as confidently classified (stops reclassification)."""
        if track_id in self._tracks:
            self._tracks[track_id].classified = True
            self._tracks[track_id].species = species
            self._tracks[track_id].confidence = confidence

    def get_predicted_boxes(self) -> list[dict]:
        """
        Get Kalman-predicted bounding boxes for all active tracked birds.

        Used for smooth video annotation: on frames where YOLO doesn't detect
        the bird (or on skipped frames), the Kalman filter still predicts where
        the bird should be based on its velocity. This keeps the bounding box
        moving smoothly instead of freezing at the last detection position.

        Returns:
            List of dicts with species, confidence, bbox, track_id for all
            active tracks that have a species prediction.
        """
        boxes = []
        for obj in self._norfair.tracked_objects:
            tid = obj.id
            if tid not in self._tracks:
                continue
            track = self._tracks[tid]
            if track.species is None:
                continue

            # Use Norfair's Kalman-filtered estimate for the box position
            est = obj.estimate.flatten().astype(int)
            predicted_bbox = (int(est[0]), int(est[1]), int(est[2]), int(est[3]))

            boxes.append({
                "species": track.species,
                "confidence": track.confidence,
                "bbox": predicted_bbox,
                "track_id": track.track_id,
            })
        return boxes

    def reset(self) -> None:
        """Clear all tracks. Used when switching between day/night modes."""
        count = len(self._tracks)
        self._tracks.clear()
        # Recreate the Norfair tracker (no public reset API)
        self._norfair = NorfairTracker(
            distance_function="iou",
            distance_threshold=0.7,
            hit_counter_max=self._hit_counter_max,
            initialization_delay=self._init_delay,
        )
        self._frame_counter = 0
        if count:
            logger.debug(f"Cleared {count} tracks")

    @property
    def active_tracks(self) -> list[BirdTrack]:
        """Get all currently active tracks."""
        return list(self._tracks.values())

    @property
    def active_count(self) -> int:
        """Number of birds currently being tracked."""
        return len(self._tracks)


def crop_bird_roi(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    padding: int = 20,
    target_size: tuple[int, int] | None = (224, 224),
) -> np.ndarray:
    """
    Crop a bird region from a frame for classification.

    Args:
        frame: Full camera frame (BGR numpy array).
        bbox: Bounding box (x1, y1, x2, y2).
        padding: Extra pixels around the bounding box.
        target_size: Resize crop to this size for the classifier.
            Use (224, 224) for MobileNetV2/EfficientNet.
            Set to None to skip resizing.

    Returns:
        Cropped (and optionally resized) BGR numpy array.
    """
    import cv2

    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]

    # Add padding, clamp to frame bounds
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    crop = frame[y1:y2, x1:x2]

    if target_size is not None and crop.size > 0:
        crop = cv2.resize(crop, target_size, interpolation=cv2.INTER_LINEAR)

    return crop
