"""
Bird tracking and ROI cropping.

Handles extracting bird regions from detection bounding boxes and tracking
individual birds across frames to avoid duplicate detections of the same
bird sitting on the feeder.
"""

import logging
import time
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BirdTrack:
    """Tracks a single bird across multiple frames."""
    track_id: int
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    centroid: tuple[float, float]
    first_seen: float
    last_seen: float
    classified: bool = False
    species: str | None = None
    confidence: float = 0.0
    frame_count: int = 1


class BirdTracker:
    """
    Simple centroid-based tracker for birds at a feeder.

    Birds at a feeder tend to stay relatively still, so a simple distance-based
    tracker works well. If a detection's centroid is within `max_distance` of
    an existing track's centroid, it's considered the same bird.

    This prevents logging the same bird 50 times while it sits on the feeder
    for 30 seconds.
    """

    def __init__(
        self,
        max_distance: float = 80.0,
        max_missing_seconds: float = 5.0,
        min_frames_for_detection: int = 3,
    ):
        """
        Args:
            max_distance: Max pixel distance between centroids to consider same bird.
            max_missing_seconds: Remove track after this many seconds without a match.
            min_frames_for_detection: Require this many consecutive frames before
                considering it a real detection (filters out noise/false positives).
        """
        self.max_distance = max_distance
        self.max_missing_seconds = max_missing_seconds
        self.min_frames_for_detection = min_frames_for_detection

        self._tracks: dict[int, BirdTrack] = {}
        self._next_id = 0

    def _centroid(self, bbox: tuple[int, int, int, int]) -> tuple[float, float]:
        """Calculate the center point of a bounding box."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def _distance(self, p1: tuple[float, float], p2: tuple[float, float]) -> float:
        """Euclidean distance between two points."""
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    def update(
        self,
        detections: list[tuple[int, int, int, int]],
        timestamp: float | None = None,
    ) -> list[BirdTrack]:
        """
        Update tracks with new detection bounding boxes.

        Args:
            detections: List of bounding boxes [(x1, y1, x2, y2), ...]
            timestamp: Current time (defaults to time.time())

        Returns:
            List of NEW tracks that have met the min_frames threshold
            and haven't been classified yet. These should be sent to
            the species classifier.
        """
        now = timestamp or time.time()
        new_tracks_ready = []

        # Remove stale tracks
        stale_ids = [
            tid for tid, track in self._tracks.items()
            if now - track.last_seen > self.max_missing_seconds
        ]
        for tid in stale_ids:
            logger.debug(f"Track {tid} expired ({self._tracks[tid].species or 'unclassified'})")
            del self._tracks[tid]

        # Match detections to existing tracks
        unmatched_detections = list(range(len(detections)))
        matched_track_ids = set()

        for det_idx in list(unmatched_detections):
            bbox = detections[det_idx]
            centroid = self._centroid(bbox)

            best_track_id = None
            best_distance = float("inf")

            for tid, track in self._tracks.items():
                if tid in matched_track_ids:
                    continue
                dist = self._distance(centroid, track.centroid)
                if dist < self.max_distance and dist < best_distance:
                    best_distance = dist
                    best_track_id = tid

            if best_track_id is not None:
                # Update existing track
                track = self._tracks[best_track_id]
                track.bbox = bbox
                track.centroid = centroid
                track.last_seen = now
                track.frame_count += 1
                matched_track_ids.add(best_track_id)
                unmatched_detections.remove(det_idx)

                # Check if track just became ready for classification
                if (
                    track.frame_count == self.min_frames_for_detection
                    and not track.classified
                ):
                    new_tracks_ready.append(track)

        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            bbox = detections[det_idx]
            centroid = self._centroid(bbox)
            track = BirdTrack(
                track_id=self._next_id,
                bbox=bbox,
                centroid=centroid,
                first_seen=now,
                last_seen=now,
            )
            self._tracks[self._next_id] = track
            self._next_id += 1

            # If min_frames is 1, immediately ready
            if self.min_frames_for_detection <= 1:
                new_tracks_ready.append(track)

        return new_tracks_ready

    def mark_classified(
        self,
        track_id: int,
        species: str,
        confidence: float,
    ) -> None:
        """Mark a track as classified with species info."""
        if track_id in self._tracks:
            self._tracks[track_id].classified = True
            self._tracks[track_id].species = species
            self._tracks[track_id].confidence = confidence

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
