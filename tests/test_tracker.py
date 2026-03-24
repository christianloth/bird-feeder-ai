"""Tests for bird tracking and ROI cropping."""

import numpy as np

from src.inference.tracker import BirdTracker, crop_bird_roi


def test_new_detection_creates_track():
    tracker = BirdTracker(min_frames_for_detection=1)
    ready = tracker.update([(100, 100, 200, 200)], timestamp=1.0)
    assert len(ready) == 1
    assert tracker.active_count == 1


def test_same_bird_not_duplicated():
    tracker = BirdTracker(max_distance=80, min_frames_for_detection=1)

    # Frame 1: bird detected
    ready1 = tracker.update([(100, 100, 200, 200)], timestamp=1.0)
    assert len(ready1) == 1

    # Frame 2: same bird, slightly moved
    ready2 = tracker.update([(105, 105, 205, 205)], timestamp=2.0)
    assert len(ready2) == 0  # Not a new track
    assert tracker.active_count == 1  # Still one bird


def test_two_different_birds():
    tracker = BirdTracker(max_distance=50, min_frames_for_detection=1)
    ready = tracker.update(
        [(100, 100, 200, 200), (500, 500, 600, 600)],
        timestamp=1.0,
    )
    assert len(ready) == 2
    assert tracker.active_count == 2


def test_stale_track_removed():
    tracker = BirdTracker(max_missing_seconds=2.0, min_frames_for_detection=1)

    tracker.update([(100, 100, 200, 200)], timestamp=1.0)
    assert tracker.active_count == 1

    # 3 seconds later, no detections
    tracker.update([], timestamp=4.0)
    assert tracker.active_count == 0


def test_min_frames_threshold():
    tracker = BirdTracker(min_frames_for_detection=3)

    # Frame 1, 2: not ready yet
    ready1 = tracker.update([(100, 100, 200, 200)], timestamp=1.0)
    assert len(ready1) == 0
    ready2 = tracker.update([(100, 100, 200, 200)], timestamp=2.0)
    assert len(ready2) == 0

    # Frame 3: now it's ready
    ready3 = tracker.update([(100, 100, 200, 200)], timestamp=3.0)
    assert len(ready3) == 1


def test_mark_classified():
    tracker = BirdTracker(min_frames_for_detection=1)
    ready = tracker.update([(100, 100, 200, 200)], timestamp=1.0)
    track = ready[0]

    tracker.mark_classified(track.track_id, "Northern Cardinal", 0.95)

    assert track.classified is True
    assert track.species == "Northern Cardinal"
    assert track.confidence == 0.95


def test_crop_bird_roi():
    # Create a 480x640 fake frame (H, W, C)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[100:200, 100:200] = 255  # White bird region

    crop = crop_bird_roi(frame, (100, 100, 200, 200), padding=10, target_size=(224, 224))
    assert crop.shape == (224, 224, 3)


def test_crop_bird_roi_no_resize():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    crop = crop_bird_roi(frame, (100, 100, 200, 200), padding=0, target_size=None)
    assert crop.shape == (100, 100, 3)


def test_crop_bird_roi_clamps_to_frame():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Bounding box near edge — padding should be clamped
    crop = crop_bird_roi(frame, (0, 0, 50, 50), padding=20, target_size=None)
    assert crop.shape[0] <= 70
    assert crop.shape[1] <= 70
