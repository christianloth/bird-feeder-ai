"""
Integration tests that verify the full flow works end-to-end.

Tests the actual pipeline: detection → tracking → storage → database → API query.
Uses fake frames and simulated detections since we don't have a real camera.
"""

import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
from sqlalchemy import create_engine, func, desc
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.backend.database import Base, Species, Detection
from src.backend.storage import ImageStorage
from src.backend.weather import WeatherService
from src.inference.tracker import BirdTracker, crop_bird_roi


def _make_db():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return engine, sessionmaker(bind=engine)


def test_full_detection_flow_in_database():
    """Simulate: bird detected → species identified → stored in DB → query back."""
    engine, SessionLocal = _make_db()

    with SessionLocal() as session:
        cardinal = Species(
            common_name="Northern Cardinal",
            scientific_name="Cardinalis cardinalis",
            family="Cardinalidae",
            class_index=0,
        )
        blue_jay = Species(
            common_name="Blue Jay",
            scientific_name="Cyanocitta cristata",
            family="Corvidae",
            class_index=1,
        )
        session.add_all([cardinal, blue_jay])
        session.flush()

        detections = [
            Detection(
                timestamp=datetime(2025, 6, 15, 7, 30),
                species_id=cardinal.id,
                confidence=0.94,
                detection_model="yolov8n",
                classifier_model="vit_small",
                bbox_x1=100, bbox_y1=150, bbox_x2=250, bbox_y2=350,
            ),
            Detection(
                timestamp=datetime(2025, 6, 15, 8, 15),
                species_id=cardinal.id,
                confidence=0.91,
                detection_model="yolov8n",
                classifier_model="vit_small",
            ),
            Detection(
                timestamp=datetime(2025, 6, 15, 10, 45),
                species_id=blue_jay.id,
                confidence=0.88,
                detection_model="yolov8n",
                classifier_model="vit_small",
            ),
        ]
        session.add_all(detections)
        session.commit()

        # Query: how many detections?
        assert session.query(Detection).count() == 3

        # Query: most common species?
        most_common = (
            session.query(Species.common_name, func.count(Detection.id).label("cnt"))
            .join(Detection.species)
            .group_by(Species.id)
            .order_by(desc("cnt"))
            .first()
        )
        assert most_common[0] == "Northern Cardinal"
        assert most_common[1] == 2

        # Verify relationship traversal
        det = session.query(Detection).first()
        assert det.species.common_name == "Northern Cardinal"


def test_image_storage_saves_and_cleans_up():
    """Verify crop/thumbnail saving and old image cleanup."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = ImageStorage(base_dir=Path(tmpdir))
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[100:300, 150:350] = [0, 0, 200]

        # Save a detection
        paths = storage.save_detection(
            frame=frame,
            bbox=(150, 100, 350, 300),
            species_name="Northern Cardinal",
            confidence=0.95,
            timestamp=datetime(2025, 6, 15, 14, 30, 22),
        )

        abs_crop = storage.get_absolute_path(paths["image_path"])
        abs_thumb = storage.get_absolute_path(paths["thumbnail_path"])
        abs_clean = storage.get_absolute_path(paths["clean_crop_path"])
        abs_frame = storage.get_absolute_path(paths["frame_path"])
        assert abs_crop.exists()
        assert abs_thumb.exists()
        assert abs_clean.exists()
        assert abs_frame.exists()
        assert abs_thumb.stat().st_size < abs_crop.stat().st_size
        assert abs_frame.stat().st_size >= abs_crop.stat().st_size
        assert "northern_cardinal" in paths["image_path"]
        assert "2025/06/15" in paths["image_path"]
        assert paths["clean_crop_path"].endswith("_clean.jpg")
        assert paths["frame_path"].endswith("_frame.jpg")

        # Save an old image, verify cleanup removes it
        storage.save_detection(
            frame, (10, 10, 190, 190), "Old Bird", 0.9,
            timestamp=datetime(2020, 1, 1, 12, 0, 0),
        )
        deleted = storage.cleanup_old_images(retention_days=30)
        assert deleted >= 1


def test_tracker_full_scenario():
    """
    Simulate: bird lands → tracked across frames → classified → leaves → new bird arrives.
    """
    tracker = BirdTracker(
        max_distance=80,
        max_missing_seconds=3.0,
        min_frames_for_detection=3,
    )

    # Frames 1-2: Cardinal appears, not yet ready
    assert len(tracker.update([(100, 100, 200, 200)], timestamp=1.0)) == 0
    assert len(tracker.update([(102, 101, 202, 201)], timestamp=2.0)) == 0

    # Frame 3: Cardinal confirmed
    ready = tracker.update([(104, 100, 204, 200)], timestamp=3.0)
    assert len(ready) == 1
    cardinal_track = ready[0]
    tracker.mark_classified(cardinal_track.track_id, "Northern Cardinal", 0.94)

    # Frames 4-5: Still there, already classified — no new tracks
    assert len(tracker.update([(106, 100, 206, 200)], timestamp=4.0)) == 0
    assert len(tracker.update([(108, 100, 208, 200)], timestamp=5.0)) == 0

    # Cardinal leaves, track expires after 3s
    for t in range(6, 10):
        tracker.update([], timestamp=float(t))
    assert tracker.active_count == 0

    # Blue Jay arrives at different location
    tracker.update([(400, 300, 500, 400)], timestamp=10.0)
    tracker.update([(402, 301, 502, 401)], timestamp=11.0)
    ready = tracker.update([(404, 302, 504, 402)], timestamp=12.0)
    assert len(ready) == 1
    assert ready[0].track_id != cardinal_track.track_id


def test_crop_then_store():
    """Verify the crop → storage pipeline works together."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = ImageStorage(base_dir=Path(tmpdir))
        frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        bbox = (500, 300, 700, 500)

        # Crop for classifier input
        crop = crop_bird_roi(frame, bbox, padding=20, target_size=(224, 224))
        assert crop.shape == (224, 224, 3)

        # Save detection image to disk
        paths = storage.save_detection(
            frame, bbox, "House Sparrow", 0.87,
        )
        assert Path(tmpdir, paths["image_path"]).exists()
        assert Path(tmpdir, paths["clean_crop_path"]).exists()
        assert Path(tmpdir, paths["frame_path"]).exists()


def test_weather_api_live():
    """Call the real Open-Meteo API for Frisco, TX weather."""
    service = WeatherService()
    try:
        weather = service.get_current_weather()
        assert weather is not None
        assert weather["temperature_c"] is not None
        assert -50 < weather["temperature_c"] < 60

        sun = service.get_daily_sun_times()
        assert sun is not None
        assert sun["sunset"] > sun["sunrise"]
    finally:
        service.close()
