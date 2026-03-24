"""
Integration tests that verify the full flow works end-to-end.

Tests the actual pipeline: detection → tracking → storage → database → API query.
Uses fake frames and simulated detections since we don't have a real camera.
"""

import tempfile
from datetime import datetime, date
from pathlib import Path

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from src.backend.database import Base, Species, Detection, WeatherObservation, DailySummary
from src.backend.storage import ImageStorage
from src.backend.weather import WeatherService, describe_weather_code
from src.inference.tracker import BirdTracker, crop_bird_roi
from src.pipeline.camera import RTSPCamera, FrameSkipper, FrameResult


# ---- Database Integration ----

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
        # 1. Register species (would come from NABirds dataset)
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

        # 2. Simulate detections over a day
        detections = [
            Detection(
                timestamp=datetime(2025, 6, 15, 7, 30),
                species_id=cardinal.id,
                confidence=0.94,
                detection_model="yolov8n",
                classifier_model="mobilenetv2",
                bbox_x1=100, bbox_y1=150, bbox_x2=250, bbox_y2=350,
                image_path="2025/06/15/20250615_073000_northern_cardinal_0.94.jpg",
                thumbnail_path="2025/06/15/20250615_073000_northern_cardinal_0.94_thumb.jpg",
            ),
            Detection(
                timestamp=datetime(2025, 6, 15, 8, 15),
                species_id=cardinal.id,
                confidence=0.91,
                detection_model="yolov8n",
                classifier_model="mobilenetv2",
                bbox_x1=120, bbox_y1=160, bbox_x2=270, bbox_y2=360,
            ),
            Detection(
                timestamp=datetime(2025, 6, 15, 10, 45),
                species_id=blue_jay.id,
                confidence=0.88,
                detection_model="yolov8n",
                classifier_model="mobilenetv2",
                bbox_x1=200, bbox_y1=100, bbox_x2=400, bbox_y2=300,
            ),
        ]
        session.add_all(detections)
        session.commit()

        # 3. Query: how many detections today?
        count = session.query(Detection).filter(
            Detection.timestamp >= datetime(2025, 6, 15),
            Detection.timestamp < datetime(2025, 6, 16),
        ).count()
        assert count == 3

        # 4. Query: which species was most common?
        from sqlalchemy import func, desc
        most_common = (
            session.query(Species.common_name, func.count(Detection.id).label("cnt"))
            .join(Detection.species)
            .group_by(Species.id)
            .order_by(desc("cnt"))
            .first()
        )
        assert most_common[0] == "Northern Cardinal"
        assert most_common[1] == 2

        # 5. Query: detections with high confidence
        high_conf = session.query(Detection).filter(
            Detection.confidence >= 0.90
        ).all()
        assert len(high_conf) == 2

        # 6. Verify relationship traversal works
        det = session.query(Detection).first()
        assert det.species.common_name == "Northern Cardinal"
        assert det.species.family == "Cardinalidae"


def test_weather_observation_storage():
    """Verify weather data can be stored and queried alongside detections."""
    engine, SessionLocal = _make_db()

    with SessionLocal() as session:
        obs = WeatherObservation(
            timestamp=datetime(2025, 6, 15, 10, 0),
            temperature_c=28.5,
            humidity_pct=65.0,
            wind_speed_kmh=12.3,
            precipitation_mm=0.0,
            cloud_cover_pct=25.0,
            weather_code=2,
        )
        session.add(obs)
        session.commit()

        result = session.query(WeatherObservation).first()
        assert result.temperature_c == 28.5
        assert result.weather_code == 2


def test_daily_summary_storage():
    """Verify daily summary records can be stored."""
    engine, SessionLocal = _make_db()

    with SessionLocal() as session:
        summary = DailySummary(
            date=date(2025, 6, 15),
            total_detections=47,
            unique_species=8,
            avg_temperature=29.5,
        )
        session.add(summary)
        session.commit()

        result = session.query(DailySummary).first()
        assert result.total_detections == 47
        assert result.unique_species == 8


# ---- Image Storage Integration ----

def test_image_storage_saves_crop_and_thumbnail():
    """Verify a bird crop and thumbnail are saved to disk correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = ImageStorage(base_dir=Path(tmpdir))

        # Create a fake 480x640 BGR frame with a "bird" region
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[100:300, 150:350] = [0, 0, 200]  # Red-ish bird (BGR)

        crop_path, thumb_path = storage.save_detection(
            frame=frame,
            bbox=(150, 100, 350, 300),
            species_name="Northern Cardinal",
            confidence=0.95,
            timestamp=datetime(2025, 6, 15, 14, 30, 22),
        )

        # Verify files exist
        abs_crop = storage.get_absolute_path(crop_path)
        abs_thumb = storage.get_absolute_path(thumb_path)
        assert abs_crop.exists()
        assert abs_thumb.exists()

        # Verify thumbnail is smaller than crop
        assert abs_thumb.stat().st_size < abs_crop.stat().st_size

        # Verify filename format
        assert "northern_cardinal" in crop_path
        assert "0.95" in crop_path
        assert "_thumb" in thumb_path

        # Verify directory structure
        assert "2025/06/15" in crop_path


def test_image_storage_disk_usage():
    """Verify disk usage calculation works."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = ImageStorage(base_dir=Path(tmpdir))

        # Should be 0 when empty
        assert storage.get_disk_usage_mb() == 0.0

        # Save an image
        frame = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        storage.save_detection(frame, (10, 10, 190, 190), "Test Bird", 0.9)

        # Should be > 0 now
        assert storage.get_disk_usage_mb() > 0.0


def test_image_storage_cleanup():
    """Verify old image cleanup works."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = ImageStorage(base_dir=Path(tmpdir))
        frame = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)

        # Save an image with a very old timestamp
        storage.save_detection(
            frame, (10, 10, 190, 190), "Old Bird", 0.9,
            timestamp=datetime(2020, 1, 1, 12, 0, 0),
        )

        # Save a recent image
        storage.save_detection(
            frame, (10, 10, 190, 190), "New Bird", 0.9,
        )

        # Cleanup with 30-day retention should delete the old one
        deleted = storage.cleanup_old_images(retention_days=30)
        assert deleted >= 1  # At least the old crop + thumb


# ---- Tracker Integration ----

def test_tracker_full_scenario():
    """
    Simulate a real scenario:
    - Bird lands on feeder (appears in multiple frames)
    - Gets tracked and classified
    - Bird leaves
    - New bird arrives
    """
    tracker = BirdTracker(
        max_distance=80,
        max_missing_seconds=3.0,
        min_frames_for_detection=3,
    )

    # Frames 1-2: Cardinal appears, not yet ready
    ready = tracker.update([(100, 100, 200, 200)], timestamp=1.0)
    assert len(ready) == 0
    ready = tracker.update([(102, 101, 202, 201)], timestamp=2.0)
    assert len(ready) == 0

    # Frame 3: Cardinal confirmed (3 consecutive frames)
    ready = tracker.update([(104, 100, 204, 200)], timestamp=3.0)
    assert len(ready) == 1
    cardinal_track = ready[0]
    assert cardinal_track.frame_count == 3

    # Classify it
    tracker.mark_classified(cardinal_track.track_id, "Northern Cardinal", 0.94)

    # Frames 4-5: Cardinal still there, but already classified
    ready = tracker.update([(106, 100, 206, 200)], timestamp=4.0)
    assert len(ready) == 0  # Already classified, not new
    ready = tracker.update([(108, 100, 208, 200)], timestamp=5.0)
    assert len(ready) == 0

    # Frame 6-8: Cardinal leaves (no detections)
    ready = tracker.update([], timestamp=6.0)
    ready = tracker.update([], timestamp=7.0)
    ready = tracker.update([], timestamp=8.0)

    # After 3s missing, cardinal track should be gone
    ready = tracker.update([], timestamp=9.0)
    assert tracker.active_count == 0

    # Frame 10-12: Blue Jay arrives at different location
    tracker.update([(400, 300, 500, 400)], timestamp=10.0)
    tracker.update([(402, 301, 502, 401)], timestamp=11.0)
    ready = tracker.update([(404, 302, 504, 402)], timestamp=12.0)
    assert len(ready) == 1  # New bird ready for classification
    assert ready[0].track_id != cardinal_track.track_id


def test_crop_then_store():
    """Verify the crop → storage pipeline works together."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = ImageStorage(base_dir=Path(tmpdir))

        # Simulate a camera frame
        frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        bbox = (500, 300, 700, 500)

        # Crop for classifier
        crop = crop_bird_roi(frame, bbox, padding=20, target_size=(224, 224))
        assert crop.shape == (224, 224, 3)

        # Also save the detection image (from original frame, not resized)
        crop_path, thumb_path = storage.save_detection(
            frame, bbox, "House Sparrow", 0.87,
        )
        assert Path(tmpdir, crop_path).exists()
        assert Path(tmpdir, thumb_path).exists()


# ---- Camera Unit Tests ----

def test_frame_skipper():
    """Verify frame skipping logic."""
    skipper = FrameSkipper(process_every_n=5)

    processed = []
    for i in range(25):
        if skipper.should_process(i):
            processed.append(i)

    # First fires at frame 4 (4 - (-1) >= 5), then every 5 frames after
    assert processed == [4, 9, 14, 19, 24]
    assert len(processed) == 5


def test_camera_init_defaults():
    """Verify camera initializes with config defaults."""
    cam = RTSPCamera()
    assert cam.rtsp_url == "rtsp://admin:password@192.168.1.100:554/stream1"
    assert cam.is_connected is False
    assert cam.frame_number == 0


def test_camera_custom_url():
    """Verify camera accepts custom RTSP URL."""
    cam = RTSPCamera(rtsp_url="rtsp://user:pass@10.0.0.5:554/live")
    assert cam.rtsp_url == "rtsp://user:pass@10.0.0.5:554/live"


# ---- Weather Integration ----

def test_weather_code_descriptions():
    """Verify weather code lookup works."""
    assert describe_weather_code(0) == "Clear sky"
    assert describe_weather_code(61) == "Slight rain"
    assert describe_weather_code(95) == "Thunderstorm"
    assert describe_weather_code(None) == "Unknown"
    assert "Unknown" in describe_weather_code(999)


def test_weather_service_fetches_current():
    """Actually call the Open-Meteo API to verify it works."""
    service = WeatherService()
    try:
        weather = service.get_current_weather()
        # API should return data (it's free, no key needed)
        assert weather is not None
        assert "temperature_c" in weather
        assert "humidity_pct" in weather
        assert weather["temperature_c"] is not None
        # Temperature should be in reasonable range
        assert -50 < weather["temperature_c"] < 60
    finally:
        service.close()


def test_weather_service_fetches_sun_times():
    """Verify sunrise/sunset data comes back for Frisco, TX."""
    service = WeatherService()
    try:
        sun = service.get_daily_sun_times()
        assert sun is not None
        assert "sunrise" in sun
        assert "sunset" in sun
        # Sunset should be after sunrise
        assert sun["sunset"] > sun["sunrise"]
    finally:
        service.close()


def test_weather_service_hourly():
    """Verify hourly weather data is returned."""
    service = WeatherService()
    try:
        hourly = service.get_hourly_weather()
        assert len(hourly) == 24  # 24 hours in a day
        assert "temperature_c" in hourly[0]
        assert "timestamp" in hourly[0]
    finally:
        service.close()
