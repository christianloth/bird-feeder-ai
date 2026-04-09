"""
Application settings. Edit these values for your setup.

Camera credentials are loaded from config/camera.yaml (gitignored).
"""

import yaml
from pydantic_settings import BaseSettings
from pathlib import Path


def _load_camera_config() -> dict:
    """Load camera credentials from camera.yaml."""
    camera_yaml = Path(__file__).parent / "camera.yaml"
    if camera_yaml.exists():
        with open(camera_yaml) as f:
            config = yaml.safe_load(f)
        cam = config.get("camera", {})
        stream_path = cam.get("main_stream", "/11") if cam.get("stream") == "main" else cam.get("sub_stream", "/12")
        return {
            "rtsp_url": f"rtsp://{cam['username']}:{cam['password']}@{cam['ip']}:{cam.get('rtsp_port', 554)}{stream_path}",
        }
    return {}


_camera_config = _load_camera_config()


class Settings(BaseSettings):
    # Project paths
    project_root: Path = Path(__file__).parent.parent
    data_dir: Path = project_root / "data"
    models_dir: Path = project_root / "models"
    detections_dir: Path = project_root / "detections"

    # Camera
    rtsp_url: str = _camera_config.get("rtsp_url", "rtsp://admin:password@192.168.1.100:554/11")
    camera_resolution: tuple[int, int] = (1920, 1080)
    inference_resolution: tuple[int, int] = (640, 640)

    # Detection (YOLOv8n on Hailo)
    detection_model_path: Path = models_dir / "hef" / "yolov8n.hef"
    detection_confidence_threshold: float = 0.4
    bird_class_id: int = 14  # COCO class 14 = "bird"

    # Classification (MobileNetV2 fine-tuned)
    classifier_model_path: Path = models_dir / "hef" / "mobilenetv2_birds.hef"
    classification_confidence_threshold: float = 0.10
    num_species: int = 555  # NABirds dataset

    # Wildlife detection (nighttime)
    wildlife_model_path: Path = (
        models_dir / "wildlife" / "yolo11n-wildlife-equal" / "weights" / "best.pt"
    )
    wildlife_confidence_threshold: float = 0.4
    wildlife_class_names: dict[int, str] = {
        0: "bird", 1: "bobcat", 2: "coyote", 3: "raccoon", 4: "rabbit",
        5: "skunk", 6: "opossum", 7: "squirrel", 8: "armadillo", 9: "cat", 10: "dog",
    }

    # Day/night mode switching
    mode_check_interval: int = 60  # Seconds between sun time checks
    night_offset_minutes: int = 30  # Enter night mode this many minutes after sunset
    day_offset_minutes: int = 30    # Enter day mode this many minutes before sunrise

    # Training
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 25
    num_workers: int = 4
    train_val_split: float = 0.8

    # Database
    database_url: str = f"sqlite:///{project_root / 'data' / 'birds.db'}"

    # Weather (Open-Meteo, free, no API key needed)
    latitude: float = 33.1507   # Frisco, TX
    longitude: float = -96.8236
    timezone: str = "America/Chicago"

    # Image storage
    save_crops: bool = True
    crop_quality: int = 85
    thumbnail_size: tuple[int, int] = (200, 200)
    thumbnail_quality: int = 75
    retention_days: int = 90  # Delete crops older than this

    model_config = {"env_prefix": "BIRD_"}


settings = Settings()
