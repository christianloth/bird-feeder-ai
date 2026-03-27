"""
Application settings. Edit these values for your setup.
"""

from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # Project paths
    project_root: Path = Path(__file__).parent.parent
    data_dir: Path = project_root / "data"
    models_dir: Path = project_root / "models"
    detections_dir: Path = project_root / "detections"

    # Camera
    rtsp_url: str = "rtsp://admin:password@192.168.1.100:554/stream1"
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
