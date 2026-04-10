"""
Application settings loaded from config/config.yaml.

Edit config.yaml for your setup (camera credentials, thresholds, etc.).
That file is gitignored -- see config.yaml.example for the template.
"""

import yaml
from dataclasses import dataclass, field
from pathlib import Path

_CONFIG_DIR = Path(__file__).parent
_PROJECT_ROOT = _CONFIG_DIR.parent


def _load_yaml() -> dict:
    """Load config.yaml, falling back to defaults if missing."""
    config_path = _CONFIG_DIR / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


_cfg = _load_yaml()
_camera = _cfg.get("camera", {})
_detection = _cfg.get("detection", {})
_classification = _cfg.get("classification", {})
_wildlife = _cfg.get("wildlife", {})
_day_night = _cfg.get("day_night", {})
_pipeline = _cfg.get("pipeline", {})
_location = _cfg.get("location", {})
_storage = _cfg.get("storage", {})


@dataclass(frozen=True)
class Settings:
    # Logging
    log_level: str = _cfg.get("log_level", "INFO")

    # Project paths
    project_root: Path = _PROJECT_ROOT
    data_dir: Path = _PROJECT_ROOT / "data"
    models_dir: Path = _PROJECT_ROOT / "models"
    detections_dir: Path = _PROJECT_ROOT / "detections"

    # Camera
    rtsp_url: str = _camera.get("rtsp_url", "rtsp://admin:password@192.168.1.100:554/11")

    # Detection (daytime bird detection)
    detection_model: str = _detection.get("model", "")
    detection_model_path: Path = field(default=None)  # Hailo HEF path
    detection_confidence_threshold: float = _detection.get("confidence_threshold", 0.4)
    bird_class_id: int = _detection.get("bird_class_id", 14)

    # Classification (fine-tuned species classifier)
    classifier_model_path: Path = field(default=None)  # Hailo HEF path
    classification_confidence_threshold: float = _classification.get("confidence_threshold", 0.10)
    num_species: int = 555  # NABirds dataset

    # Wildlife detection (nighttime)
    wildlife_model: str = _wildlife.get("model", "")
    wildlife_model_path: Path = field(default=None)  # Hailo HEF path
    wildlife_confidence_threshold: float = _wildlife.get("confidence_threshold", 0.4)
    wildlife_class_names: dict[int, str] = field(default_factory=lambda: {
        0: "bird", 1: "bobcat", 2: "coyote", 3: "raccoon", 4: "rabbit",
        5: "skunk", 6: "opossum", 7: "squirrel", 8: "armadillo", 9: "cat", 10: "dog",
    })

    # Day/night mode switching
    mode_check_interval: int = _day_night.get("mode_check_interval", 60)
    night_offset_minutes: int = _day_night.get("night_offset_minutes", 30)
    day_offset_minutes: int = _day_night.get("day_offset_minutes", 30)

    # Pipeline
    process_every_n: int = _pipeline.get("process_every_n", 5)

    # Database
    database_url: str = f"sqlite:///{_PROJECT_ROOT / 'data' / 'birds.db'}"

    # Weather / Location
    latitude: float = _location.get("latitude", 33.1507)
    longitude: float = _location.get("longitude", -96.8236)
    timezone: str = _location.get("timezone", "America/Chicago")

    # Image storage
    save_crops: bool = _storage.get("save_crops", True)
    crop_quality: int = _storage.get("crop_quality", 85)
    thumbnail_size: tuple[int, int] = tuple(_storage.get("thumbnail_size", [200, 200]))
    thumbnail_quality: int = _storage.get("thumbnail_quality", 75)
    retention_days: int = _storage.get("retention_days", 90)

    def __post_init__(self):
        # Resolve model paths that depend on models_dir
        if self.detection_model_path is None:
            object.__setattr__(
                self, "detection_model_path",
                self.models_dir / "hef" / "yolov8n.hef",
            )
        if self.classifier_model_path is None:
            object.__setattr__(
                self, "classifier_model_path",
                self.models_dir / "hef" / "mobilenetv2_birds.hef",
            )
        if self.wildlife_model_path is None:
            object.__setattr__(
                self, "wildlife_model_path",
                self.models_dir / "wildlife" / "yolo11n-wildlife-equal" / "weights" / "best.pt",
            )


settings = Settings()


def get_device(override: str | None = None) -> str:
    """Select the best available device: CUDA → MPS → CPU.

    Args:
        override: Force a specific device (e.g., "cuda", "mps", "cpu").
                  If None, auto-detects the best available.

    Returns:
        Device string suitable for both PyTorch and Ultralytics.
    """
    if override:
        return override

    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
