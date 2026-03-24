"""
Image storage service for detected bird crops and thumbnails.

Handles saving cropped bird images, generating thumbnails, and cleaning up
old images based on the retention policy.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image

from config.settings import settings

logger = logging.getLogger(__name__)


class ImageStorage:
    """Manages saving and organizing detected bird images on disk."""

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or settings.detections_dir

    def _get_day_dir(self, timestamp: datetime) -> Path:
        """Get the directory for a given day: detections/YYYY/MM/DD/"""
        day_dir = self.base_dir / timestamp.strftime("%Y/%m/%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        return day_dir

    def _build_filename(
        self,
        timestamp: datetime,
        species_name: str,
        confidence: float,
    ) -> str:
        """Build a descriptive filename like '20250315_143022_northern_cardinal_0.87'."""
        time_str = timestamp.strftime("%Y%m%d_%H%M%S")
        safe_name = species_name.lower().replace(" ", "_").replace("'", "")
        return f"{time_str}_{safe_name}_{confidence:.2f}"

    def save_detection(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        species_name: str,
        confidence: float,
        timestamp: datetime | None = None,
        padding: int = 20,
    ) -> tuple[str, str]:
        """
        Crop a bird from the frame, save the crop and a thumbnail.

        Args:
            frame: Full camera frame (BGR numpy array from OpenCV).
            bbox: Bounding box (x1, y1, x2, y2) in pixel coordinates.
            species_name: Predicted species name.
            confidence: Classification confidence score.
            timestamp: Detection time. Defaults to now.
            padding: Extra pixels around the bounding box.

        Returns:
            Tuple of (crop_path, thumbnail_path) relative to base_dir.
        """
        timestamp = timestamp or datetime.now()
        x1, y1, x2, y2 = bbox

        # Add padding, clamp to frame bounds
        h, w = frame.shape[:2]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)

        # Crop from frame (OpenCV uses BGR, PIL expects RGB)
        crop_bgr = frame[y1:y2, x1:x2]
        crop_rgb = crop_bgr[:, :, ::-1]  # BGR → RGB
        img = Image.fromarray(crop_rgb)

        # Build paths
        day_dir = self._get_day_dir(timestamp)
        base_name = self._build_filename(timestamp, species_name, confidence)
        crop_path = day_dir / f"{base_name}.jpg"
        thumb_path = day_dir / f"{base_name}_thumb.jpg"

        # Save full crop
        img.save(crop_path, "JPEG", quality=settings.crop_quality, optimize=True)

        # Save thumbnail
        thumb = img.copy()
        thumb.thumbnail(settings.thumbnail_size)
        thumb.save(thumb_path, "JPEG", quality=settings.thumbnail_quality, optimize=True)

        # Return paths relative to base_dir for database storage
        rel_crop = str(crop_path.relative_to(self.base_dir))
        rel_thumb = str(thumb_path.relative_to(self.base_dir))

        logger.debug(f"Saved detection image: {rel_crop}")
        return rel_crop, rel_thumb

    def get_absolute_path(self, relative_path: str) -> Path:
        """Convert a relative path from the database back to an absolute path."""
        return self.base_dir / relative_path

    def cleanup_old_images(self, retention_days: int | None = None) -> int:
        """
        Delete detection images older than the retention period.

        Args:
            retention_days: Days to keep images. Defaults to config value.

        Returns:
            Number of files deleted.
        """
        retention_days = retention_days or settings.retention_days
        cutoff = datetime.now() - timedelta(days=retention_days)
        deleted = 0

        if not self.base_dir.exists():
            return 0

        for jpg_file in self.base_dir.rglob("*.jpg"):
            try:
                # Parse timestamp from filename: YYYYMMDD_HHMMSS_...
                date_str = jpg_file.stem[:15]  # "20250315_143022"
                file_time = datetime.strptime(date_str, "%Y%m%d_%H%M%S")
                if file_time < cutoff:
                    jpg_file.unlink()
                    deleted += 1
            except (ValueError, IndexError):
                continue

        # Remove empty date directories
        if self.base_dir.exists():
            for day_dir in sorted(self.base_dir.rglob("*"), reverse=True):
                if day_dir.is_dir() and not any(day_dir.iterdir()):
                    day_dir.rmdir()

        if deleted > 0:
            logger.info(f"Cleaned up {deleted} images older than {retention_days} days.")

        return deleted

    def get_disk_usage_mb(self) -> float:
        """Calculate total disk usage of stored images in megabytes."""
        if not self.base_dir.exists():
            return 0.0
        total = sum(f.stat().st_size for f in self.base_dir.rglob("*.jpg"))
        return total / (1024 * 1024)
