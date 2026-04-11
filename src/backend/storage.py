"""
Image storage service for detection frames.

Saves one full frame per detection. Crops, thumbnails, and annotated
views are generated on-the-fly from the frame + bounding box coordinates
stored in the database.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image

from config.settings import settings

logger = logging.getLogger(__name__)


class ImageStorage:
    """Manages saving and organizing detection frame images on disk."""

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or settings.detections_dir / "rtsp"

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
        safe_name = species_name.lower().replace(" ", "_").replace("'", "").replace("/", "-")
        return f"{time_str}_{safe_name}_{confidence:.2f}"

    def save_detection(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        species_name: str,
        confidence: float,
        timestamp: datetime | None = None,
    ) -> dict[str, str]:
        """
        Save the full frame for a detection.

        Crops, thumbnails, and annotated views are generated on-the-fly
        from the frame image + bounding box coordinates in the database.

        Args:
            frame: Full camera frame (BGR numpy array from OpenCV).
            bbox: Bounding box (x1, y1, x2, y2) in pixel coordinates.
            species_name: Predicted species name.
            confidence: Classification confidence score.
            timestamp: Detection time. Defaults to now.

        Returns:
            Dict with relative path: frame_path.
        """
        timestamp = timestamp or datetime.now()

        day_dir = self._get_day_dir(timestamp)
        base_name = self._build_filename(timestamp, species_name, confidence)
        frame_path = day_dir / f"{base_name}.jpg"

        frame_img = Image.fromarray(frame[:, :, ::-1])
        frame_img.save(frame_path, "JPEG", quality=settings.crop_quality, optimize=True)

        result = {
            "frame_path": str(frame_path.relative_to(self.base_dir)),
        }

        frame_kb = frame_path.stat().st_size / 1024
        logger.debug(f"Saved frame: {result['frame_path']} ({frame_kb:.0f}KB)")
        return result

    @staticmethod
    def crop_from_frame(
        frame_path: Path,
        bbox: tuple[float, float, float, float],
        padding: int = 20,
    ) -> Image.Image:
        """
        Crop a detection region from a saved frame.

        Args:
            frame_path: Path to the full frame JPEG.
            bbox: Bounding box (x1, y1, x2, y2) in pixel coordinates.
            padding: Extra pixels around the bounding box.

        Returns:
            PIL Image of the cropped region.
        """
        img = Image.open(frame_path)
        w, h = img.size
        x1, y1, x2, y2 = bbox
        crop_box = (
            max(0, int(x1) - padding),
            max(0, int(y1) - padding),
            min(w, int(x2) + padding),
            min(h, int(y2) + padding),
        )
        return img.crop(crop_box)

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
            logger.info(f"Cleaned up {deleted} images older than {retention_days} days")
        else:
            logger.debug(f"No images older than {retention_days} days to clean up")

        return deleted

    def get_disk_usage_mb(self) -> float:
        """Calculate total disk usage of stored images in megabytes."""
        if not self.base_dir.exists():
            return 0.0
        total = sum(f.stat().st_size for f in self.base_dir.rglob("*.jpg"))
        return total / (1024 * 1024)
