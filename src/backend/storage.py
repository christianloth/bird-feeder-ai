"""
Image storage service for detected bird crops and thumbnails.

Handles saving cropped bird images, generating thumbnails, and cleaning up
old images based on the retention policy.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path

import cv2
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
        safe_name = species_name.lower().replace(" ", "_").replace("'", "").replace("/", "-")
        return f"{time_str}_{safe_name}_{confidence:.2f}"

    def save_detection(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        species_name: str,
        confidence: float,
        timestamp: datetime | None = None,
        padding: int = 20,
    ) -> dict[str, str]:
        """
        Save all images for a detection: annotated crop, clean crop,
        thumbnail, and full frame.

        Args:
            frame: Full camera frame (BGR numpy array from OpenCV).
            bbox: Bounding box (x1, y1, x2, y2) in pixel coordinates.
            species_name: Predicted species name.
            confidence: Classification confidence score.
            timestamp: Detection time. Defaults to now.
            padding: Extra pixels around the bounding box.

        Returns:
            Dict with relative paths: image_path, thumbnail_path,
            clean_crop_path, frame_path.
        """
        timestamp = timestamp or datetime.now()
        x1, y1, x2, y2 = bbox

        # Add padding, clamp to frame bounds
        h, w = frame.shape[:2]
        pad_x1 = max(0, x1 - padding)
        pad_y1 = max(0, y1 - padding)
        pad_x2 = min(w, x2 + padding)
        pad_y2 = min(h, y2 + padding)

        # Clean crop (no annotations -- for training)
        clean_crop_bgr = frame[pad_y1:pad_y2, pad_x1:pad_x2].copy()

        # Annotated crop (red bounding box drawn -- for review)
        annotated_bgr = clean_crop_bgr.copy()
        box_x1 = x1 - pad_x1
        box_y1 = y1 - pad_y1
        box_x2 = x2 - pad_x1
        box_y2 = y2 - pad_y1
        cv2.rectangle(annotated_bgr, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 255), 2)

        # Build paths
        day_dir = self._get_day_dir(timestamp)
        base_name = self._build_filename(timestamp, species_name, confidence)

        crop_path = day_dir / f"{base_name}.jpg"
        thumb_path = day_dir / f"{base_name}_thumb.jpg"
        clean_path = day_dir / f"{base_name}_clean.jpg"
        frame_path = day_dir / f"{base_name}_frame.jpg"

        # Save annotated crop (with red bbox)
        annotated_img = Image.fromarray(annotated_bgr[:, :, ::-1])
        annotated_img.save(crop_path, "JPEG", quality=settings.crop_quality, optimize=True)

        # Save thumbnail
        thumb = annotated_img.copy()
        thumb.thumbnail(settings.thumbnail_size)
        thumb.save(thumb_path, "JPEG", quality=settings.thumbnail_quality, optimize=True)

        # Save clean crop (no annotations -- for classifier retraining)
        clean_img = Image.fromarray(clean_crop_bgr[:, :, ::-1])
        clean_img.save(clean_path, "JPEG", quality=settings.crop_quality, optimize=True)

        # Save full frame (for YOLO retraining)
        frame_img = Image.fromarray(frame[:, :, ::-1])
        frame_img.save(frame_path, "JPEG", quality=settings.crop_quality, optimize=True)

        # Return paths relative to base_dir for database storage
        result = {
            "image_path": str(crop_path.relative_to(self.base_dir)),
            "thumbnail_path": str(thumb_path.relative_to(self.base_dir)),
            "clean_crop_path": str(clean_path.relative_to(self.base_dir)),
            "frame_path": str(frame_path.relative_to(self.base_dir)),
        }

        crop_kb = crop_path.stat().st_size / 1024
        frame_kb = frame_path.stat().st_size / 1024
        logger.debug(
            f"Saved detection: {result['image_path']} "
            f"(crop={crop_kb:.0f}KB, frame={frame_kb:.0f}KB)"
        )
        return result

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
