"""
Export production detections as training data.

Queries the database for detections and exports them in two formats:

1. Classification (ImageFolder) -- for retraining the species classifier:
   data/field-collected/classification/{species_name}/image.jpg
   Uses clean crops (no bounding box annotations).

2. YOLO detection -- for retraining the object detector:
   data/field-collected/yolo/images/{split}/image.jpg
   data/field-collected/yolo/labels/{split}/image.txt
   Uses full frames with normalized bbox coordinates.

Supports filtering by confidence, reviewed status, and date range.
Includes perceptual hash deduplication to skip near-identical images.

Usage:
    python scripts/export_training_data.py --format classification --min-confidence 0.7
    python scripts/export_training_data.py --format yolo --reviewed-only
    python scripts/export_training_data.py --format both
"""

import argparse
import logging
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from sqlalchemy import and_

from config.settings import settings
from src.backend.database import Detection, get_session
from src.backend.storage import ImageStorage

logger = logging.getLogger(__name__)

EXPORT_BASE = settings.data_dir / "field-collected"


def perceptual_hash(image_path: Path, hash_size: int = 8) -> str:
    """
    Compute a simple average perceptual hash for an image.

    Similar images produce similar hashes. Two images with the same
    hash (or Hamming distance < threshold) are near-duplicates.
    """
    img = Image.open(image_path).convert("L").resize(
        (hash_size + 1, hash_size), Image.LANCZOS,
    )
    pixels = np.array(img)
    diff = pixels[:, 1:] > pixels[:, :-1]
    return "".join(str(int(b)) for b in diff.flatten())


def hamming_distance(h1: str, h2: str) -> int:
    """Number of differing bits between two hashes."""
    return sum(c1 != c2 for c1, c2 in zip(h1, h2))


def query_detections(
    session,
    min_confidence: float = 0.0,
    reviewed_only: bool = False,
    exclude_false_positives: bool = True,
    since: str | None = None,
) -> list[Detection]:
    """Query detections with filters."""
    filters = []

    if min_confidence > 0:
        filters.append(Detection.confidence >= min_confidence)
    if reviewed_only:
        filters.append(Detection.reviewed.is_(True))
    if exclude_false_positives:
        filters.append(Detection.is_false_positive.is_(False))
    if since:
        since_dt = datetime.fromisoformat(since)
        filters.append(Detection.timestamp >= since_dt)

    detections = (
        session.query(Detection)
        .filter(and_(*filters) if filters else True)
        .order_by(Detection.timestamp)
        .all()
    )
    return detections


def get_training_label(detection: Detection) -> str | None:
    """
    Get the training label for a detection.

    Uses corrected_species if the detection was reviewed and relabeled,
    otherwise falls back to the original prediction.
    """
    if detection.corrected_species:
        return detection.corrected_species.common_name
    if detection.species:
        return detection.species.common_name
    return None


def export_classification(
    detections: list[Detection],
    storage: ImageStorage,
    dedup_threshold: int = 5,
) -> int:
    """
    Export clean crops in ImageFolder format for classifier retraining.

    Output: data/field-collected/classification/{species_name}/image.jpg
    """
    output_dir = EXPORT_BASE / "classification"
    exported = 0
    skipped_dedup = 0
    skipped_missing = 0

    # Track hashes per species for deduplication
    hashes_by_species: dict[str, list[str]] = {}

    for det in detections:
        label = get_training_label(det)
        if label is None:
            continue

        # Use clean crop (no bounding box annotations)
        crop_rel = det.clean_crop_path
        if not crop_rel:
            # Fall back for old detections that don't have clean crops
            crop_rel = det.image_path
            if not crop_rel:
                skipped_missing += 1
                continue
            logger.debug(
                f"Detection {det.id} has no clean crop, "
                "falling back to annotated crop"
            )

        crop_path = storage.get_absolute_path(crop_rel)
        if not crop_path.exists():
            skipped_missing += 1
            logger.debug(f"Image not found: {crop_path}")
            continue

        # Deduplicate by perceptual hash
        phash = perceptual_hash(crop_path)
        species_hashes = hashes_by_species.setdefault(label, [])
        is_duplicate = any(
            hamming_distance(phash, existing) < dedup_threshold
            for existing in species_hashes
        )
        if is_duplicate:
            skipped_dedup += 1
            logger.debug(f"Skipping near-duplicate for {label}: {crop_rel}")
            continue
        species_hashes.append(phash)

        # Copy to species directory
        safe_name = label.lower().replace(" ", "_").replace("'", "")
        species_dir = output_dir / safe_name
        species_dir.mkdir(parents=True, exist_ok=True)

        dest = species_dir / crop_path.name
        shutil.copy2(crop_path, dest)
        exported += 1

    logger.info(
        f"Classification export: {exported} images, "
        f"{skipped_dedup} duplicates skipped, "
        f"{skipped_missing} missing files"
    )

    # Write a manifest
    if exported > 0:
        species_counts = {
            k: len(v) for k, v in hashes_by_species.items() if v
        }
        manifest = output_dir / "manifest.txt"
        with open(manifest, "w") as f:
            f.write(f"# Exported {datetime.now().isoformat()}\n")
            f.write(f"# Total: {exported} images\n")
            for species, count in sorted(
                species_counts.items(), key=lambda x: -x[1],
            ):
                f.write(f"{species}: {count}\n")

    return exported


def export_yolo(
    detections: list[Detection],
    storage: ImageStorage,
    val_split: float = 0.2,
    dedup_threshold: int = 5,
) -> int:
    """
    Export full frames + YOLO labels for detector retraining.

    Output:
      data/field-collected/yolo/images/train/image.jpg
      data/field-collected/yolo/images/val/image.jpg
      data/field-collected/yolo/labels/train/image.txt
      data/field-collected/yolo/labels/val/image.txt
      data/field-collected/yolo/dataset.yaml
    """
    output_dir = EXPORT_BASE / "yolo"
    exported = 0
    skipped_dedup = 0
    skipped_missing = 0

    # Build class list from the detections
    all_labels = set()
    for det in detections:
        label = get_training_label(det)
        if label:
            all_labels.add(label)
    class_list = sorted(all_labels)
    class_to_id = {name: idx for idx, name in enumerate(class_list)}

    # Group detections by frame (multiple detections can share the same frame)
    frames: dict[str, list[Detection]] = {}
    for det in detections:
        frame_rel = det.frame_path
        if not frame_rel:
            skipped_missing += 1
            continue
        frames.setdefault(frame_rel, []).append(det)

    # Deduplicate frames by perceptual hash
    frame_hashes: list[str] = []
    unique_frames: list[tuple[str, list[Detection]]] = []

    for frame_rel, frame_dets in frames.items():
        frame_path = storage.get_absolute_path(frame_rel)
        if not frame_path.exists():
            skipped_missing += 1
            continue

        phash = perceptual_hash(frame_path)
        is_duplicate = any(
            hamming_distance(phash, existing) < dedup_threshold
            for existing in frame_hashes
        )
        if is_duplicate:
            skipped_dedup += 1
            continue
        frame_hashes.append(phash)
        unique_frames.append((frame_rel, frame_dets))

    # Split into train/val
    n_val = max(1, int(len(unique_frames) * val_split))
    val_frames = unique_frames[:n_val]
    train_frames = unique_frames[n_val:]

    for split_name, split_frames in [("train", train_frames), ("val", val_frames)]:
        img_dir = output_dir / "images" / split_name
        lbl_dir = output_dir / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for frame_rel, frame_dets in split_frames:
            frame_path = storage.get_absolute_path(frame_rel)

            # Get frame dimensions for normalizing bbox coords
            img = Image.open(frame_path)
            img_w, img_h = img.size

            # Copy frame
            dest_img = img_dir / frame_path.name
            shutil.copy2(frame_path, dest_img)

            # Write YOLO label file
            label_file = lbl_dir / (frame_path.stem + ".txt")
            with open(label_file, "w") as f:
                for det in frame_dets:
                    label = get_training_label(det)
                    if label is None or label not in class_to_id:
                        continue

                    class_id = class_to_id[label]
                    # Convert absolute bbox to YOLO normalized format
                    cx = ((det.bbox_x1 + det.bbox_x2) / 2) / img_w
                    cy = ((det.bbox_y1 + det.bbox_y2) / 2) / img_h
                    bw = (det.bbox_x2 - det.bbox_x1) / img_w
                    bh = (det.bbox_y2 - det.bbox_y1) / img_h
                    f.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

            exported += 1

    # Write dataset.yaml
    if exported > 0:
        yaml_path = output_dir / "dataset.yaml"
        with open(yaml_path, "w") as f:
            f.write(f"# Exported {datetime.now().isoformat()}\n")
            f.write("path: .  # resolved at runtime by train script\n")
            f.write("train: images/train\n")
            f.write("val: images/val\n\n")
            f.write(f"nc: {len(class_list)}\n")
            f.write("names:\n")
            for idx, name in enumerate(class_list):
                f.write(f"  {idx}: {name}\n")

    logger.info(
        f"YOLO export: {exported} frames ({len(train_frames)} train, "
        f"{len(val_frames)} val), {len(class_list)} classes, "
        f"{skipped_dedup} duplicates skipped, {skipped_missing} missing files"
    )
    return exported


def main():
    parser = argparse.ArgumentParser(
        description="Export production detections as training data",
    )
    parser.add_argument(
        "--format", choices=["classification", "yolo", "both"], default="both",
        help="Export format (default: both)",
    )
    parser.add_argument(
        "--min-confidence", type=float, default=0.0,
        help="Minimum confidence threshold for export",
    )
    parser.add_argument(
        "--reviewed-only", action="store_true",
        help="Only export reviewed detections",
    )
    parser.add_argument(
        "--include-false-positives", action="store_true",
        help="Include detections marked as false positives",
    )
    parser.add_argument(
        "--since", type=str, default=None,
        help="Only export detections after this date (ISO format, e.g., 2026-03-01)",
    )
    parser.add_argument(
        "--dedup-threshold", type=int, default=5,
        help="Perceptual hash Hamming distance threshold for dedup (lower = stricter)",
    )
    parser.add_argument(
        "--val-split", type=float, default=0.2,
        help="Fraction of data for validation in YOLO export (default: 0.2)",
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="Delete existing export directory before exporting",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    session = get_session()
    storage = ImageStorage()

    detections = query_detections(
        session,
        min_confidence=args.min_confidence,
        reviewed_only=args.reviewed_only,
        exclude_false_positives=not args.include_false_positives,
        since=args.since,
    )
    logger.info(f"Found {len(detections)} detections matching filters")

    if not detections:
        logger.warning("No detections to export")
        session.close()
        return

    if args.clean and EXPORT_BASE.exists():
        shutil.rmtree(EXPORT_BASE)
        logger.info(f"Cleaned existing export directory: {EXPORT_BASE}")

    if args.format in ("classification", "both"):
        export_classification(
            detections, storage, dedup_threshold=args.dedup_threshold,
        )

    if args.format in ("yolo", "both"):
        export_yolo(
            detections, storage,
            val_split=args.val_split,
            dedup_threshold=args.dedup_threshold,
        )

    session.close()
    logger.info(f"Export complete. Output: {EXPORT_BASE}")


if __name__ == "__main__":
    main()
