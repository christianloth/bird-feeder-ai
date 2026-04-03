"""Remove deer and fox from the wildlife YOLO dataset.

Backs up removed images/labels, removes annotations for deer (class 4) and
fox (class 6), remaps remaining class indices to be contiguous, and updates
dataset.yaml.

Images that contained ONLY deer/fox are moved to a backup folder.
Images that contained deer/fox AND other animals keep the other annotations.

Usage:
    python scripts/remove_classes_from_dataset.py
    python scripts/remove_classes_from_dataset.py --dry-run
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "wildlife-yolo"
BACKUP_DIR = DATASET_DIR / "backup_deer_fox"

# Classes to remove
REMOVE_CLASSES = {4, 6}  # deer=4, fox=6

# Old class index -> new class index (skip 4 and 6)
# 0->0, 1->1, 2->2, 3->3, 5->4, 7->5, 8->6, 9->7, 10->8, 11->9, 12->10
REMAP = {}
new_id = 0
for old_id in range(13):
    if old_id in REMOVE_CLASSES:
        continue
    REMAP[old_id] = new_id
    new_id += 1

NEW_NAMES = {
    0: "bird",
    1: "bobcat",
    2: "coyote",
    3: "raccoon",
    4: "rabbit",
    5: "skunk",
    6: "opossum",
    7: "squirrel",
    8: "armadillo",
    9: "cat",
    10: "dog",
}


def process_split(split: str, dry_run: bool) -> dict[str, int]:
    label_dir = DATASET_DIR / "labels" / split
    image_dir = DATASET_DIR / "images" / split
    backup_label_dir = BACKUP_DIR / "labels" / split
    backup_image_dir = BACKUP_DIR / "images" / split

    stats = {"updated": 0, "removed_images": 0, "removed_annotations": 0, "unchanged": 0}

    if not label_dir.exists():
        return stats

    for label_file in sorted(label_dir.iterdir()):
        if label_file.suffix != ".txt":
            continue

        if not label_file.exists():
            continue

        lines = label_file.read_text().splitlines()
        new_lines = []
        removed_count = 0

        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            class_id = int(parts[0])
            if class_id in REMOVE_CLASSES:
                removed_count += 1
                continue
            parts[0] = str(REMAP[class_id])
            new_lines.append(" ".join(parts))

        stats["removed_annotations"] += removed_count

        image_name = label_file.stem + ".jpg"
        image_file = image_dir / image_name

        if not new_lines:
            # Image only had deer/fox -- back it up and remove
            stats["removed_images"] += 1
            if not dry_run:
                backup_label_dir.mkdir(parents=True, exist_ok=True)
                backup_image_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(label_file), str(backup_label_dir / label_file.name))
                if image_file.exists():
                    shutil.move(str(image_file), str(backup_image_dir / image_name))
        elif removed_count > 0 or any(int(line.strip().split()[0]) != int(parts[0])
                                       for line, parts in zip(lines, [l.split() for l in new_lines])
                                       if line.strip()):
            # Had some deer/fox annotations removed, or class indices need remapping
            stats["updated"] += 1
            if not dry_run:
                # Back up original label
                backup_label_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(label_file), str(backup_label_dir / label_file.name))
                # Write updated label
                label_file.write_text("\n".join(new_lines) + "\n")
        else:
            stats["unchanged"] += 1
            # Still need to remap class indices even if no deer/fox were present
            if not dry_run:
                label_file.write_text("\n".join(new_lines) + "\n")
            stats["updated"] += 1
            stats["unchanged"] -= 1

    return stats


def update_dataset_yaml(dry_run: bool) -> None:
    yaml_path = DATASET_DIR / "dataset.yaml"
    content = f"""# Wildlife detection dataset for YOLOv8n
# Merged from Caltech Camera Traps (SW US trail cameras) + WCS Camera Traps (armadillo)
# Mix of daytime color and nighttime infrared images
# Deer and fox removed due to insufficient training samples

path: {DATASET_DIR}
train: images/train
val: images/val

nc: {len(NEW_NAMES)}
names:
"""
    for idx in sorted(NEW_NAMES.keys()):
        content += f"  {idx}: {NEW_NAMES[idx]}\n"

    if not dry_run:
        # Back up original
        shutil.copy2(str(yaml_path), str(BACKUP_DIR / "dataset.yaml.bak"))
        yaml_path.write_text(content)


def delete_caches(dry_run: bool) -> None:
    """Delete .cache files so Ultralytics rebuilds them."""
    for cache_file in DATASET_DIR.rglob("*.cache"):
        print(f"  Deleting cache: {cache_file}")
        if not dry_run:
            cache_file.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove deer and fox from wildlife dataset")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    args = parser.parse_args()

    if args.dry_run:
        print("=== DRY RUN (no changes will be made) ===\n")

    print("Class remapping:")
    old_names = {0: "bird", 1: "bobcat", 2: "coyote", 3: "raccoon", 4: "deer",
                 5: "rabbit", 6: "fox", 7: "skunk", 8: "opossum", 9: "squirrel",
                 10: "armadillo", 11: "cat", 12: "dog"}
    for old_id, new_id in REMAP.items():
        print(f"  {old_id} ({old_names[old_id]}) -> {new_id} ({NEW_NAMES[new_id]})")
    print(f"  4 (deer) -> REMOVED")
    print(f"  6 (fox) -> REMOVED")

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val"]:
        print(f"\nProcessing {split}...")
        stats = process_split(split, args.dry_run)
        print(f"  Labels updated (remapped): {stats['updated']}")
        print(f"  Images removed (deer/fox only): {stats['removed_images']}")
        print(f"  Annotations removed: {stats['removed_annotations']}")

    print("\nUpdating dataset.yaml...")
    update_dataset_yaml(args.dry_run)

    print("\nCleaning up cache files...")
    delete_caches(args.dry_run)

    if args.dry_run:
        print("\n=== DRY RUN complete. Run without --dry-run to apply. ===")
    else:
        print(f"\nDone! Backups saved to: {BACKUP_DIR}")
        print(f"Dataset now has {len(NEW_NAMES)} classes.")


if __name__ == "__main__":
    main()
