"""
Merge Caltech Camera Traps + WCS Armadillo datasets into YOLO format.

Both source datasets use COCO Camera Traps JSON format where bounding boxes
are stored as [x, y, width, height] in absolute pixel coordinates inside a
single JSON annotation file. YOLO (ultralytics) expects a completely different
layout: one .txt file per image with normalized [class center_x center_y width height]
values (0-1 range relative to image dimensions).

This script:
  1. Reads COCO JSON annotations from both Caltech and WCS datasets
  2. Maps each dataset's category names to a unified set of 13 wildlife classes
  3. Converts bounding boxes from COCO pixel format to YOLO normalized format
  4. Groups annotations by image (one image can contain multiple animals)
  5. Splits images into train/val (85/15) and copies them into the directory
     structure that ultralytics expects:
       wildlife-yolo/
         images/train/   images/val/
         labels/train/   labels/val/
  6. Writes dataset.yaml for ultralytics training

Caltech Camera Traps provides 12 of the 13 classes (birds and common North
American wildlife). WCS Camera Traps fills in the armadillo class specifically,
since armadillos are common in the target deployment area (Frisco, TX) but
absent from the Caltech dataset.
"""

import json
import shutil
import random
from pathlib import Path
from collections import Counter

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
CCT_DIR = PROJECT_ROOT / "data" / "caltech-camera-traps"
WCS_DIR = PROJECT_ROOT / "data" / "wcs-armadillo"
OUT_DIR = PROJECT_ROOT / "data" / "wildlife-yolo"

# Target classes for the wildlife detector
CLASS_NAMES = [
    "bird",       # 0
    "bobcat",     # 1
    "coyote",     # 2
    "raccoon",    # 3
    "deer",       # 4
    "rabbit",     # 5
    "fox",        # 6
    "skunk",      # 7
    "opossum",    # 8
    "squirrel",   # 9
    "armadillo",  # 10
    "cat",        # 11
    "dog",        # 12
]

# CCT category name -> our unified class index.
# Caltech uses its own category IDs internally; we remap by name to our 13-class scheme.
CCT_MAPPING = {
    "bird": 0,
    "bobcat": 1,
    "coyote": 2,
    "raccoon": 3,
    "deer": 4,
    "rabbit": 5,
    "fox": 6,
    "skunk": 7,
    "opossum": 8,
    "squirrel": 9,
    "cat": 11,
    "dog": 12,
}

# WCS uses Latin species names. All armadillo species map to our single armadillo class (10).
WCS_MAPPING = {
    "dasypus novemcinctus": 10,   # nine-banded armadillo (most common in TX)
    "dasypus kappleri": 10,        # greater long-nosed armadillo
    "unknown armadillo": 10,
}

VAL_SPLIT = 0.15  # 15% validation
RANDOM_SEED = 42


def load_cct_data():
    """Load Caltech Camera Traps bounding box annotations."""
    bbox_file = CCT_DIR / "caltech_bboxes.json"
    with open(bbox_file) as f:
        data = json.load(f)

    cat_map = {c["id"]: c["name"] for c in data["categories"]}
    img_map = {img["id"]: img for img in data["images"]}

    # Get available image files (CCT20 benchmark subset)
    available_images = set()
    img_dir = CCT_DIR / "eccv_18_all_images_sm"
    if img_dir.exists():
        available_images = {f.name for f in img_dir.glob("*.jpg")}

    print(f"CCT: {len(data['annotations'])} annotations, {len(available_images)} available images")

    entries = []
    for ann in data["annotations"]:
        cat_name = cat_map.get(ann["category_id"], "")
        if cat_name not in CCT_MAPPING:
            continue

        img_info = img_map.get(ann["image_id"])
        if img_info is None:
            continue

        # Check if we have this image file
        fname = img_info.get("file_name", "")
        # CCT20 images use the image ID as filename
        img_id_name = f"{ann['image_id']}.jpg"

        if fname in available_images:
            src_path = img_dir / fname
        elif img_id_name in available_images:
            src_path = img_dir / img_id_name
            fname = img_id_name
        else:
            continue

        # COCO bbox format: [x, y, width, height] in absolute pixels
        bbox = ann["bbox"]
        img_w = img_info.get("width", 0)
        img_h = img_info.get("height", 0)

        if img_w == 0 or img_h == 0 or bbox[2] <= 0 or bbox[3] <= 0:
            continue

        entries.append({
            "src_path": src_path,
            "filename": fname,
            "class_id": CCT_MAPPING[cat_name],
            "bbox": bbox,
            "img_w": img_w,
            "img_h": img_h,
            "source": "cct",
        })

    return entries


def load_wcs_armadillo_data():
    """Load WCS armadillo bounding box annotations."""
    bbox_file = WCS_DIR / "armadillo_bbox_annotations.json"
    with open(bbox_file) as f:
        data = json.load(f)

    cat_map = {c["id"]: c["name"] for c in data["categories"]}
    img_map = {img["id"]: img for img in data["images"]}

    # Available images (downloaded with safe filenames)
    img_dir = WCS_DIR / "images"
    available_images = {f.name for f in img_dir.glob("*.jpg")} if img_dir.exists() else set()

    print(f"WCS: {len(data['annotations'])} annotations, {len(available_images)} available images")

    entries = []
    for ann in data["annotations"]:
        cat_name = cat_map.get(ann["category_id"], "")
        if cat_name not in WCS_MAPPING:
            continue

        img_info = img_map.get(ann["image_id"])
        if img_info is None:
            continue

        # WCS images were saved with path separators replaced by underscores
        original_fname = img_info.get("file_name", "")
        safe_fname = original_fname.replace("/", "_")

        if safe_fname not in available_images:
            continue

        src_path = img_dir / safe_fname
        bbox = ann["bbox"]
        img_w = img_info.get("width", 0)
        img_h = img_info.get("height", 0)

        if img_w == 0 or img_h == 0 or bbox[2] <= 0 or bbox[3] <= 0:
            continue

        entries.append({
            "src_path": src_path,
            "filename": f"wcs_{safe_fname}",
            "class_id": WCS_MAPPING[cat_name],
            "bbox": bbox,
            "img_w": img_w,
            "img_h": img_h,
            "source": "wcs",
        })

    return entries


def coco_to_yolo(bbox: list, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """Convert COCO bbox [x, y, w, h] to YOLO [x_center, y_center, w, h] (normalized 0-1).

    COCO format: top-left corner (x, y) + width/height in absolute pixels.
    YOLO format: center (x, y) + width/height as fractions of the image dimensions.
    Values are clamped to [0, 1] to handle annotations that slightly exceed image bounds.
    """
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h

    # Clamp to [0, 1]
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    w_norm = max(0.0, min(1.0, w_norm))
    h_norm = max(0.0, min(1.0, h_norm))

    return x_center, y_center, w_norm, h_norm


def main():
    random.seed(RANDOM_SEED)

    # Load both datasets
    print("Loading CCT annotations...")
    cct_entries = load_cct_data()
    print(f"  Matched {len(cct_entries)} CCT entries")

    print("Loading WCS armadillo annotations...")
    wcs_entries = load_wcs_armadillo_data()
    print(f"  Matched {len(wcs_entries)} WCS entries")

    all_entries = cct_entries + wcs_entries

    # Group annotations by image -- a single camera trap image can contain
    # multiple animals, and YOLO expects all boxes for one image in a single .txt file
    from collections import defaultdict
    img_annotations = defaultdict(list)
    img_meta = {}
    for entry in all_entries:
        key = entry["filename"]
        img_annotations[key].append(entry)
        img_meta[key] = entry  # for src_path, img_w, img_h

    print(f"\nTotal unique images: {len(img_annotations)}")

    # Count per class
    class_counts = Counter()
    for entries in img_annotations.values():
        for e in entries:
            class_counts[CLASS_NAMES[e["class_id"]]] += 1

    print("\nAnnotations per class:")
    for name in CLASS_NAMES:
        print(f"  {name:15s} {class_counts.get(name, 0):6d}")
    print(f"  {'TOTAL':15s} {sum(class_counts.values()):6d}")

    # Split into train/val by image (not by annotation, so all boxes from
    # the same image stay in the same split -- prevents data leakage)
    all_image_keys = list(img_annotations.keys())
    random.shuffle(all_image_keys)

    val_count = int(len(all_image_keys) * VAL_SPLIT)
    val_keys = set(all_image_keys[:val_count])
    train_keys = set(all_image_keys[val_count:])

    print(f"\nSplit: {len(train_keys)} train, {len(val_keys)} val")

    # Process and write
    out_dirs = {
        "train": (OUT_DIR / "images" / "train", OUT_DIR / "labels" / "train"),
        "val": (OUT_DIR / "images" / "val", OUT_DIR / "labels" / "val"),
    }

    for split_name, (img_out, lbl_out) in out_dirs.items():
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

    copied = 0
    for img_key in all_image_keys:
        entries = img_annotations[img_key]
        meta = img_meta[img_key]
        split = "val" if img_key in val_keys else "train"
        img_out, lbl_out = out_dirs[split]

        # Copy image
        dst_img = img_out / img_key
        if not dst_img.exists():
            shutil.copy2(meta["src_path"], dst_img)

        # Write YOLO label file -- one line per bounding box:
        # "class_id center_x center_y width height" (all normalized 0-1)
        label_name = Path(img_key).stem + ".txt"
        label_lines = []
        for e in entries:
            xc, yc, w, h = coco_to_yolo(e["bbox"], e["img_w"], e["img_h"])
            label_lines.append(f"{e['class_id']} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        with open(lbl_out / label_name, "w") as f:
            f.write("\n".join(label_lines) + "\n")

        copied += 1
        if copied % 2000 == 0:
            print(f"  Processed {copied}/{len(all_image_keys)} images...")

    print(f"\nDone! Processed {copied} images.")
    print(f"Output: {OUT_DIR}")

    # Count per split
    for split_name, (img_out, lbl_out) in out_dirs.items():
        n_imgs = len(list(img_out.glob("*.jpg")))
        n_lbls = len(list(lbl_out.glob("*.txt")))
        print(f"  {split_name}: {n_imgs} images, {n_lbls} labels")


if __name__ == "__main__":
    main()
