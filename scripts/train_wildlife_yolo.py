"""Train YOLOv8n for wildlife detection.

Trains a YOLOv8n model on the merged Caltech + WCS camera trap dataset.
All paths are absolute so this can be run from any working directory.

Balancing modes:
    --balanced   Inverse-frequency weighting: rare classes sampled more often,
                 but not perfectly equal. Good general-purpose option.
    --equal      True equalization: every class is sampled equally per epoch.
                 Best for single-class-per-image datasets (like trail cameras).

Usage:
    python scripts/train_wildlife_yolo.py
    python scripts/train_wildlife_yolo.py --resume
    python scripts/train_wildlife_yolo.py --epochs 50 --batch 8
    python scripts/train_wildlife_yolo.py --balanced
    python scripts/train_wildlife_yolo.py --equal
    python scripts/train_wildlife_yolo.py --equal --min-samples 50
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
import ultralytics.data.build as build

PROJECT_ROOT = Path("/Users/christianloth/Documents/Programs/bird-feeder-ai")
DATASET_YAML = PROJECT_ROOT / "data" / "wildlife-yolo" / "dataset.yaml"
OUTPUT_DIR = PROJECT_ROOT / "models" / "wildlife"


class YOLOWeightedDataset(YOLODataset):
    """YOLODataset with weighted sampling to address class imbalance.

    Images containing rare classes are sampled more frequently during training.
    Weights are computed as inverse class frequency: total_instances / class_count.
    Only affects training -- validation uses sequential indexing as normal.

    Based on: https://y-t-g.github.io/tutorials/yolo-class-balancing/
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.train_mode = "train" in self.prefix

        self._count_instances()
        class_weights = np.sum(self.counts) / self.counts

        self.class_weights = np.array(class_weights)
        self._image_weights = self._calculate_image_weights()
        self._probabilities = self._calculate_probabilities()

    def _count_instances(self) -> None:
        """Count annotation instances per class across the dataset."""
        self.counts = np.zeros(len(self.data["names"]), dtype=np.float64)
        for label in self.labels:
            cls = label["cls"].reshape(-1).astype(int)
            for class_id in cls:
                self.counts[class_id] += 1
        # Floor zero-count classes to 1 to avoid division by zero
        self.counts = np.where(self.counts == 0, 1, self.counts)

    def _calculate_image_weights(self) -> list[float]:
        """Assign each image a weight based on the classes it contains."""
        weights = []
        for label in self.labels:
            cls = label["cls"].reshape(-1).astype(int)
            if cls.size == 0:
                weights.append(1.0)
                continue
            weight = float(np.mean(self.class_weights[cls]))
            weights.append(weight)
        return weights

    def _calculate_probabilities(self) -> list[float]:
        """Normalize image weights into sampling probabilities."""
        total = sum(self._image_weights)
        return [w / total for w in self._image_weights]

    def __getitem__(self, index):
        """Override index selection to use weighted sampling for training."""
        if not self.train_mode:
            return self.transforms(self.get_image_and_label(index))
        index = np.random.choice(len(self.labels), p=self._probabilities)
        return self.transforms(self.get_image_and_label(index))


class YOLOEqualDataset(YOLODataset):
    """YOLODataset with equal class sampling at a fixed count per class.

    Each class gets exactly `samples_per_class` samples per epoch.
    A class is chosen uniformly at random, then an image from that class
    is chosen uniformly at random. Classes with fewer images than the
    target will have images repeated (with different augmentations).

    Set the class variable `samples_per_class` before training starts.

    Best suited for datasets where each image contains a single class
    (e.g., trail camera images with one animal per trigger).
    """

    samples_per_class: int = 1000  # set from --equal N before training

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.train_mode = "train" in self.prefix
        self._build_class_index()

    def _build_class_index(self) -> None:
        """Build a mapping from class_id -> list of image indices."""
        num_classes = len(self.data["names"])
        self._class_to_indices: dict[int, list[int]] = {c: [] for c in range(num_classes)}

        for img_idx, label in enumerate(self.labels):
            cls = label["cls"].reshape(-1).astype(int)
            for class_id in cls:
                self._class_to_indices[class_id].append(img_idx)

        # Remove empty classes (no images)
        self._active_classes = [c for c in self._class_to_indices if self._class_to_indices[c]]
        n = self.samples_per_class

        print(f"Equal sampling: {n} samples/class, "
              f"{len(self._active_classes)} classes, "
              f"{n * len(self._active_classes)} total samples/epoch:")
        names = self.data["names"]
        for c in self._active_classes:
            actual = len(self._class_to_indices[c])
            repeat = f"({n / actual:.1f}x oversample)" if actual < n else ""
            print(f"  {c:2d} {names[c]:15s} {actual:6d} images  {repeat}")

    def __len__(self) -> int:
        """Epoch size = num_classes * samples_per_class."""
        if not self.train_mode:
            return super().__len__()
        return len(self._active_classes) * self.samples_per_class

    def __getitem__(self, index):
        """Pick a random class, then a random image from that class."""
        if not self.train_mode:
            return self.transforms(self.get_image_and_label(index))
        class_id = np.random.choice(self._active_classes)
        index = np.random.choice(self._class_to_indices[class_id])
        return self.transforms(self.get_image_and_label(index))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8n wildlife detector")
    parser.add_argument("--model", default="yolov8n.pt", help="Base model (default: yolov8n.pt)")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="mps", help="Device: mps, cuda, cpu")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--name", default="yolov8n-wildlife", help="Run name")
    parser.add_argument(
        "--balanced", action="store_true",
        help="Use inverse-frequency weighted dataloader to oversample rare classes",
    )
    parser.add_argument(
        "--equal", type=int, metavar="N",
        help="Equal class sampling: N samples per class per epoch (e.g., --equal 1000)",
    )
    parser.add_argument(
        "--min-samples", type=int, default=0,
        help="Warn about classes with fewer than this many training annotations (default: 0)",
    )
    return parser.parse_args()


def print_class_warnings(min_samples: int) -> None:
    """Read the dataset labels and warn about underrepresented classes."""
    import yaml

    with open(DATASET_YAML) as f:
        ds = yaml.safe_load(f)

    label_dir = Path(ds["path"]) / "labels" / "train"
    if not label_dir.exists():
        return

    names = ds["names"]
    counts = {i: 0 for i in names}

    for txt_file in label_dir.iterdir():
        if txt_file.suffix != ".txt":
            continue
        for line in txt_file.read_text().splitlines():
            parts = line.strip().split()
            if parts:
                class_id = int(parts[0])
                counts[class_id] = counts.get(class_id, 0) + 1

    print("\nTraining set class distribution:")
    for class_id in sorted(names.keys()):
        name = names[class_id]
        count = counts.get(class_id, 0)
        flag = "  *** LOW ***" if min_samples > 0 and count < min_samples else ""
        print(f"  {class_id:2d} {name:15s} {count:6d}{flag}")

    if min_samples > 0:
        low = [names[i] for i in names if counts.get(i, 0) < min_samples]
        if low:
            print(f"\nWARNING: {len(low)} class(es) below --min-samples={min_samples}: "
                  f"{', '.join(low)}")
            print("These classes may not be learned reliably. Consider dropping them "
                  "from the dataset or collecting more data.\n")


def main() -> None:
    args = parse_args()

    if args.balanced and args.equal:
        raise ValueError("--balanced and --equal are mutually exclusive")

    if args.balanced:
        print("Balanced mode: inverse-frequency weighted sampling")
        build.YOLODataset = YOLOWeightedDataset
    elif args.equal:
        print(f"Equal mode: {args.equal} samples per class per epoch")
        YOLOEqualDataset.samples_per_class = args.equal
        build.YOLODataset = YOLOEqualDataset

    if args.balanced or args.equal or args.min_samples > 0:
        print_class_warnings(args.min_samples)

    if args.resume:
        last_weights = OUTPUT_DIR / args.name / "weights" / "last.pt"
        if not last_weights.exists():
            raise FileNotFoundError(f"No checkpoint to resume from: {last_weights}")
        # Patch checkpoint so resume writes to our project dir, not whatever
        # save_dir was baked in from a previous run/environment.
        import torch
        ckpt = torch.load(str(last_weights), map_location="cpu", weights_only=False)
        if "train_args" in ckpt:
            ckpt["train_args"]["save_dir"] = str(OUTPUT_DIR / args.name)
            ckpt["train_args"]["project"] = str(OUTPUT_DIR)
            ckpt["train_args"]["name"] = args.name
            ckpt["train_args"]["exist_ok"] = True
            torch.save(ckpt, str(last_weights))
            print(f"Patched checkpoint save_dir → {OUTPUT_DIR / args.name}")
        model = YOLO(str(last_weights))
    else:
        model = YOLO(args.model)

    print(f"Device: {args.device}")
    print(f"Output: {OUTPUT_DIR / args.name}")
    print(f"Resume: {args.resume}")

    train_kwargs = dict(
        data=str(DATASET_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(OUTPUT_DIR),
        name=args.name,
        exist_ok=True,
        patience=args.patience,
        save=True,
        save_period=10,
        plots=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        erasing=0.2,
        workers=args.workers,
        verbose=True,
    )

    if args.resume:
        train_kwargs["resume"] = True

    results = model.train(**train_kwargs)

    print("Training complete!")
    print(f"Best model: {results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    main()
