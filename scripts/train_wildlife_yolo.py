"""Train YOLOv8n for wildlife detection.

Trains a YOLOv8n model on the merged Caltech + WCS camera trap dataset
(13 wildlife classes). All paths are absolute so this can be run from
any working directory.

Usage:
    python scripts/train_wildlife_yolo.py
    python scripts/train_wildlife_yolo.py --resume
    python scripts/train_wildlife_yolo.py --epochs 50 --batch 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path("/Users/christianloth/Documents/Programs/bird-feeder-ai")
DATASET_YAML = PROJECT_ROOT / "data" / "wildlife-yolo" / "dataset.yaml"
OUTPUT_DIR = PROJECT_ROOT / "models" / "wildlife"


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.resume:
        last_weights = OUTPUT_DIR / args.name / "weights" / "last.pt"
        if not last_weights.exists():
            raise FileNotFoundError(f"No checkpoint to resume from: {last_weights}")
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
