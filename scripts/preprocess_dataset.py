"""
Pre-process NABirds images with full transforms applied.
Saves ready-to-train float32 tensors (224x224, normalized) to disk,
eliminating ALL CPU work during training.

Trade-off: augmentation is "frozen" — each image gets one fixed random
augmentation rather than a different one every epoch.

Usage:
    python scripts/preprocess_dataset.py
    python scripts/preprocess_dataset.py --data-dir data/nabirds --output-dir data/nabirds/preprocessed
"""

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from src.training.dataset import NABirdsDataset
from src.training.transforms import get_train_transforms, get_val_transforms


def preprocess(data_dir: Path, output_dir: Path) -> None:
    """Apply full transforms and save as ready-to-train tensors."""
    splits = {
        "train": get_train_transforms(),
        "test": get_val_transforms(),
    }

    for split, transform in splits.items():
        dataset = NABirdsDataset(data_dir, split=split, transform=transform)
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nPreprocessing {split} split ({len(dataset)} images)...")
        for idx in tqdm(range(len(dataset))):
            image, label = dataset[idx]
            torch.save((image, label), split_dir / f"{idx:06d}.pt")

        torch.save({
            "num_classes": dataset.num_classes,
            "class_to_species": dataset.class_to_species,
            "num_samples": len(dataset),
        }, output_dir / f"{split}_metadata.pt")
        print(f"  Saved {len(dataset)} tensors to {split_dir}")

    print("\nPreprocessing complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-process NABirds dataset")
    parser.add_argument("--data-dir", type=Path, default=Path("data/nabirds"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/nabirds/preprocessed"))
    args = parser.parse_args()
    preprocess(args.data_dir, args.output_dir)
