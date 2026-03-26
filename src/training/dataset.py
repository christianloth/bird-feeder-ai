"""
YOUR CODE: PyTorch Dataset for NABirds

This is the first file you'll implement. A PyTorch Dataset teaches you how
data flows into a neural network.

WHAT TO LEARN:
- torch.utils.data.Dataset requires __init__, __len__, __getitem__
- __getitem__ must return (image_tensor, label_integer) for classification
- Images need to be loaded, transformed, and converted to tensors
- Labels need to be mapped from string species names to integer indices

NABirds DATASET STRUCTURE (after download):
    nabirds/
        images/
            0001/           # species folder
                img1.jpg
                img2.jpg
            0002/
                ...
        images.txt          # image_id  path
        classes.txt         # class_id  class_name
        train_test_split.txt  # image_id  is_train (1 or 0)
        image_class_labels.txt  # image_id  class_id

STEPS:
1. In __init__, parse the text files to build:
   - A list of (image_path, label) tuples
   - A mapping of class_id -> species_name (for display)
   - Filter by train/test split

2. In __len__, return the total number of images

3. In __getitem__, given an index:
   - Load the image from disk using PIL
   - Apply transforms (you'll define these in transforms.py)
   - Return (transformed_image, label)

HINTS:
- Use PIL.Image.open(path).convert("RGB") to load images
- The transform pipeline will handle converting PIL -> Tensor
- Keep the class_to_species dict so you can display results later
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class NABirdsDataset(Dataset):
    """PyTorch Dataset for the NABirds bird species dataset."""

    def __init__(self, root_dir: str | Path, split: str = "train", transform: transforms.Compose | None = None):
        """
        Args:
            root_dir: Path to the nabirds/ directory
            split: "train" or "test"
            transform: torchvision transforms to apply to images
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform

        # STEP 1: Read images.txt to get image_id -> relative_path mapping
        # Format: "image_id  relative/path/to/image.jpg"
        image_paths = {}
        with open(self.root_dir / "images.txt") as f:
            for line in f:
                image_id, path = line.strip().split(maxsplit=1)
                image_paths[image_id] = path

        # STEP 2: Read image_class_labels.txt to get image_id -> class_id mapping
        # Format: "image_id  class_id"
        image_labels = {}
        with open(self.root_dir / "image_class_labels.txt") as f:
            for line in f:
                image_id, class_id = line.strip().split()
                image_labels[image_id] = int(class_id)

        # STEP 3: Read train_test_split.txt to know which images are train vs test
        # Format: "image_id  is_train" (1 = train, 0 = test)
        train_test = {}
        with open(self.root_dir / "train_test_split.txt") as f:
            for line in f:
                image_id, is_train = line.strip().split()
                train_test[image_id] = int(is_train)

        # STEP 4: Read classes.txt to get class_id -> species name
        # Format: "class_id  species_name"
        class_names = {}
        with open(self.root_dir / "classes.txt") as f:
            for line in f:
                class_id, name = line.strip().split(maxsplit=1)
                class_names[int(class_id)] = name

        # STEP 5: Filter to only the split we want (train or test)
        # and remap class_ids to contiguous indices for CrossEntropyLoss.
        # WHY REMAP? NABirds class_ids are non-contiguous (e.g., 23, 27, 645, 817...).
        # CrossEntropyLoss expects labels 0, 1, 2, ... N-1. So we collect all
        # unique class_ids that appear in our split and map them to 0..N-1.
        is_train = 1 if split == "train" else 0
        split_image_ids = [img_id for img_id, flag in train_test.items() if flag == is_train]

        # Collect unique class_ids in this split and sort for deterministic mapping
        unique_class_ids = sorted({image_labels[img_id] for img_id in split_image_ids})
        self._class_id_to_idx = {cid: idx for idx, cid in enumerate(unique_class_ids)}
        self._idx_to_class_id = {idx: cid for cid, idx in self._class_id_to_idx.items()}

        # Build the species name lookups using contiguous indices
        self.class_to_species = {
            self._class_id_to_idx[cid]: name
            for cid, name in class_names.items()
            if cid in self._class_id_to_idx
        }
        self.species_to_class = {name: idx for idx, name in self.class_to_species.items()}

        # Build samples list: (full_image_path, contiguous_label)
        self.samples = []
        for img_id in split_image_ids:
            path = self.root_dir / "images" / image_paths[img_id]
            label = self._class_id_to_idx[image_labels[img_id]]
            self.samples.append((path, label))

    def __len__(self) -> int:
        # PyTorch's DataLoader calls this to know how many samples exist.
        # It uses this to determine how many batches make up one epoch.
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        # PyTorch's DataLoader calls this repeatedly to fetch individual samples.
        # It then collates them into batches (e.g., 32 images at a time).
        # Flow: DataLoader asks for sample[0], sample[1], ... sample[31]
        #       → __getitem__ loads each image from disk and applies transforms
        #       → DataLoader stacks them into a batch tensor (32, 3, 224, 224)
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    @property
    def num_classes(self) -> int:
        """Return the number of unique species classes."""
        return len(self.class_to_species)

    def get_species_name(self, class_idx: int) -> str:
        """Convert a class index back to a species name for display."""
        return self.class_to_species[class_idx]
