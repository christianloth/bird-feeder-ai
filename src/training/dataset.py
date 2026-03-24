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
from PIL import Image


class NABirdsDataset(Dataset):
    """PyTorch Dataset for the NABirds bird species dataset."""

    def __init__(self, root_dir: str | Path, split: str = "train", transform=None):
        """
        Args:
            root_dir: Path to the nabirds/ directory
            split: "train" or "test"
            transform: torchvision transforms to apply to images
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform

        # TODO: Parse the dataset files and build:
        # self.samples = []         # List of (image_path, label) tuples
        # self.class_to_species = {} # Dict mapping class_id (int) -> species name (str)
        # self.species_to_class = {} # Dict mapping species name (str) -> class_id (int)

        # STEP 1: Read images.txt to get image_id -> relative_path mapping
        # Format: "image_id  relative/path/to/image.jpg"
        # Hint: with open(self.root_dir / "images.txt") as f: ...

        # STEP 2: Read image_class_labels.txt to get image_id -> class_id mapping
        # Format: "image_id  class_id"

        # STEP 3: Read train_test_split.txt to know which images are train vs test
        # Format: "image_id  is_train" (1 = train, 0 = test)

        # STEP 4: Read classes.txt to get class_id -> species name
        # Format: "class_id  species_name"

        # STEP 5: Filter to only the split we want (train or test)
        # Build self.samples as a list of (full_image_path, class_index) tuples
        # IMPORTANT: class_id in the dataset may not be contiguous (0, 1, 2, ...)
        # You'll need to remap them to contiguous indices for CrossEntropyLoss

        raise NotImplementedError("Implement me! Read the hints above.")

    def __len__(self) -> int:
        # TODO: Return the number of samples
        raise NotImplementedError("Implement me!")

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        # TODO:
        # 1. Get the (image_path, label) for this index
        # 2. Load the image with PIL
        # 3. Apply self.transform if it exists
        # 4. Return (image_tensor, label)
        raise NotImplementedError("Implement me!")

    @property
    def num_classes(self) -> int:
        """Return the number of unique species classes."""
        # TODO: Return the count of unique classes
        raise NotImplementedError("Implement me!")

    def get_species_name(self, class_idx: int) -> str:
        """Convert a class index back to a species name for display."""
        # TODO: Look up the species name from the class index
        raise NotImplementedError("Implement me!")
