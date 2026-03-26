"""
YOUR CODE: Data Augmentation Transforms

Data augmentation artificially expands your training set by applying random
transformations to images. This prevents overfitting — the model learns to
recognize a bird regardless of angle, lighting, or crop.

WHAT TO LEARN:
- torchvision.transforms compose a pipeline of image operations
- Training transforms include RANDOM augmentations (the model sees different
  versions of the same image each epoch)
- Validation/test transforms are DETERMINISTIC (no randomness — you want
  consistent evaluation)
- All pipelines must end with ToTensor() and Normalize()

MobileNetV2 REQUIREMENTS:
- Input size: 224x224 pixels
- Normalize with ImageNet stats: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
  (Because we're using a backbone pretrained on ImageNet, our data must be
  normalized the same way ImageNet was during that pretraining)
- WHY normalize? The backbone's weights were optimized assuming inputs follow
  ImageNet's distribution. Without normalization, the backbone produces garbage
  features, and it doesn't matter how good your classifier head is.
  Normalization shifts each channel: normalized = (pixel - mean) / std
  so values are centered around 0 with consistent spread — exactly what the
  pretrained backbone expects to see.

AUGMENTATIONS TO CONSIDER FOR BIRDS:
- RandomResizedCrop(224): Randomly crop and resize — simulates different distances
- RandomHorizontalFlip(): Birds can face left or right
- RandomRotation(15): Slight tilt — birds don't always perch perfectly level
- ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2): Lighting varies
  throughout the day at your feeder
- RandomAffine: Small translations
- DO NOT use RandomVerticalFlip — birds don't appear upside down at a feeder

ORDER MATTERS:
1. Spatial transforms first (crop, flip, rotate) — these work on PIL Images
2. ColorJitter — also works on PIL Images
3. ToTensor() — converts PIL Image (H,W,C) uint8 [0,255] to Tensor (C,H,W) float [0,1]
4. Normalize() — shifts to ImageNet distribution

DOCS: https://pytorch.org/vision/stable/transforms.html
"""

from torchvision import transforms


# ImageNet normalization values (used because MobileNetV2 was pretrained on ImageNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT_SIZE = 224


def get_train_transforms() -> transforms.Compose:
    """
    Training transforms with data augmentation.

    Pipeline:
    1. RandomResizedCrop(INPUT_SIZE) — crop random region and resize to 224x224
    2. RandomHorizontalFlip() — 50% chance to mirror horizontally
    3. RandomRotation(15) — rotate up to 15 degrees
    4. ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    5. transforms.ToTensor()
    6. transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(INPUT_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms() -> transforms.Compose:
    """
    Validation/test transforms — NO randomness.

    Pipeline:
    1. Resize(256) — resize shortest edge to 256
    2. CenterCrop(INPUT_SIZE) — deterministic center crop to 224x224
    3. transforms.ToTensor()
    4. transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    WHY different from training?
    - No randomness ensures evaluation is reproducible
    - Resize(256) then CenterCrop(224) is the standard ImageNet eval protocol
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_inference_transforms() -> transforms.Compose:
    """
    Inference transforms for production use (same as validation).
    Used when classifying birds from camera crops.
    """
    return get_val_transforms()
