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

MODEL INPUT SIZES:
- ViT-Small: 224x224 pixels
- EfficientNet-Lite4: 300x300 pixels
- The input_size parameter controls this — defaults to 224 (ViT-Small)

NORMALIZATION (same for both models — both pretrained on ImageNet):
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
- RandomResizedCrop(input_size): Randomly crop and resize — simulates different distances
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


# Normalization values for both ViT-Small and EfficientNet-Lite4.
# Both timm models (vit_small_patch16_224.augreg_in21k_ft_in1k and
# tf_efficientnet_lite4.in1k) were pretrained with mean=0.5, std=0.5,
# which maps pixel values from [0,1] to [-1,1].
# This also matches the Hailo Model Zoo .alls scripts: [127.5]/[127.5] on 0-255 scale.
IMAGENET_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STD = [0.5, 0.5, 0.5]
DEFAULT_INPUT_SIZE = 224  # ViT-Small=224, EfficientNet-Lite4=300


def get_train_transforms(input_size: int = DEFAULT_INPUT_SIZE) -> transforms.Compose:
    """
    Training transforms with data augmentation.

    Args:
        input_size: Model input resolution (224 for ViT-Small, 300 for EfficientNet-Lite4)

    Pipeline:
    1. RandomResizedCrop(input_size) — crop random region and resize to input_size x input_size
    2. RandomHorizontalFlip() — 50% chance to mirror horizontally
    3. RandomRotation(15) — rotate up to 15 degrees
    4. ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    5. transforms.ToTensor()
    6. transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms(input_size: int = DEFAULT_INPUT_SIZE) -> transforms.Compose:
    """
    Validation/test transforms — NO randomness.

    Args:
        input_size: Model input resolution (224 for ViT-Small, 300 for EfficientNet-Lite4)

    Pipeline:
    1. Resize(input_size + 32) — resize shortest edge (standard ImageNet eval protocol)
    2. CenterCrop(input_size) — deterministic center crop
    3. transforms.ToTensor()
    4. transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    WHY different from training?
    - No randomness ensures evaluation is reproducible
    - Resize then CenterCrop is the standard ImageNet eval protocol
    """
    return transforms.Compose([
        transforms.Resize(input_size + 32),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_inference_transforms(input_size: int = DEFAULT_INPUT_SIZE) -> transforms.Compose:
    """
    Inference transforms for production use (same as validation).
    Used when classifying birds from camera crops.

    Args:
        input_size: Model input resolution (224 for ViT-Small, 300 for EfficientNet-Lite4)
    """
    return get_val_transforms(input_size)
