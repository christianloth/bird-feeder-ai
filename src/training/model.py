"""
YOUR CODE: Transfer Learning with MobileNetV2

Transfer learning is the key insight that makes this project feasible:
instead of training a model from scratch (which would need millions of images),
we take a model already trained on ImageNet (1.2M images, 1000 classes) and
adapt it for bird species.

WHAT TO LEARN:
- Pretrained models have two parts: BACKBONE (feature extractor) and HEAD (classifier)
- The backbone has already learned to detect edges, textures, shapes, eyes, feathers, etc.
- We replace the head with our own (for 555 bird species instead of 1000 ImageNet classes)
- "Freezing" means setting requires_grad=False — those weights won't update during training
- Strategy: freeze backbone first (train only the head), then unfreeze and fine-tune everything

MobileNetV2 ARCHITECTURE:
    Input (3, 224, 224)
        → features (backbone): 18 inverted residual blocks → (1280, 7, 7)
        → avgpool: (1280, 7, 7) → (1280,)
        → classifier (head): Linear(1280, 1000)  ← WE REPLACE THIS

    We change: classifier = Linear(1280, num_species)

WHY MobileNetV2?
- Small (3.4M params) — trains fast, converts well to Hailo
- Designed for mobile/edge — perfect for Raspberry Pi
- Depthwise separable convolutions — efficient but still accurate

DOCS:
- https://pytorch.org/vision/stable/models/mobilenetv2.html
- https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
"""

import torch.nn as nn
from torchvision import models


def create_model(num_classes: int, pretrained: bool = True, freeze_backbone: bool = True) -> nn.Module:
    """
    Create a MobileNetV2 model adapted for bird species classification.

    Args:
        num_classes: Number of bird species (555 for NABirds)
        pretrained: Load ImageNet pretrained weights
        freeze_backbone: If True, freeze the feature extraction layers

    Returns:
        Modified MobileNetV2 model

    TODO:
    1. Load pretrained MobileNetV2:
       model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
       (if pretrained=True, otherwise weights=None)

    2. Freeze the backbone (if freeze_backbone=True):
       Loop through model.features.parameters() and set requires_grad = False
       This means only the classifier head will be trained initially

    3. Replace the classifier head:
       The original head is: model.classifier = Sequential(Dropout(0.2), Linear(1280, 1000))
       Replace it with:      model.classifier = Sequential(Dropout(0.2), Linear(1280, num_classes))

       Access the in_features from the existing layer:
       in_features = model.classifier[1].in_features  # Should be 1280

    4. Return the model
    """
    raise NotImplementedError("Implement me!")


def unfreeze_backbone(model: nn.Module, unfreeze_from: int = 14) -> None:
    """
    Unfreeze backbone layers for fine-tuning (Phase 2 of training).

    After the classifier head is trained, we unfreeze the backbone and train
    the entire model with a LOWER learning rate. This lets the backbone adapt
    its features specifically for birds.

    Args:
        model: The MobileNetV2 model
        unfreeze_from: Unfreeze from this layer index onwards.
            MobileNetV2 has 18 feature blocks (0-17).
            Unfreezing from 14 means: layers 14-17 + classifier are trainable.
            Earlier layers (0-13) stay frozen — they detect basic features
            (edges, textures) that are universal.

    TODO:
    1. First, freeze everything: set requires_grad=False for all parameters
    2. Then unfreeze layers from unfreeze_from onwards:
       for layer in model.features[unfreeze_from:]:
           for param in layer.parameters():
               param.requires_grad = True
    3. Also make sure the classifier head is unfrozen
    """
    raise NotImplementedError("Implement me!")


def count_parameters(model: nn.Module) -> dict:
    """
    Count trainable vs frozen parameters. Useful for verifying freeze/unfreeze.

    TODO:
    Return a dict with:
    - "total": total number of parameters
    - "trainable": parameters with requires_grad=True
    - "frozen": parameters with requires_grad=False

    Hint: sum(p.numel() for p in model.parameters())
    """
    raise NotImplementedError("Implement me!")
