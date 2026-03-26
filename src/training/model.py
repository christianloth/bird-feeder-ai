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

WHAT EACH LAYER GROUP "SEES" (and how fine-tuning changes them):
    Before fine-tuning (pretrained on ImageNet — 1000 generic classes like "bus", "pizza", "tabby cat"):

    Layers 0-6  (early):  Edges, corners, textures, basic color gradients
                           → These are UNIVERSAL visual features. A feather edge looks
                             the same whether you're classifying birds or cars.
                           → Barely change during fine-tuning. Stay frozen.

    Layers 7-13 (middle): Shapes, patterns, eyes, repeated textures
                           → Slightly more task-specific but still broadly useful.
                           → Minor adaptation during fine-tuning. Stay frozen.

    Layers 14-17 (late):  High-level semantic features — "object parts"
                           → BEFORE fine-tuning: generic parts (wheels, fur, petals)
                           → AFTER fine-tuning: bird-specific parts (beak shapes,
                             breast streaking, wing bars, eye rings)
                           → These are the layers we UNFREEZE in Phase 2.

    Classifier head:       The final decision layer
                           → BEFORE: 1000 ImageNet classes (bus, pizza, robin, ...)
                           → AFTER: 555 NABirds species (House Finch, Purple Finch, ...)
                           → Completely REPLACED — this is the first thing we train.

    This is called "catastrophic forgetting" — and here it's INTENTIONAL.
    After fine-tuning, the model will NOT recognize "school bus" or "pizza" anymore.
    But it WILL distinguish a House Finch from a Purple Finch.

    Think of it like a general handyman retraining as a specialized electrician:
    they don't forget how to use a screwdriver (early layers), but they stop
    thinking about plumbing (old classifier) and develop deep expertise in
    wiring (bird-specific features).

WHY MobileNetV2?
- Small (3.4M params) — trains fast, converts well to Hailo
- Designed for mobile/edge — perfect for Raspberry Pi
- Depthwise separable convolutions — efficient but still accurate
- Google uses it for their official iNaturalist bird classifier on Coral TPU
- Well-documented with tons of transfer learning tutorials

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
    # 1. Load MobileNetV2 with or without ImageNet pretrained weights
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v2(weights=weights)

    # 2. Freeze the backbone so only the classifier head trains in Phase 1
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    # 3. Replace the classifier head: Linear(1280, 1000) → Linear(1280, num_classes)
    in_features = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, num_classes),
    )

    return model


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

    WHY ONLY UNFREEZE LATE LAYERS?
        Layer 0-6:  Edges, textures       → universal, no need to change
        Layer 7-13: Shapes, patterns       → mostly universal, leave frozen
        Layer 14-17: High-level features   → THESE need to shift from "generic object
                                              parts" to "bird-specific features" like
                                              beak curvature, feather patterns, eye rings
        Classifier: Final decision         → already trained in Phase 1

        By only unfreezing 14+, we get bird-specific adaptation without
        destroying the valuable low-level feature detectors.

    WHY A LOWER LEARNING RATE?
        The unfrozen backbone layers already have good weights from ImageNet.
        We want to NUDGE them toward birds, not randomly scramble them.
        Typical strategy: use 1/10th of the Phase 1 learning rate.

    TODO:
    1. First, freeze everything: set requires_grad=False for all parameters
    2. Then unfreeze layers from unfreeze_from onwards:
       for layer in model.features[unfreeze_from:]:
           for param in layer.parameters():
               param.requires_grad = True
    3. Also make sure the classifier head is unfrozen
    """
    # Step 1: Freeze everything first (clean slate)
    for param in model.parameters():
        param.requires_grad = False

    # Step 2: Unfreeze late backbone layers (14-17)
    for layer in model.features[unfreeze_from:]:
        for param in layer.parameters():
            param.requires_grad = True

    # Step 3: Unfreeze the classifier head (always needs to be trainable)
    for param in model.classifier.parameters():
        param.requires_grad = True


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
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }
