"""
YOUR CODE: Transfer Learning with ViT-Small / ViT-Base / EfficientNet-Lite4

Transfer learning is the key insight that makes this project feasible:
instead of training a model from scratch (which would need millions of images),
we take a model already trained on ImageNet (1.2M images, 1000 classes) and
adapt it for bird species.

WHAT TO LEARN:
- Pretrained models have two parts: BACKBONE (feature extractor) and HEAD (classifier)
- The backbone has already learned to detect edges, textures, shapes, eyes, feathers, etc.
- We replace the head with our own (for 555 bird species instead of 1000 ImageNet classes)
- "Freezing" means setting requires_grad=False — those weights won't update during training

SUPPORTED MODELS:

  ViT-Small ARCHITECTURE (primary — best for fine-grained bird classification):
    Input (3, 224, 224)
        → patch_embed: split image into 14x14 grid of 16x16 patches → 196 tokens
        → 12 transformer encoder blocks (dim=384, 6 heads) with self-attention
        → head (classifier): Linear(384, 1000)  ← WE REPLACE THIS

    Medium (21.1M params). Self-attention captures long-range relationships between
    distant bird features (e.g., bill shape ↔ tail pattern). Validated on NABirds:
    ViT achieves 89.9% on NABirds (He et al., TransFG, AAAI 2022).
    Training strategy: SINGLE-PHASE end-to-end with lower LR + warmup for
    transformer stability.

  ViT-Base ARCHITECTURE (higher-capacity alternative):
    Input (3, 224, 224)
        → patch_embed: split image into 14x14 grid of 16x16 patches → 196 tokens
        → 12 transformer encoder blocks (dim=768, 12 heads) with self-attention
        → head (classifier): Linear(768, 1000)  ← WE REPLACE THIS

    Large (86.5M params). Same patch grid as ViT-Small but wider hidden dim
    and more attention heads — should yield higher accuracy at ~4x the params
    and ~2x the memory. Hailo Model Zoo reports 84.5% float / 83.6% hardware
    top-1 on ImageNet-1K at 57 FPS (single pass) on Hailo10H.
    Training strategy: same as ViT-Small (low LR, end-to-end).

  EfficientNet-Lite4 ARCHITECTURE (runner-up — CNN alternative):
    Input (3, 300, 300)
        → features (backbone): compound-scaled blocks with ReLU6 (no SE blocks) → (1280, 10, 10)
        → avgpool: (1280, 10, 10) → (1280,)
        → classifier (head): Linear(1280, 1000)  ← WE REPLACE THIS

    Medium (13.0M params), 300x300 resolution captures fine plumage detail.
    Hardware-friendly: no squeeze-and-excitation blocks, ReLU6 instead of swish.
    Runs in a single pass on Hailo-10H NPU (confirmed in Hailo Model Zoo).
    Training strategy: SINGLE-PHASE end-to-end.

WHY ViT-Small? (primary)
- Self-attention excels at fine-grained classification (TransFG, AAAI 2022)
- Multi-head attention naturally discovers discriminative bird body parts
- 80.5% HW accuracy on Hailo-10H at 116 FPS (single pass, confirmed)
- Validated on NABirds — the exact dataset this project uses

WHY EfficientNet-Lite4? (runner-up)
- CNN with different error profile vs ViT (complementary strengths)
- 300x300 resolution — more pixels on fine plumage details
- 80.1% HW accuracy on Hailo-10H at 137 FPS (single pass, confirmed)
- Designed for edge accelerators — no SE blocks that cause multi-pass issues

DOCS:
- https://huggingface.co/timm/vit_small_patch16_224.augreg_in21k_ft_in1k
- https://huggingface.co/timm/vit_base_patch16_224.augreg_in21k_ft_in1k
- https://github.com/huggingface/pytorch-image-models
- https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
"""

import torch.nn as nn

# Model configurations — the single source of truth for each architecture.
# When you add a new model, add its config here and handle it in create_model().
MODEL_CONFIGS: dict[str, dict] = {
    "vit_small": {
        "input_size": 224,
        "feature_dim": 384,
        "dropout": 0.1,
        "timm_model": "vit_small_patch16_224.augreg_in21k_ft_in1k",
    },
    "vit_base": {
        "input_size": 224,
        "feature_dim": 768,
        "dropout": 0.1,
        "timm_model": "vit_base_patch16_224.augreg_in21k_ft_in1k",
    },
    "efficientnet_lite4": {
        "input_size": 300,
        "feature_dim": 1280,
        "dropout": 0.3,
        "timm_model": "tf_efficientnet_lite4.in1k",
    },
}


def get_model_config(model_name: str) -> dict:
    """Return configuration for the given model architecture."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model: {model_name}. Choose from: {list(MODEL_CONFIGS.keys())}"
        )
    return MODEL_CONFIGS[model_name]


def create_model(
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    model_name: str = "vit_small",
) -> nn.Module:
    """
    Create a model adapted for bird species classification.

    Args:
        num_classes: Number of bird species (555 for NABirds)
        pretrained: Load ImageNet pretrained weights (via timm)
        freeze_backbone: If True, freeze the feature extraction layers
        model_name: Which architecture — "vit_small", "vit_base", or "efficientnet_lite4"

    Returns:
        Modified model with new classifier head
    """
    import timm

    config = get_model_config(model_name)

    # Load model via timm with pretrained weights and custom head
    model = timm.create_model(
        config["timm_model"],
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=config["dropout"],
    )

    # Optionally freeze backbone (everything except the classifier head)
    if freeze_backbone:
        for name, param in model.named_parameters():
            # timm uses "head" for the classifier in both ViT and EfficientNet
            if "head" not in name and "classifier" not in name:
                param.requires_grad = False

    # Store model name for downstream use
    model._model_name = model_name

    return model


def count_parameters(model: nn.Module) -> dict:
    """
    Count trainable vs frozen parameters. Useful for verifying freeze/unfreeze.

    Returns a dict with:
    - "total": total number of parameters
    - "trainable": parameters with requires_grad=True
    - "frozen": parameters with requires_grad=False
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }
