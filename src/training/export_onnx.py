"""
Export trained PyTorch model to ONNX format.

ONNX (Open Neural Network Exchange) is an intermediate format that Hailo's
DFC compiler can convert to HEF for the NPU. The chain is:

    PyTorch (.pth) → ONNX (.onnx) → Hailo DFC → HEF (.hef)

WHAT TO LEARN:
- torch.onnx.export traces your model with a dummy input
- The dummy input must match the exact shape your model expects
- opset_version controls which ONNX operators are available (11 works with Hailo)
- You can verify the exported model with onnx.checker
"""

from pathlib import Path

import torch
import torch.nn as nn
import onnx


def export_to_onnx(
    model: nn.Module,
    output_path: str | Path,
    num_classes: int,
    input_size: int = 224,
    opset_version: int = 11,
) -> None:
    """
    Export a trained PyTorch model to ONNX format.

    Args:
        model: Trained MobileNetV2 model
        output_path: Where to save the .onnx file
        num_classes: Number of output classes
        input_size: Input image dimension (224 for MobileNetV2)
        opset_version: ONNX opset (11 recommended for Hailo)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()

    # Create a dummy input matching the expected shape: (batch, channels, height, width)
    dummy_input = torch.randn(1, 3, input_size, input_size)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        opset_version=opset_version,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    # Verify the exported model
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    print(f"ONNX model exported to: {output_path}")
    print(f"  Input shape:  (batch, 3, {input_size}, {input_size})")
    print(f"  Output shape: (batch, {num_classes})")
    print(f"  File size:    {output_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    import argparse
    from src.training.model import create_model, get_model_config

    parser = argparse.ArgumentParser(description="Export trained model to ONNX")
    parser.add_argument(
        "--model", type=str, default="mobilenetv2",
        choices=["mobilenetv2", "efficientnet_b2"],
        help="model architecture (default: mobilenetv2)",
    )
    args = parser.parse_args()

    model_config = get_model_config(args.model)
    num_classes = 555  # NABirds

    # Load your trained model
    checkpoint_path = Path("models/checkpoints") / args.model / "best_model.pth"

    model = create_model(
        num_classes=num_classes, pretrained=False,
        freeze_backbone=False, model_name=args.model,
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

    export_to_onnx(
        model=model,
        output_path=f"models/onnx/{args.model}_birds.onnx",
        num_classes=num_classes,
        input_size=model_config["input_size"],
    )
