"""
Export trained PyTorch models to ONNX format.

ONNX (Open Neural Network Exchange) is an intermediate format that Hailo's
DFC compiler can convert to HEF for the NPU. The chain is:

    PyTorch (.pth) → ONNX (.onnx) → Hailo DFC → HEF (.hef)

Supports two model types:
- Classification (ViT-Small, EfficientNet-Lite4): custom training checkpoint
- YOLO detection (YOLOv8n, YOLO11s, etc.): Ultralytics .pt file

WHAT TO LEARN:
- torch.onnx.export traces your model with a dummy input
- The dummy input must match the exact shape your model expects
- opset_version controls which ONNX operators are available (11 works with Hailo)
- YOLO models use Ultralytics' built-in exporter because they have complex
  post-processing layers (detection heads, NMS) that need special handling
"""

from pathlib import Path

import torch
import torch.nn as nn


def export_classifier_to_onnx(
    model: nn.Module,
    output_path: str | Path,
    num_classes: int,
    input_size: int = 224,
    opset_version: int = 17,
) -> Path:
    """
    Export a trained classification model (ViT-Small/EfficientNet-Lite4) to ONNX.

    Args:
        model: Trained classifier model.
        output_path: Where to save the .onnx file.
        num_classes: Number of output classes.
        input_size: Input image dimension (224 for ViT-Small, 300 for EfficientNet-Lite4).
        opset_version: ONNX opset (17 for ViT — matches Hailo Model Zoo).

    Returns:
        Path to the exported ONNX file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()

    dummy_input = torch.randn(1, 3, input_size, input_size)

    # Use legacy TorchScript exporter (dynamo=False) to avoid onnxscript
    # dependency conflict with norfair's numpy<2.0 pin
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
        dynamo=False,
    )

    # Verify the exported model if onnx is available
    try:
        import onnx
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print("  ONNX validation: passed")
    except ImportError:
        print("  ONNX validation: skipped (onnx package not installed)")

    print(f"ONNX model exported to: {output_path}")
    print(f"  Input shape:  (batch, 3, {input_size}, {input_size})")
    print(f"  Output shape: (batch, {num_classes})")
    print(f"  File size:    {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    return output_path


def export_yolo_to_onnx(
    pt_path: str | Path,
    output_path: str | Path | None = None,
    imgsz: int = 640,
    opset_version: int = 11,
) -> Path:
    """
    Export a YOLO model (.pt) to ONNX using Ultralytics' built-in exporter.

    YOLO models have complex detection heads and NMS layers that require
    Ultralytics' export pipeline -- a raw torch.onnx.export won't work.

    Args:
        pt_path: Path to the Ultralytics .pt weights file.
        output_path: Where to save the .onnx file. If None, saves next to
            the .pt file and moves to models/onnx/.
        imgsz: Input image size (640 for standard YOLO).
        opset_version: ONNX opset (11 recommended for Hailo).

    Returns:
        Path to the exported ONNX file.
    """
    from ultralytics import YOLO
    import shutil

    pt_path = Path(pt_path)
    if not pt_path.exists():
        raise FileNotFoundError(f"YOLO weights not found: {pt_path}")

    model = YOLO(str(pt_path))
    model.export(format="onnx", opset=opset_version, imgsz=imgsz)

    # Ultralytics saves the .onnx next to the .pt file
    exported = pt_path.with_suffix(".onnx")
    if not exported.exists():
        raise RuntimeError(f"Expected ONNX output at {exported} but not found")

    # Move to target location
    if output_path is None:
        output_dir = Path("models/onnx")
        output_dir.mkdir(parents=True, exist_ok=True)
        # Derive a descriptive name from the parent directory
        # e.g., models/wildlife/yolo11s-wildlife-equal/weights/best.pt → yolo11s_wildlife.onnx
        model_name = pt_path.parent.parent.name  # "yolo11s-wildlife-equal"
        # Simplify: take the architecture part before the first hyphen-separated descriptor
        output_path = output_dir / f"{model_name.replace('-', '_')}.onnx"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    shutil.move(str(exported), str(output_path))

    print(f"YOLO ONNX model exported to: {output_path}")
    print(f"  Input size:  {imgsz}x{imgsz}")
    print(f"  File size:   {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export trained model to ONNX")
    subparsers = parser.add_subparsers(dest="type", required=True)

    # --- Classifier subcommand ---
    cls_parser = subparsers.add_parser(
        "classifier",
        help="Export a classification model (ViT-Small, EfficientNet-Lite4)",
    )
    cls_parser.add_argument(
        "--model", type=str, default="vit_small",
        choices=["vit_small", "efficientnet_lite4"],
        help="Model architecture (default: vit_small)",
    )
    cls_parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to checkpoint. If omitted, finds the latest in models/bird-classifier/<model>/",
    )
    cls_parser.add_argument(
        "--output", type=str, default=None,
        help="Output path. Default: models/onnx/<model>_birds.onnx",
    )

    # --- YOLO subcommand ---
    yolo_parser = subparsers.add_parser(
        "yolo",
        help="Export a YOLO detection model (.pt)",
    )
    yolo_parser.add_argument(
        "--weights", type=str, required=True,
        help="Path to Ultralytics .pt weights file",
    )
    yolo_parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Input image size (default: 640)",
    )
    yolo_parser.add_argument(
        "--output", type=str, default=None,
        help="Output path. Default: models/onnx/<model_name>.onnx",
    )

    args = parser.parse_args()

    if args.type == "classifier":
        from src.training.model import create_model, get_model_config

        model_config = get_model_config(args.model)
        num_classes = 555  # NABirds

        if args.checkpoint:
            checkpoint_path = Path(args.checkpoint)
        else:
            model_dir = Path("models/bird-classifier") / args.model
            best_models = sorted(model_dir.glob("*/best_model.pth"))
            if not best_models:
                raise FileNotFoundError(f"No best_model.pth found in {model_dir}/*/")
            checkpoint_path = best_models[-1]

        print(f"Loading classifier from: {checkpoint_path}")
        model = create_model(
            num_classes=num_classes, pretrained=False,
            freeze_backbone=False, model_name=args.model,
        )
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))

        output = args.output or f"models/onnx/{args.model}_birds.onnx"
        export_classifier_to_onnx(
            model=model,
            output_path=output,
            num_classes=num_classes,
            input_size=model_config["input_size"],
        )

    elif args.type == "yolo":
        export_yolo_to_onnx(
            pt_path=args.weights,
            output_path=args.output,
            imgsz=args.imgsz,
        )
