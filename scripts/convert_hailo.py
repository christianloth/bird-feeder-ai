"""
Convert ONNX model to Hailo HEF format.

Pipeline: ONNX (float32) → HAR (parsed) → HAR (INT8 quantized) → HEF (compiled)

REQUIREMENTS:
- Must run on x86_64 Linux (Hailo DFC does not support macOS or ARM)
- Install: pip install hailo_sdk_client hailo_model_zoo
- Free Hailo Developer Zone account: https://hailo.ai/developer-zone/

The resulting .hef file can then be deployed on any Hailo device
(Raspberry Pi AI HAT+, Hailo-8, etc.).

USAGE:
    python scripts/convert_hailo.py --onnx models/onnx/efficientnet_b2_birds.onnx

    # Custom input size and calibration directory
    python scripts/convert_hailo.py --onnx models/onnx/efficientnet_b2_birds.onnx \\
        --input-size 260 --calib-dir data/nabirds/images

NORMALIZATION:
    ImageNet normalization is baked into the HEF via a model script so that
    the NPU handles it on-chip. At inference time, the host sends raw uint8
    pixels (0-255) and the NPU normalizes internally. This means the Hailo
    backend in classifier.py should NOT apply ImageNet normalization — just
    resize, center-crop, and send raw pixels.
"""

import argparse
import logging
import random
from pathlib import Path

import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ImageNet normalization scaled to 0-255 range for Hailo's on-device normalization.
# PyTorch uses mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] on [0,1] data.
# Hailo operates on [0,255] data, so multiply both by 255.
HAILO_MEAN = [123.675, 116.28, 103.53]
HAILO_STD = [58.395, 57.12, 57.375]


def load_calibration_images(
    calib_dir: Path,
    num_images: int = 1024,
    input_size: int = 260,
    seed: int = 42,
) -> np.ndarray:
    """
    Load calibration images as RAW pixels (0-255) in NHWC format.

    The Hailo DFC expects unnormalized images when a normalization model script
    is used (which bakes ImageNet normalization into the HEF). The preprocessing
    here matches the validation/inference transforms: Resize → CenterCrop.

    Args:
        calib_dir: Directory containing images (searches recursively for .jpg/.png).
        num_images: Number of images to use for calibration.
        input_size: Target crop size (260 for EfficientNet-B2, 224 for MobileNetV2).
        seed: Random seed for reproducible image selection.

    Returns:
        Numpy array of shape (num_images, input_size, input_size, 3), float32,
        values in 0-255 range (NHWC format, no normalization applied).
    """
    image_paths = sorted(
        list(calib_dir.rglob("*.jpg"))
        + list(calib_dir.rglob("*.jpeg"))
        + list(calib_dir.rglob("*.png"))
    )
    if len(image_paths) == 0:
        raise FileNotFoundError(f"No images found in {calib_dir}")

    random.seed(seed)
    if len(image_paths) > num_images:
        image_paths = random.sample(image_paths, num_images)
    logger.info(f"Loading {len(image_paths)} calibration images from {calib_dir}")

    # Resize shortest edge to input_size + 32 (matches PyTorch val transforms)
    resize_size = input_size + 32

    images = []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            # Resize shortest edge to resize_size
            w, h = img.size
            if w < h:
                new_w, new_h = resize_size, int(resize_size * h / w)
            else:
                new_w, new_h = int(resize_size * w / h), resize_size
            img = img.resize((new_w, new_h), Image.BILINEAR)
            # Center crop to input_size x input_size
            left = (new_w - input_size) // 2
            top = (new_h - input_size) // 2
            img = img.crop((left, top, left + input_size, top + input_size))
            # Raw pixels as float32, NO normalization (the model script handles it)
            arr = np.array(img, dtype=np.float32)  # (H, W, 3), values 0-255
            images.append(arr)
        except Exception as e:
            logger.warning(f"Skipping {path}: {e}")
            continue

    if len(images) == 0:
        raise RuntimeError("No valid images could be loaded for calibration")

    logger.info(f"Loaded {len(images)} calibration images ({input_size}x{input_size}, NHWC, 0-255)")
    return np.array(images, dtype=np.float32)


def convert_onnx_to_hef(
    onnx_path: Path,
    output_dir: Path,
    calib_dir: Path,
    hw_arch: str = "hailo10h",
    input_size: int = 260,
    num_calib_images: int = 1024,
) -> Path:
    """
    Full ONNX → HEF conversion pipeline for a classification model.

    Steps:
    1. Parse ONNX model into Hailo Archive (HAR) format
    2. Load model script with on-device ImageNet normalization
    3. Load calibration images (raw 0-255 pixels, NHWC)
    4. Quantize to INT8 using calibration data
    5. Compile to HEF for the target hardware

    Args:
        onnx_path: Path to the ONNX model.
        output_dir: Directory to save output files.
        calib_dir: Directory containing calibration images.
        hw_arch: Target Hailo hardware architecture.
            "hailo8"  = Hailo-8 (26 TOPS, AI HAT+ standard)
            "hailo8l" = Hailo-8L (13 TOPS, AI Kit)
            "hailo10h" = Hailo-10H (40 TOPS, AI HAT+ 2)
        input_size: Model input dimension (260 for EfficientNet-B2).
        num_calib_images: Number of calibration images for quantization.

    Returns:
        Path to the compiled HEF file.
    """
    from hailo_sdk_client import ClientRunner

    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = onnx_path.stem  # e.g., "efficientnet_b2_birds"

    # --- Stage 1: Parse ONNX → HAR ---
    logger.info(f"Stage 1: Parsing ONNX model ({onnx_path})")
    runner = ClientRunner(hw_arch=hw_arch)
    runner.translate_onnx_model(
        onnx_model_path=str(onnx_path),
        model_name=model_name,
        start_node_names=["input"],
        end_node_names=["output"],
        net_input_shapes={"input": [1, 3, input_size, input_size]},
    )

    har_path = output_dir / f"{model_name}_parsed.har"
    runner.save_har(str(har_path))
    logger.info(f"  Parsed HAR saved to {har_path}")

    # --- Stage 2: Load model script (on-device normalization) ---
    # This bakes ImageNet normalization into the HEF so the NPU handles it.
    # At inference time, the host just sends raw uint8 pixels — no CPU-side
    # normalization needed.
    model_script = (
        f"normalization1 = normalization({HAILO_MEAN}, {HAILO_STD})\n"
    )
    logger.info("Stage 2: Loading model script (on-device ImageNet normalization)")
    logger.info(f"  mean={HAILO_MEAN}, std={HAILO_STD}")
    runner.load_model_script(model_script)

    # --- Stage 3: Load calibration data ---
    logger.info(f"Stage 3: Loading calibration data from {calib_dir}")
    calib_data = load_calibration_images(
        calib_dir, num_images=num_calib_images, input_size=input_size,
    )

    # --- Stage 4: Quantize to INT8 ---
    logger.info("Stage 4: Quantizing to INT8 (this may take several minutes)...")
    runner.optimize(calib_data)

    quantized_har_path = output_dir / f"{model_name}_quantized.har"
    runner.save_har(str(quantized_har_path))
    logger.info(f"  Quantized HAR saved to {quantized_har_path}")

    # --- Stage 5: Compile to HEF ---
    logger.info(f"Stage 5: Compiling to HEF for {hw_arch}...")
    hef_data = runner.compile()

    hef_path = output_dir / f"{model_name}.hef"
    with open(hef_path, "wb") as f:
        f.write(hef_data)

    hef_size_mb = hef_path.stat().st_size / (1024 * 1024)
    logger.info(f"  HEF compiled: {hef_path} ({hef_size_mb:.1f} MB)")

    return hef_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert ONNX model to Hailo HEF format",
    )
    parser.add_argument(
        "--onnx", type=str, required=True,
        help="Path to the ONNX model file",
    )
    parser.add_argument(
        "--output-dir", type=str, default="models/hef",
        help="Directory to save output HEF and intermediate files",
    )
    parser.add_argument(
        "--calib-dir", type=str, default="data/nabirds/images",
        help="Directory containing calibration images",
    )
    parser.add_argument(
        "--hw-arch", type=str, default="hailo10h",
        choices=["hailo8", "hailo8l", "hailo10h", "hailo15h"],
        help="Target Hailo hardware architecture",
    )
    parser.add_argument(
        "--input-size", type=int, default=260,
        help="Model input size (260 for EfficientNet-B2, 224 for MobileNetV2)",
    )
    parser.add_argument(
        "--num-calib", type=int, default=1024,
        help="Number of calibration images for INT8 quantization",
    )
    args = parser.parse_args()

    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        logger.error(f"ONNX model not found: {onnx_path}")
        raise SystemExit(1)

    calib_dir = Path(args.calib_dir)
    if not calib_dir.exists():
        logger.error(f"Calibration directory not found: {calib_dir}")
        raise SystemExit(1)

    hef_path = convert_onnx_to_hef(
        onnx_path=onnx_path,
        output_dir=Path(args.output_dir),
        calib_dir=calib_dir,
        hw_arch=args.hw_arch,
        input_size=args.input_size,
        num_calib_images=args.num_calib,
    )

    print("\nConversion complete!")
    print(f"  ONNX:      {onnx_path}")
    print(f"  HEF:       {hef_path}")
    print(f"  Hardware:   {args.hw_arch}")
    print(f"  Input size: {args.input_size}x{args.input_size}")
    print(f"\nTo deploy on Raspberry Pi, copy {hef_path} to the Pi and run:")
    print("  python -m src.pipeline.pipeline --mode hailo")
