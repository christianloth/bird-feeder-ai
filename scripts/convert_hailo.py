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
    python scripts/convert_hailo.py --onnx models/onnx/mobilenetv2_birds.onnx

    # Specify target hardware
    python scripts/convert_hailo.py --onnx models/onnx/mobilenetv2_birds.onnx --hw-arch hailo10h

    # Custom calibration images
    python scripts/convert_hailo.py --onnx models/onnx/mobilenetv2_birds.onnx --calib-dir data/nabirds/images
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

# ImageNet normalization (must match training transforms)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
INPUT_SIZE = 224


def load_calibration_images(
    calib_dir: Path,
    num_images: int = 1024,
    seed: int = 42,
) -> np.ndarray:
    """
    Load and preprocess calibration images for INT8 quantization.

    The quantizer uses these images to collect activation statistics and
    determine optimal INT8 scaling factors. More images = better accuracy
    but slower compilation. 1024 is a good balance.

    Args:
        calib_dir: Directory containing images (searches recursively for .jpg/.png).
        num_images: Number of images to use for calibration.
        seed: Random seed for reproducible image selection.

    Returns:
        Numpy array of shape (num_images, 224, 224, 3), float32, normalized.
    """
    image_paths = sorted(
        list(calib_dir.rglob("*.jpg")) + list(calib_dir.rglob("*.png"))
    )
    if len(image_paths) == 0:
        raise FileNotFoundError(f"No images found in {calib_dir}")

    # Sample a random subset for calibration
    random.seed(seed)
    if len(image_paths) > num_images:
        image_paths = random.sample(image_paths, num_images)
    logger.info(f"Loading {len(image_paths)} calibration images from {calib_dir}")

    images = []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            # Match validation/inference transforms: Resize(256) → CenterCrop(224)
            # Resize shortest edge to 256
            w, h = img.size
            if w < h:
                new_w, new_h = 256, int(256 * h / w)
            else:
                new_w, new_h = int(256 * w / h), 256
            img = img.resize((new_w, new_h), Image.BILINEAR)
            # Center crop to 224x224
            left = (new_w - INPUT_SIZE) // 2
            top = (new_h - INPUT_SIZE) // 2
            img = img.crop((left, top, left + INPUT_SIZE, top + INPUT_SIZE))
            # Convert to float32, normalize with ImageNet stats
            arr = np.array(img, dtype=np.float32) / 255.0
            arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
            images.append(arr)
        except Exception as e:
            logger.warning(f"Skipping {path}: {e}")
            continue

    if len(images) == 0:
        raise RuntimeError("No valid images could be loaded for calibration")

    logger.info(f"Loaded {len(images)} calibration images")
    return np.array(images, dtype=np.float32)


def convert_onnx_to_hef(
    onnx_path: Path,
    output_dir: Path,
    calib_dir: Path,
    hw_arch: str = "hailo8",
    num_calib_images: int = 1024,
) -> Path:
    """
    Full ONNX → HEF conversion pipeline.

    Steps:
    1. Parse ONNX model into Hailo Archive (HAR) format
    2. Load calibration images from the dataset
    3. Quantize to INT8 using calibration data
    4. Compile to HEF for the target hardware

    Args:
        onnx_path: Path to the ONNX model.
        output_dir: Directory to save output files.
        calib_dir: Directory containing calibration images.
        hw_arch: Target Hailo hardware architecture.
            "hailo8"  = Hailo-8 (26 TOPS, AI HAT+ standard)
            "hailo8l" = Hailo-8L (13 TOPS, AI Kit)
            "hailo10h" = Hailo-10H (40 TOPS, AI HAT+ 2)
        num_calib_images: Number of calibration images for quantization.

    Returns:
        Path to the compiled HEF file.
    """
    from hailo_sdk_client import ClientRunner

    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = onnx_path.stem  # e.g., "mobilenetv2_birds"

    # --- Stage 1: Parse ONNX → HAR ---
    logger.info(f"Stage 1: Parsing ONNX model ({onnx_path})")
    runner = ClientRunner(hw_arch=hw_arch)
    runner.translate_onnx_model(
        onnx_model_path=str(onnx_path),
        start_node_names=["input"],
        end_node_names=["output"],
        net_input_shapes={"input": [1, 3, INPUT_SIZE, INPUT_SIZE]},
    )

    har_path = output_dir / f"{model_name}_parsed.har"
    runner.save_har(str(har_path))
    logger.info(f"  Parsed HAR saved to {har_path}")

    # --- Stage 2: Load calibration data ---
    logger.info(f"Stage 2: Loading calibration data from {calib_dir}")
    calib_data = load_calibration_images(calib_dir, num_images=num_calib_images)

    # --- Stage 3: Quantize to INT8 ---
    logger.info("Stage 3: Quantizing to INT8 (this may take several minutes)...")
    runner.optimize(calib_data)

    quantized_har_path = output_dir / f"{model_name}_quantized.har"
    runner.save_har(str(quantized_har_path))
    logger.info(f"  Quantized HAR saved to {quantized_har_path}")

    # --- Stage 4: Compile to HEF ---
    logger.info(f"Stage 4: Compiling to HEF for {hw_arch}...")
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
        num_calib_images=args.num_calib,
    )

    print("\nConversion complete!")
    print(f"  ONNX:     {onnx_path}")
    print(f"  HEF:      {hef_path}")
    print(f"  Hardware:  {args.hw_arch}")
    print(f"\nTo deploy on Raspberry Pi, copy {hef_path} to the Pi and update")
    print("config/settings.py classifier_model_path to point to it.")
