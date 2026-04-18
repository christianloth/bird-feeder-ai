"""
Local HEF compilation script — run on x86_64 Linux with NVIDIA GPU.

This is the local-machine equivalent of notebooks/convert_to_hef.ipynb.
Use this when you have a Linux workstation with a compatible NVIDIA GPU
and want to avoid Google Colab.

HARDWARE REQUIREMENTS:
  - x86_64 Linux (Ubuntu 20.04/22.04/24.04)
  - NVIDIA GPU with Pascal/Turing/Ampere architecture
    (GTX 1080 Ti, RTX 2080/3080, RTX A4000, etc.)
  - CUDA 11.8 or 12.5.1 (depends on DFC version)
  - cuDNN 8.9 or 9.10
  - 16+ GB RAM (32+ recommended)

SOFTWARE SETUP:
  1. Download Hailo DFC wheel from https://hailo.ai/developer-zone/
     (free registration required)

  2. Create a virtualenv and install it:

     python3 -m venv hailo_env
     source hailo_env/bin/activate
     pip install hailo_dataflow_compiler-*.whl
     pip install "numpy>=1.23,<2.0" "scipy>=1.10,<1.12" "Pillow>=9.0"
     pip install onnx-simplifier

  3. Verify GPU is detected:

     python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

     If the list is empty, the DFC will fall back to CPU-only (slower,
     and optimization_level auto-drops to 0 for models without a fixed
     .alls optimization_level — not a concern for ViT-Small which
     forces level 0 anyway).

USAGE:
  # Export ONNX on your Mac/dev machine first:
  python -m src.training.export_onnx classifier --model vit_small   # or vit_base / efficientnet_lite4

  # Copy to the Linux box:
  scp models/onnx/vit_small_birds.onnx user@linux-box:/path/to/project/

  # On the Linux box, run:
  python scripts/convert_hailo_local.py \\
      --model vit_small \\
      --onnx models/onnx/vit_small_birds.onnx \\
      --calib-dir data/nabirds/images \\
      --output-dir models/hef

WHY THIS SCRIPT EXISTS:
  The Colab notebook handles everything in one place (mount Drive, install
  DFC, run conversion). Locally, the install is a one-time step handled
  outside this script — so this is just the conversion logic.

  Pipeline is identical to the notebook:
    0. Pre-simplify ONNX (fixes ViT attention Mul parser bug)
    1. Parse ONNX -> HAR
    2. Load .alls model script (mixed-precision a16_w16 for ViT-Small/ViT-Base)
    3. Load calibration images
    4. Quantize to INT8
    5. Compile to HEF
"""

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import onnx
from onnxsim import simplify
from PIL import Image
from hailo_sdk_client import ClientRunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_CONFIGS = {
    "vit_small": {
        "input_size": 224,
        "hef_name": "vit_small_birds",
    },
    "vit_base": {
        "input_size": 224,
        "hef_name": "vit_base_birds",
    },
    "efficientnet_lite4": {
        "input_size": 300,
        "hef_name": "efficientnet_lite4_birds",
    },
}


def simplify_onnx(input_path: Path, output_path: Path) -> None:
    """Pre-simplify ONNX with onnx-simplifier.

    ViT models have fused attention operations (q = q * scale) that PyTorch
    exports as Mul nodes with a Sqrt-computed scale input. The Hailo DFC
    parser fails to translate these, reporting:
        "ew mult layer ew_mult1 expects 2 inputs but found 1"

    Running onnx-simplifier first folds the Sqrt into a constant initializer,
    which the DFC handles correctly. Matches Hailo Model Zoo's preprocessing.
    """
    logger.info(f"Simplifying ONNX: {input_path.name}")
    model = onnx.load(str(input_path))
    simplified, check = simplify(model)
    if not check:
        raise RuntimeError("onnx-simplifier validation failed")
    onnx.save(simplified, str(output_path))
    size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info(f"  Simplified ONNX saved: {output_path.name} ({size_mb:.1f} MB)")


def load_calibration_images(
    calib_dir: Path, num_images: int, input_size: int
) -> np.ndarray:
    """Load RAW unnormalized calibration images (NHWC, 0-255).

    Uses direct resize (not Resize+CenterCrop) to match production preprocessing.
    Production input is a tight YOLO crop — center-cropping would discard
    discriminative features at the edges.
    """
    image_paths = sorted(
        list(calib_dir.rglob("*.jpg"))
        + list(calib_dir.rglob("*.jpeg"))
        + list(calib_dir.rglob("*.png"))
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {calib_dir}")

    random.seed(42)
    if len(image_paths) > num_images:
        image_paths = random.sample(image_paths, num_images)
    logger.info(f"Loading {len(image_paths)} calibration images")

    images = []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            img = img.resize((input_size, input_size), Image.BILINEAR)
            images.append(np.array(img, dtype=np.float32))
        except Exception as e:
            logger.warning(f"Skipping {path}: {e}")

    logger.info(f"Loaded {len(images)} images ({input_size}x{input_size}, NHWC, 0-255)")
    return np.array(images, dtype=np.float32)


def get_model_script_vit_small(hef_name: str) -> str:
    """ViT-Small model script — copied from Hailo Model Zoo vit_small.alls.

    Critical: ew_add layers need 16-bit precision for self-attention to
    work. ew_add1 stays at 8-bit. Normalization matches timm's mean=0.5
    std=0.5 (= [127.5]/[127.5] on 0-255 scale).

    The layer prefix must match the model_name passed to translate_onnx_model.
    """
    return f"""norm_layer1 = normalization([127.5, 127.5, 127.5], [127.5, 127.5, 127.5])

context_switch_param(mode=enabled)
allocator_param(enable_partial_row_buffers=disabled)
allocator_param(automatic_reshapes=disabled)
buffers(conv1_s2d, conv1, 0, PARTIAL_ROW)
resources_param(strategy=greedy, max_compute_utilization=0.8, max_control_utilization=1.0, max_memory_utilization=0.8)

model_optimization_config(calibration, batch_size=16, calibset_size=1024)
post_quantization_optimization(finetune, policy=disabled)
pre_quantization_optimization(equalization, policy=enabled)
pre_quantization_optimization(ew_add_fusing, policy=disabled)
model_optimization_flavor(optimization_level=0, compression_level=0)

pre_quantization_optimization(matmul_correction, layers={{matmul*}}, correction_type=zp_comp_block)
model_optimization_config(negative_exponent, layers={{*}}, rank=0)

quantization_param({{{hef_name}/ew_add*}}, precision_mode=a16_w16)
quantization_param({{{hef_name}/ew_add1}}, precision_mode=a8_w8)
"""


def get_model_script_vit_base(hef_name: str) -> str:
    """ViT-Base model script — copied from Hailo Model Zoo vit_base.alls.

    Same mixed-precision strategy as ViT-Small: ew_add layers need 16-bit
    precision for self-attention accuracy; ew_add1 stays at 8-bit.
    Normalization matches timm's mean=0.5 std=0.5 (= [127.5]/[127.5]).

    Differences from vit_small.alls:
      - calibration batch_size=8 (ViT-Base is ~4x larger, 86.5M params)
      - No context_switch_param / allocator_param / buffers tuning
        (the Model Zoo .alls for vit_base omits these — the compiler picks
        defaults that work for this model)
      - max_control_utilization=0.85 (vs 1.0 for vit_small)
      - No post_quantization_optimization(finetune, ...) line

    The layer prefix (e.g., vit_base_birds/ew_add*) must match the model_name
    passed to translate_onnx_model.
    """
    return f"""norm_layer1 = normalization([127.5, 127.5, 127.5], [127.5, 127.5, 127.5])
model_optimization_config(calibration, batch_size=8, calibset_size=1024)
pre_quantization_optimization(equalization, policy=enabled)
pre_quantization_optimization(ew_add_fusing, policy=disabled)
model_optimization_flavor(optimization_level=0, compression_level=0)
pre_quantization_optimization(matmul_correction, layers={{matmul*}}, correction_type=zp_comp_block)
model_optimization_config(negative_exponent, layers={{*}}, rank=0)
quantization_param({{{hef_name}/ew_add*}}, precision_mode=a16_w16)
quantization_param({{{hef_name}/ew_add1}}, precision_mode=a8_w8)

resources_param(strategy=greedy, max_compute_utilization=0.8, max_control_utilization=0.85, max_memory_utilization=0.8)
"""


def get_model_script_efficientnet_lite4() -> str:
    """EfficientNet-Lite4 model script — simple normalization, standard INT8."""
    return "normalization1 = normalization([127.5, 127.5, 127.5], [127.5, 127.5, 127.5])\n"


def convert(
    model: str,
    onnx_path: Path,
    calib_dir: Path,
    output_dir: Path,
    num_calib_images: int = 1024,
) -> Path:
    """Full ONNX -> HEF conversion pipeline.

    Args:
        model: Either "vit_small" or "efficientnet_lite4".
        onnx_path: Path to the exported ONNX model.
        calib_dir: Directory of calibration images (NABirds dataset).
        output_dir: Where to write intermediate HARs and the final HEF.
        num_calib_images: Number of calibration images for quantization.

    Returns:
        Path to the compiled HEF file.
    """
    if model not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model}. Choose from: {list(MODEL_CONFIGS.keys())}")

    cfg = MODEL_CONFIGS[model]
    input_size = cfg["input_size"]
    hef_name = cfg["hef_name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Converting {model} ({onnx_path.name}) for hailo10h")
    logger.info(f"Input size: {input_size}x{input_size}, Output: {hef_name}.hef")

    # Stage 0: Pre-simplify ONNX
    simplified_onnx = output_dir / f"{hef_name}_simplified.onnx"
    simplify_onnx(onnx_path, simplified_onnx)

    # Stage 1: Parse
    logger.info("Stage 1: Parsing ONNX model")
    runner = ClientRunner(hw_arch="hailo10h")
    runner.translate_onnx_model(
        str(simplified_onnx),
        hef_name,
        start_node_names=["input"],
        end_node_names=["output"],
        net_input_shapes={"input": [1, 3, input_size, input_size]},
    )
    runner.save_har(str(output_dir / f"{hef_name}_parsed.har"))
    logger.info("  Parsed HAR saved")

    # Stage 2: Load model script
    if model == "vit_small":
        logger.info("Stage 2: Loading ViT-Small .alls (mixed-precision a16_w16 for attention)")
        model_script = get_model_script_vit_small(hef_name)
    elif model == "vit_base":
        logger.info("Stage 2: Loading ViT-Base .alls (mixed-precision a16_w16 for attention)")
        model_script = get_model_script_vit_base(hef_name)
    else:
        logger.info("Stage 2: Loading EfficientNet-Lite4 .alls (standard INT8)")
        model_script = get_model_script_efficientnet_lite4()
    runner.load_model_script(model_script)

    # Stage 3: Load calibration data
    logger.info("Stage 3: Loading calibration images")
    calib_data = load_calibration_images(calib_dir, num_calib_images, input_size)

    # Stage 4: Quantize
    logger.info("Stage 4: Quantizing to INT8...")
    runner.optimize(calib_data)
    runner.save_har(str(output_dir / f"{hef_name}_quantized.har"))
    logger.info("  Quantized HAR saved")

    # Stage 5: Compile
    logger.info("Stage 5: Compiling to HEF for hailo10h...")
    hef_data = runner.compile()
    hef_path = output_dir / f"{hef_name}.hef"
    hef_path.write_bytes(hef_data)

    size_mb = hef_path.stat().st_size / 1024 / 1024
    logger.info(f"Done! HEF saved: {hef_path} ({size_mb:.1f} MB)")
    return hef_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert ONNX classifier to Hailo HEF (local Linux + GPU)"
    )
    parser.add_argument(
        "--model", required=True, choices=list(MODEL_CONFIGS.keys()),
        help="Model architecture",
    )
    parser.add_argument(
        "--onnx", type=Path, required=True,
        help="Path to exported ONNX file",
    )
    parser.add_argument(
        "--calib-dir", type=Path, required=True,
        help="Directory of calibration images (NABirds)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("models/hef"),
        help="Where to save HAR and HEF files (default: models/hef)",
    )
    parser.add_argument(
        "--num-calib", type=int, default=1024,
        help="Number of calibration images (default: 1024)",
    )
    args = parser.parse_args()

    if not args.onnx.exists():
        raise SystemExit(f"ONNX not found: {args.onnx}")
    if not args.calib_dir.exists():
        raise SystemExit(f"Calibration dir not found: {args.calib_dir}")

    convert(
        model=args.model,
        onnx_path=args.onnx,
        calib_dir=args.calib_dir,
        output_dir=args.output_dir,
        num_calib_images=args.num_calib,
    )
