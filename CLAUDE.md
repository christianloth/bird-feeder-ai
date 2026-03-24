# Bird Feeder AI

## Project
24/7 bird species detection and tracking system. Raspberry Pi 5 + AI HAT+ 2 (Hailo NPU) + SV3C 4K PTZ camera at a bird feeder in Frisco, TX.

## Architecture
Two-stage pipeline: YOLOv8n (detection, COCO class 14) → MobileNetV2 (species classification, fine-tuned on NABirds). Both run on Hailo NPU. FastAPI backend with SQLite. BirdNET for optional audio classification.

## Important
- The `src/training/` files (dataset.py, transforms.py, model.py, train.py, evaluate.py) are designed for the user to implement themselves as a PyTorch learning exercise. Each file has detailed TODO comments. Do NOT fill in the implementations unless explicitly asked.
- export_onnx.py is already implemented (mechanical, not a learning opportunity).
- The `src/inference/`, `src/pipeline/`, and `src/backend/` directories will be implemented by Claude (infrastructure code).

## Conventions
- Python 3.10+
- Use pathlib.Path for file paths
- Type hints on all function signatures
- ruff for linting (line-length=100)
