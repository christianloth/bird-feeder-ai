# Bird Feeder AI

24/7 bird species detection and tracking system using a Raspberry Pi 5, Hailo AI HAT+ NPU, and an SV3C 4K PTZ camera pointed at a bird feeder in Frisco, TX.

The system identifies birds in real time using a two-stage AI pipeline: YOLOv8n detects birds in the camera feed, then a fine-tuned MobileNetV2 classifies each bird to one of 555 species from the NABirds dataset. All inference runs on-device via the Hailo NPU at 26 TOPS, with no cloud dependency. A FastAPI backend with SQLite stores every detection for review and analysis.

## Architecture

```
Camera (RTSP) --> YOLOv8n (detection) --> Tracker --> MobileNetV2 (classification) --> Database
                  "Is there a bird?"      Dedup       "What species?"                  SQLite
                  COCO class 14           centroid     555 NABirds classes              + image crops
```

**Two-stage pipeline:**

1. **Detection** -- YOLOv8n identifies birds (COCO class 14) in each frame and outputs bounding boxes.
2. **Tracking** -- A centroid-based tracker deduplicates detections so a bird sitting on the feeder for 30 seconds is logged once, not 50 times.
3. **Classification** -- Cropped bird regions are classified by a MobileNetV2 fine-tuned on NABirds (555 North American species).
4. **Storage** -- Detections are saved to SQLite with species, confidence, bounding box, cropped image, and thumbnail. Weather data from Open-Meteo is correlated for activity analysis.

**Inference backends** (both detector and classifier support pluggable backends):

| Backend | Use case | Device |
|---|---|---|
| Ultralytics / PyTorch | Development and testing | Mac (MPS), NVIDIA GPU (CUDA), CPU |
| ONNX Runtime | Cross-platform inference | Any platform |
| Hailo HEF | Production deployment | Raspberry Pi 5 + AI HAT+ |

## Hardware

| Component | Model |
|---|---|
| Compute | Raspberry Pi 5 (8 GB) |
| AI Accelerator | Hailo AI HAT+ (26 TOPS NPU) |
| Camera | SV3C 4K PTZ (RTSP stream) |
| Location | Frisco, TX (33.15N, -96.82W) |

## Project Structure

```
bird-feeder-ai/
├── config/
│   └── settings.py            # All configuration (paths, thresholds, camera, training)
├── src/
│   ├── training/              # PyTorch training pipeline
│   │   ├── dataset.py         # NABirds dataset loader
│   │   ├── transforms.py      # Train/val/inference augmentations
│   │   ├── model.py           # MobileNetV2 with custom classifier head
│   │   ├── train.py           # Training loop (two-phase strategy)
│   │   ├── evaluate.py        # Metrics, confusion matrix, plotting
│   │   └── export_onnx.py     # Export trained model to ONNX
│   ├── inference/
│   │   ├── classifier.py      # Species classifier (PyTorch / ONNX / Hailo backends)
│   │   └── tracker.py         # Centroid-based bird tracker + ROI cropping
│   ├── pipeline/
│   │   ├── main.py            # Main pipeline orchestrator and CLI entry point
│   │   ├── detector.py        # YOLOv8n bird detector (Ultralytics / Hailo backends)
│   │   └── camera.py          # RTSP camera capture with background thread
│   ├── backend/
│   │   ├── api.py             # FastAPI REST API
│   │   ├── database.py        # SQLAlchemy 2.0 models (detections, species, weather)
│   │   ├── schemas.py         # Pydantic request/response schemas
│   │   ├── storage.py         # Image crop and thumbnail storage with retention policy
│   │   └── weather.py         # Open-Meteo weather integration
│   └── audio/                 # BirdNET audio classification (planned)
├── models/
│   ├── checkpoints/           # Trained PyTorch model weights (.pth)
│   ├── onnx/                  # Exported ONNX models
│   └── hef/                   # Compiled Hailo HEF models
├── data/
│   └── nabirds/               # NABirds dataset (downloaded separately)
├── detections/                # Saved bird crop images (organized by date)
├── notebooks/                 # Jupyter notebooks for exploration
├── scripts/                   # Utility scripts
├── tests/                     # Test suite
├── pyproject.toml             # Dependencies and project metadata
└── CLAUDE.md                  # AI assistant project context
```

## Setup

**Requirements:** Python 3.10+ (developed on 3.13)

### Install with uv (recommended)

```bash
uv sync
```

### Install with pip

```bash
pip install -e .
```

### Install dev dependencies

```bash
# uv
uv sync --extra dev

# pip
pip install -e ".[dev]"
```

### Hailo runtime (Raspberry Pi only)

```bash
sudo apt install hailo-all
```

## Training

The training pipeline fine-tunes a pretrained MobileNetV2 on the NABirds dataset to classify 555 North American bird species.

### 1. Download the NABirds dataset

Download NABirds from [https://dl.allaboutbirds.org/nabirds](https://dl.allaboutbirds.org/nabirds) and extract it to `data/nabirds/`. The directory should contain `images/`, `classes.txt`, `train_test_split.txt`, and related metadata files.

### 2. Run training

```bash
python -m src.training.train
```

This executes a **two-phase training strategy**:

**Phase 1 -- Train the classifier head (10 epochs, lr=0.001)**
The MobileNetV2 backbone is frozen (pretrained ImageNet weights). Only the new classifier head learns to map visual features to 555 bird species. This is fast because only the final layer has trainable parameters.

**Phase 2 -- Fine-tune late backbone layers (15 epochs, lr=0.0001)**
Backbone layers 14+ are unfrozen and trained with a 10x lower learning rate. This lets the late layers shift from detecting generic object parts to detecting bird-specific features (beak shapes, wing bars, plumage patterns) without destroying the pretrained weights.

The best model (by validation accuracy) is saved to `models/checkpoints/best_model.pth`. A full checkpoint with optimizer state (for resuming training) is saved to `models/checkpoints/checkpoint.pth`. A training history plot is saved to `models/checkpoints/training_history.png`.

### 3. Export to ONNX

After training, export the model to ONNX format for cross-platform inference or conversion to Hailo HEF:

```bash
python -m src.training.export_onnx
```

This produces `models/onnx/mobilenetv2_birds.onnx`. The ONNX model is the intermediate step in the deployment chain:

```
PyTorch (.pth) --> ONNX (.onnx) --> Hailo DFC compiler --> HEF (.hef)
```

## Running the Pipeline

### CLI usage

```bash
python -m src.pipeline.pipeline [OPTIONS]
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--mode {dev,hailo}` | `dev` | Deployment mode: `dev` for Mac/PC, `hailo` for Raspberry Pi |
| `--image PATH` | None | Process a single image instead of live camera |
| `--checkpoint PATH` | `models/checkpoints/best_model.pth` | Path to trained model checkpoint (dev mode) |
| `--rtsp-url URL` | From config | RTSP camera URL (overrides `config/settings.py`) |
| `--device {mps,cuda,cpu}` | Auto-detected | Force compute device (dev mode only) |

### Development mode (Mac / PC)

Test on a single image:

```bash
python -m src.pipeline.pipeline --mode dev --image path/to/bird.jpg
```

Run with a live camera:

```bash
python -m src.pipeline.pipeline --mode dev --rtsp-url rtsp://user:pass@192.168.1.100:554/stream1
```

Force a specific device:

```bash
python -m src.pipeline.pipeline --mode dev --image bird.jpg --device cpu
```

Development mode uses Ultralytics YOLO for detection and PyTorch for classification. It auto-selects MPS (Apple Silicon), CUDA (NVIDIA), or CPU.

### Hailo production mode (Raspberry Pi)

```bash
python -m src.pipeline.pipeline --mode hailo --rtsp-url rtsp://user:pass@192.168.1.100:554/stream1
```

Production mode uses pre-compiled HEF models on the Hailo NPU for both detection and classification. Model paths default to `models/hef/yolov8n.hef` and `models/hef/mobilenetv2_birds.hef` (configurable in `config/settings.py`).

## API

The FastAPI backend provides a REST API for querying detections, species stats, and system status.

### Start the API server

```bash
uvicorn src.backend.api:app --host 0.0.0.0 --port 8000
```

Interactive API docs are available at `http://localhost:8000/docs`.

### Key endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/api/detections` | List detections (filterable by species, date range, confidence) |
| `GET` | `/api/detections/{id}` | Get a single detection |
| `PATCH` | `/api/detections/{id}/review` | Mark detection as correct or false positive |
| `GET` | `/api/stats` | Overall detection statistics |
| `GET` | `/api/stats/species` | Detection count per species |
| `GET` | `/api/stats/hourly` | Hourly detection counts for a given day |
| `GET` | `/api/species` | List all known species |
| `GET` | `/api/species/{id}` | Get species details |
| `GET` | `/api/weather/current` | Current weather at the feeder location |
| `GET` | `/api/daily` | Daily summary data (last N days) |
| `GET` | `/api/system/status` | System health (disk usage, uptime, detection count) |
| `POST` | `/api/system/cleanup` | Trigger old image cleanup |

### Query parameters for `/api/detections`

- `skip` / `limit` -- Pagination (limit max 500)
- `species_id` -- Filter by species
- `since` / `until` -- Date range filter
- `min_confidence` -- Minimum confidence threshold

## Configuration

All settings are managed in `config/settings.py` using Pydantic Settings. Every setting can be overridden with an environment variable using the `BIRD_` prefix.

| Setting | Env var | Default | Description |
|---|---|---|---|
| `rtsp_url` | `BIRD_RTSP_URL` | `rtsp://admin:password@...` | Camera RTSP stream URL |
| `detection_confidence_threshold` | `BIRD_DETECTION_CONFIDENCE_THRESHOLD` | `0.4` | Minimum YOLO confidence |
| `classification_confidence_threshold` | `BIRD_CLASSIFICATION_CONFIDENCE_THRESHOLD` | `0.5` | Minimum species confidence |
| `batch_size` | `BIRD_BATCH_SIZE` | `32` | Training batch size |
| `learning_rate` | `BIRD_LEARNING_RATE` | `0.001` | Training learning rate |
| `num_epochs` | `BIRD_NUM_EPOCHS` | `25` | Training epochs |
| `save_crops` | `BIRD_SAVE_CROPS` | `True` | Save detection crop images |
| `retention_days` | `BIRD_RETENTION_DAYS` | `90` | Days to keep crop images |
| `latitude` / `longitude` | `BIRD_LATITUDE` / `BIRD_LONGITUDE` | Frisco, TX | Weather location |
| `database_url` | `BIRD_DATABASE_URL` | `sqlite:///data/birds.db` | SQLite database path |

## License

Private project.
