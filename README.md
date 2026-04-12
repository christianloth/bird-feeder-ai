# Bird Feeder AI

24/7 bird and wildlife detection system using a Raspberry Pi 5, Hailo AI HAT+ 2 (Hailo-10H NPU), and an SV3C 4K PTZ camera at a bird feeder in Frisco, TX.

During the day, a two-stage AI pipeline detects and classifies birds to one of 555 species. At night, the system automatically switches to a wildlife detector (11 classes) trained on camera trap data. All inference runs on-device with no cloud dependency.

## Quick Start

### 1. Install

```bash
uv sync
```

### 2. Configure

```bash
cp config/config.yaml.example config/config.yaml
```

Edit `config/config.yaml` with your camera RTSP URL, location, and preferences.

### 3. Run the pipeline

```bash
python -m src.pipeline.pipeline --mode dev
```

That's it. The pipeline connects to your camera, detects birds (day) or wildlife (night), classifies species, and saves everything to the database and disk.

## Architecture

```
                          DAYTIME                                    NIGHTTIME
Camera (RTSP) --> YOLO11n (COCO bird) --> Tracker --> EfficientNet-B2 --> DB
  or video file   "Is there a bird?"       Dedup      "What species?"     SQLite
                                                       555 NABirds        + images

Camera (RTSP) --> YOLO11s (wildlife) ---> Tracker ----------------------> DB
  or video file   "What animal?"           Dedup    class from detector   SQLite
                  11 classes                                              + images
```

The system checks sunrise/sunset times (Open-Meteo API) every 60 seconds. Night mode activates 30 minutes after sunset and deactivates 30 minutes before sunrise. Both offsets are configurable.

**Daytime pipeline (bird species):**
1. **Detection** -- YOLO11n finds birds (COCO class 14) and outputs bounding boxes
2. **Tracking** -- Centroid-based tracker deduplicates so a bird sitting for 30 seconds is logged once
3. **Classification** -- Cropped bird region is classified by EfficientNet-B2 (555 NABirds species)
4. **Storage** -- Detection saved to SQLite with species, confidence, bbox coordinates, and one full frame image (crops generated on-the-fly)

**Nighttime pipeline (wildlife):**
1. **Detection** -- YOLO11n wildlife model detects 11 classes: bird, bobcat, coyote, raccoon, rabbit, skunk, opossum, squirrel, armadillo, cat, dog
2. **Tracking** -- Same tracker, reset on mode switch
3. **Storage** -- Same storage path, species comes directly from the YOLO model (no separate classifier)

**Inference backends:**

| Backend | Use case | Device |
|---|---|---|
| Ultralytics / PyTorch | Development and testing | Mac (MPS), NVIDIA GPU (CUDA), CPU |
| ONNX Runtime | Cross-platform inference | Any platform |
| Hailo HEF | Production deployment | Raspberry Pi 5 + AI HAT+ |

## Hardware

| Component | Model |
|---|---|
| Compute | Raspberry Pi 5 (8 GB) |
| AI Accelerator | Hailo AI HAT+ 2 (Hailo-10H, 40 TOPS INT4 / 20 TOPS INT8) |
| Camera | SV3C 4K PTZ (RTSP stream) |
| Location | Frisco, TX (33.15N, -96.82W) |

## Setup

**Requirements:** Python 3.13

### Install with uv (recommended)

**Mac / Linux (development):**

```bash
uv sync

# Training / export (adds Ultralytics, ONNX, scikit-learn, matplotlib, etc.)
uv sync --extra training
```

**Raspberry Pi (Hailo inference):**

The Pi needs access to `hailo_platform`, which is installed as a system package via apt. Create the venv using the system Python so it can see apt-installed packages:

```bash
sudo apt install hailo-all
uv venv --python /usr/bin/python --system-site-packages .venv
uv sync
```

### Install with pip

```bash
pip install -e .

# Training / export
pip install -e ".[training]"
```

### Install dev dependencies

```bash
# uv
uv sync --extra dev

# pip
pip install -e ".[dev]"
```

## Running the Pipeline

### Start live detection

```bash
python -m src.pipeline.pipeline --mode dev
```

The RTSP URL comes from `config/config.yaml`. The pipeline runs continuously until you press Ctrl+C.

### CLI options

| Flag | Default | Description |
|---|---|---|
| `--mode {dev,hailo}` | `dev` | `dev` for Mac/PC, `hailo` for Raspberry Pi |
| `--image PATH` | None | Process a single image instead of live camera |
| `--video PATH` | None | Process a video file instead of live camera |
| `--process-every-n N` | From config | Process every Nth frame in video mode |
| `--checkpoint PATH` | `models/bird-classifier/efficientnet_b2/best_model.pth` | Path to classifier checkpoint |
| `--rtsp-url URL` | From config | RTSP camera URL (overrides config) |
| `--device {cuda,mps,cpu}` | Auto (CUDA → MPS → CPU) | Force compute device |
| `--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}` | From config | Logging verbosity |
| `--no-night` | Off | Disable night mode (daytime bird detection only) |
| `--no-save` | Off | Disable saving to database and disk (log only) |

### Examples

Test on a single image:

```bash
python -m src.pipeline.pipeline --mode dev --image path/to/bird.jpg
```

Run inference on a video file (annotated video output only, no database):

```bash
python -m src.pipeline.pipeline --video path/to/video.mp4 --output
```

Virtual RTSP — test the full RTSP pipeline (saves to database + `detections/rtsp/`) using a video file:

```bash
python -m src.pipeline.pipeline --video clip.mp4 --virtual-rtsp --day --output
```

Process every 12th frame of a video (faster, skips redundant frames):

```bash
python -m src.pipeline.pipeline --video clip.mp4 --process-every-n 12
```

Run with verbose logging (see per-frame inference timing, tracker details):

```bash
python -m src.pipeline.pipeline --mode dev --log-level DEBUG
```

Disable night mode (bird detection only, no wildlife switching):

```bash
python -m src.pipeline.pipeline --mode dev --no-night
```

Force CPU for testing:

```bash
python -m src.pipeline.pipeline --mode dev --image bird.jpg --device cpu
```

### Hailo production mode (Raspberry Pi)

```bash
python -m src.pipeline.pipeline --mode hailo
```

Uses pre-compiled HEF models on the Hailo NPU. Model paths are configurable in `config/config.yaml`.

### What gets saved per detection

Each detection saves one full frame image and a database row:

| Saved | Where | Purpose |
|---|---|---|
| Full frame | `detections/YYYY/MM/DD/{name}.jpg` | Source of truth image |
| Bbox coordinates | SQLite `detections` table | `bbox_x1, y1, x2, y2` pixel values |
| Metadata | SQLite `detections` table | Species, confidence, model, timestamp, source |

Crops, thumbnails, and annotated views are generated on-the-fly from the frame + bbox coordinates (no redundant image files).

### Logging levels

| Level | What you see |
|---|---|
| `DEBUG` | Per-frame inference timing, tracker lifecycle, classification scores, mode check results |
| `INFO` | Startup config, detections, mode transitions, 5-minute status summaries |
| `WARNING` | Slow inference (>500ms detect, >200ms classify), empty crops, high track counts, reconnection attempts |
| `ERROR` | Frame processing failures, database save failures, API failures |
| `CRITICAL` | Model file not found, camera unreachable after max retries |

## API Server

The FastAPI backend provides a REST API for querying detections, reviewing them, and correcting misclassifications.

### Start the API server

```bash
uvicorn src.backend.api:app --host 0.0.0.0 --port 8000
```

Interactive API docs: `http://localhost:8000/docs`

### Review detections

Open `http://localhost:8000/review` to review detections. Keyboard shortcuts: **A** = confirm, **X** = false positive, **C** = correct species, **Esc** = cancel.

### Key endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/review` | Detection review UI (HTMX) |
| `GET` | `/api/detections` | List detections (filterable by species, date, confidence) |
| `GET` | `/api/detections/{id}` | Get a single detection with all metadata |
| `GET` | `/api/detections/{id}/crop` | Cropped detection image (generated on-the-fly) |
| `GET` | `/api/detections/{id}/frame` | Full frame image |
| `PATCH` | `/api/detections/{id}/review` | Review: confirm, reject, or correct species |
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

### Reviewing detections

Mark a detection as correct:

```bash
curl -X PATCH http://localhost:8000/api/detections/1/review \
  -H "Content-Type: application/json" \
  -d '{"is_false_positive": false}'
```

Mark as false positive (not a real bird/animal):

```bash
curl -X PATCH http://localhost:8000/api/detections/1/review \
  -H "Content-Type: application/json" \
  -d '{"is_false_positive": true}'
```

Correct a misclassification (e.g., classifier said Common Ground-Dove but it's actually a Mourning Dove with species ID 42):

```bash
curl -X PATCH http://localhost:8000/api/detections/1/review \
  -H "Content-Type: application/json" \
  -d '{"is_false_positive": false, "corrected_species_id": 42}'
```

The corrected species label is used when exporting training data, so your corrections feed back into the next model.

## Exporting Training Data

After accumulating detections, export them as labeled training data for retraining your models. The export script queries the database, deduplicates near-identical images using perceptual hashing, and outputs data in the format your training pipeline expects.

### Export commands

```bash
# Export everything (both classifier + YOLO formats)
python -m scripts.export_training_data --format both --clean

# Only high-confidence detections
python -m scripts.export_training_data --format both --min-confidence 0.7

# Only manually reviewed detections
python -m scripts.export_training_data --format both --reviewed-only

# Classification format only (for retraining EfficientNet)
python -m scripts.export_training_data --format classification --min-confidence 0.5

# YOLO format only (for retraining the detector)
python -m scripts.export_training_data --format yolo

# Filter by date
python -m scripts.export_training_data --format both --since 2026-04-01
```

### Export options

| Flag | Default | Description |
|---|---|---|
| `--format {classification,yolo,both}` | `both` | Export format |
| `--min-confidence` | `0.0` | Minimum confidence threshold |
| `--reviewed-only` | Off | Only export reviewed detections |
| `--include-false-positives` | Off | Include detections marked as false positives |
| `--since DATE` | None | Only export detections after this date (ISO format) |
| `--dedup-threshold` | `5` | Perceptual hash Hamming distance (lower = stricter dedup) |
| `--val-split` | `0.2` | Fraction of data for validation (YOLO export) |
| `--clean` | Off | Delete existing export directory before exporting |

## Training

The training pipeline fine-tunes a pretrained model on the NABirds dataset to classify 555 North American bird species.

| Model | Input Size | Params | Training Strategy | LR Scheduler |
|---|---|---|---|---|
| **EfficientNet-B2** (default) | 260x260 | 7.8M | Single-phase end-to-end | ReduceLROnPlateau |
| **MobileNetV2** | 224x224 | 3.4M | Two-phase (freeze then unfreeze) | StepLR |

### 1. Download the NABirds dataset

Download NABirds from [https://dl.allaboutbirds.org/nabirds](https://dl.allaboutbirds.org/nabirds) and extract it to `data/nabirds/`. The directory should contain `images/`, `classes.txt`, `train_test_split.txt`, and related metadata files.

### 2. Run training

```bash
# EfficientNet-B2 (default)
python -m src.training.train

# MobileNetV2
python -m src.training.train --model mobilenetv2
```

#### Training CLI options

| Flag | Default | Description |
|---|---|---|
| `--model {efficientnet_b2,mobilenetv2}` | `efficientnet_b2` | Model architecture |
| `--batch-size` | `32` | Batch size |
| `--num-workers` | `4` | Data loading workers |
| `--amp` | Off | Enable mixed precision (CUDA only) |
| `--resume` | Off | Resume from latest run's checkpoint |
| `--preprocessed PATH` | None | Use preprocessed dataset (MobileNetV2 only) |

#### Training strategies

**EfficientNet-B2** trains end-to-end in a single phase with all layers unfrozen. ReduceLROnPlateau only drops the learning rate when validation accuracy stalls. Early stopping (patience=5) halts training if no improvement for 5 consecutive epochs.

**MobileNetV2** uses a two-phase strategy:
- **Phase 1** (10 epochs, lr=0.001) -- Backbone frozen, only the classifier head trains.
- **Phase 2** (25 epochs, lr=0.0001) -- Backbone layers 14+ unfrozen, fine-tuned with lower LR.

#### Training output

Each run creates a timestamped folder:

```
models/bird-classifier/efficientnet_b2/
  2026-03-27_04-12/
    best_model.pth          # Best weights (by val accuracy)
    checkpoint.pth          # Full state for --resume
    training_history.png    # Loss and accuracy curves
    confusion_matrices/     # Per-epoch confusion matrix images
```

#### Per-epoch metrics

Each epoch prints detailed validation metrics:
- Total correct/wrong, precision, recall, F1, top-5 accuracy
- Worst 5 species by F1 with TP/FP/FN breakdown
- Top 5 most confused species pairs
- Confusion matrix saved as image

#### Resume interrupted training

```bash
python -m src.training.train --resume
```

Automatically finds the latest run's checkpoint and continues from where it left off (preserving optimizer and scheduler state). Checkpoints are saved atomically after every epoch.

### 3. Export to ONNX

After training, export models to ONNX format for cross-platform inference or conversion to Hailo HEF. The export script handles both classification and YOLO models:

```bash
# Export EfficientNet-B2 classifier (auto-finds latest checkpoint)
python -m src.training.export_onnx classifier

# Export MobileNetV2 classifier
python -m src.training.export_onnx classifier --model mobilenetv2

# Export a specific checkpoint
python -m src.training.export_onnx classifier --checkpoint path/to/best_model.pth

# Export YOLO wildlife detector
python -m src.training.export_onnx yolo --weights models/wildlife/yolo11s-wildlife-equal/weights/best.pt

# Export any YOLO variant
python -m src.training.export_onnx yolo --weights models/wildlife/yolo11n-wildlife-equal/weights/best.pt
```

#### Export CLI options

**Classifier subcommand:**

| Flag | Default | Description |
|---|---|---|
| `--model {efficientnet_b2,mobilenetv2}` | `efficientnet_b2` | Model architecture |
| `--checkpoint PATH` | Auto (latest run) | Path to `.pth` checkpoint |
| `--output PATH` | `models/onnx/<model>_birds.onnx` | Output path |

**YOLO subcommand:**

| Flag | Default | Description |
|---|---|---|
| `--weights PATH` | *(required)* | Path to Ultralytics `.pt` file |
| `--imgsz N` | `640` | Input image size |
| `--output PATH` | `models/onnx/<model_name>.onnx` | Output path |

The ONNX model is the intermediate step in the deployment chain:

```
PyTorch (.pth) --> ONNX (.onnx) --> Hailo DFC compiler --> HEF (.hef)
```

### 4. Convert to Hailo HEF

The ONNX-to-HEF conversion requires the Hailo DFC compiler, which only runs on x86_64 Linux. Use the provided Google Colab notebook for compilation from any machine:

1. Download the Hailo DFC wheel from [Hailo Developer Zone](https://hailo.ai/developer-zone/software-downloads/) (free account)
2. Open `scripts/convert_to_hef.ipynb` in Google Colab
3. Upload the DFC wheel, your ONNX files, and calibration images
4. Run all cells -- downloads the compiled HEF files when done

Pre-compiled HEFs for standard models (e.g., YOLOv11n for bird detection) are available from the [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo) and don't need conversion.

## Configuration

All settings are in `config/config.yaml`. Copy the example to get started:

```bash
cp config/config.yaml.example config/config.yaml
```

This file is gitignored (contains camera credentials). Key settings:

| Section | Setting | Default | Description |
|---|---|---|---|
| Top-level | `log_level` | `INFO` | Logging verbosity |
| `camera` | `rtsp_url` | -- | Camera RTSP stream URL |
| `detection` | `confidence_threshold` | `0.4` | Minimum YOLO detection confidence |
| `classification` | `confidence_threshold` | `0.10` | Minimum species classification confidence |
| `wildlife` | `confidence_threshold` | `0.4` | Minimum wildlife detection confidence |
| `day_night` | `enabled` | `true` | Enable automatic day/night mode switching |
| `day_night` | `night_offset_minutes` | `30` | Start night mode N minutes after sunset |
| `day_night` | `day_offset_minutes` | `30` | Start day mode N minutes before sunrise |
| `pipeline` | `process_every_n` | `5` | Process every Nth frame |
| `location` | `latitude` / `longitude` | Frisco, TX | Location for sunrise/sunset calculation |
| `storage` | `retention_days` | `90` | Delete detection images older than this |

## License

Private project.
