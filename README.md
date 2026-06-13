# Bird Feeder AI

24/7 bird detection system using a Raspberry Pi 5, Hailo AI HAT+ 2 (Hailo-10H NPU), and an SV3C 4K PTZ camera at a bird feeder in Frisco, TX.

A two-stage AI pipeline detects and classifies birds to one of 555 North American species. All inference runs on-device with no cloud dependency.

## Architecture

```
Camera (RTSP) --> YOLO11x (COCO bird) --> Tracker --> ViT-Small ---------> DB
  or video file   "Is there a bird?"       Dedup      "What species?"     SQLite
                                                       555 NABirds        + images
```

1. **Detection** — YOLO11x finds birds (COCO class 14) and outputs bounding boxes.
2. **Tracking** — a Kalman + IOU tracker (Norfair) links a bird across frames so one bird sitting for 30 seconds is logged once.
3. **Classification** — the cropped bird is classified by ViT-Small. Logits are averaged across all frames of a track before softmax for a more stable prediction than any single frame.
4. **Storage** — saved to SQLite with species, confidence, and bbox, plus one full-frame image (crops/thumbnails/annotated views are generated on the fly).

**Backends:** Ultralytics/PyTorch (dev on Mac MPS / CUDA / CPU), ONNX Runtime (cross-platform), and Hailo HEF (production on the Pi).

## Hardware

| Component | Model |
|---|---|
| Compute | Raspberry Pi 5 (8 GB) |
| AI Accelerator | Hailo AI HAT+ 2 (Hailo-10H, 40 TOPS INT4 / 20 TOPS INT8) |
| Camera | SV3C 4K PTZ (RTSP stream) |
| Location | Frisco, TX |

## Setup

Requires Python 3.13.

**Mac / Linux (development):**

```bash
uv sync                    # core
uv sync --extra training   # adds Ultralytics, ONNX, scikit-learn, etc.
```

**Raspberry Pi (Hailo inference):** the Pi needs the apt-installed `hailo_platform`, so build the venv against the system Python:

```bash
sudo apt install hailo-all
uv venv --python /usr/bin/python --system-site-packages .venv
uv sync
```

If `import cv2` fails inside the venv, symlink the system OpenCV in:

```bash
ln -sf /usr/lib/python3/dist-packages/cv2.cpython-313-aarch64-linux-gnu.so \
       .venv/lib/python3.13/site-packages/
```

Then copy and edit the config (gitignored — it holds camera credentials):

```bash
cp config/config.yaml.example config/config.yaml
```

## Running the pipeline

```bash
python -m src.pipeline.pipeline --mode dev      # Mac/PC, PyTorch
python -m src.pipeline.pipeline --mode hailo    # Raspberry Pi, Hailo NPU
```

The RTSP URL comes from `config/config.yaml`; the pipeline runs until Ctrl+C.

Common flags: `--image PATH` / `--video PATH` to run on a file instead of the camera, `--output` to write an annotated video, `--virtual-rtsp` to loop a video through the full RTSP path (saves to the DB), `--no-save` to log only, and `--log-level DEBUG` for per-frame timing. Run with `--help` for the full list.

The `scripts/` directory has `start_pipeline.sh` / `stop_pipeline.sh` / `start_backend.sh` / `stop_backend.sh` to run things detached with PID tracking in `run/`.

## API server

```bash
uvicorn src.backend.api:app --host 0.0.0.0 --port 8000
```

- **Dashboard** — `/dashboard`: filterable gallery with stats (totals, unique species, today's count, avg confidence).
- **Review** — `/review`: step through pending detections. Keys: **A** confirm, **X** false positive, **C** correct species.
- Detection, stats, species, and weather data are served under `/api/*`.

Weather is backfilled and polled hourly by `src/backend/weather_ingest.py` (Open-Meteo) into the `weather_observations` table.

Corrections made in the review UI are used as labels when exporting training data, so they feed back into the next model.

## Monitoring dashboard (Grafana)

A provisioned Grafana dashboard lives in `grafana/`, fronted by Caddy with basic auth. The SQLite datasource is opened read-only.

```bash
cp grafana/.env.example grafana/.env
# generate a bcrypt hash for the basic-auth password (double every $ to $$ for compose):
docker run --rm caddy:2 caddy hash-password --plaintext 'yourpassword'
cd grafana && docker compose --env-file .env up -d
```

See `grafana/README.md` for dashboard editing and exposure notes.

## Training

Fine-tunes a pretrained model on the NABirds dataset (555 species). Both architectures run in a single pass on the Hailo-10H.

| Model | Input | Params | Hailo-10H FPS |
|---|---|---|---|
| **ViT-Small** (default) | 224×224 | 21.1M | 116 |
| **ViT-Base** | 224×224 | 86.5M | 57 |

ViT-Small is the primary model — self-attention excels at fine-grained classification (TransFG, AAAI 2022: 89.9% on NABirds). Both are pretrained on ImageNet-21K (timm `augreg_in21k_ft_in1k`) and trained end-to-end with ReduceLROnPlateau + early stopping.

```bash
# 1. download NABirds to data/nabirds/ from https://dl.allaboutbirds.org/nabirds
# 2. train
python -m src.training.train                 # ViT-Small
python -m src.training.train --model vit_base # ViT-Base (--resume to continue)
# 3. export to ONNX
python -m src.training.export_onnx classifier
```

Deployment chain: `PyTorch (.pth) --> ONNX --> Hailo DFC compiler --> HEF`. The DFC compiler only runs on x86_64 Linux — use `scripts/convert_to_hef.ipynb` in Google Colab (DFC wheel from the [Hailo Developer Zone](https://hailo.ai/developer-zone/software-downloads/)). Pre-compiled detector HEFs are available from the [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo).

Accumulated detections can be exported back into labeled training data with `python -m scripts.export_training_data --format both` (supports `--min-confidence`, `--reviewed-only`, `--since`, and perceptual-hash dedup).

## Configuration

All settings live in `config/config.yaml` (gitignored). Key ones:

| Section | Setting | Default | Description |
|---|---|---|---|
| `camera` | `rtsp_url` | — | Camera RTSP stream URL |
| `camera` | `codec` | `h264` | `h264` (FFMPEG) or `h265` (GStreamer SW decode) |
| `bird_detection` | `hef_model` | `yolov11x.hef` | Hailo detector HEF (hailo mode) |
| `bird_detection` | `confidence_threshold` | `0.5` | Min YOLO bird-detection confidence |
| `species_classification` | `hef_model` | `vit_small_birds.hef` | Classifier HEF |
| `species_classification` | `confidence_threshold` | `0.60` | Min species-classification confidence |
| `pipeline` | `process_every_n` | `5` | Process every Nth frame |
| `pipeline` | `species_cooldown_seconds` | `120` | Suppress duplicate saves of same species |
| `pipeline` | `min_frames_for_detection` | `3` | Consecutive frames before a track is classified |
| `storage` | `retention_days` | `90` | Delete detection images older than this |

## License

Private project.
