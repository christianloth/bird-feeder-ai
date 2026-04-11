# Bird Feeder AI - Project Plan

## Overview
A 24/7 bird species detection and tracking system using a Raspberry Pi 5 + AI HAT+ 2 (Hailo NPU) pointed at a bird feeder in Frisco, TX. Captures RTSP stream from an SV3C 4K PTZ WiFi camera, detects birds, classifies species, and stores sighting data for seasonal analysis.

## Hardware
- Raspberry Pi 5
- Raspberry Pi AI HAT+ 2 (Hailo-10H, 40 TOPS)
- SV3C 4K PTZ WiFi Camera (RTSP stream)
- SanDisk MAX Endurance 128GB microSDXC (SDSQQVR-128G-GN6IA) for OS + storage
- Optional: USB microphone for BirdNET audio classification

## Architecture

```
SV3C Camera (RTSP 1080p)
       |
[GStreamer RTSP Pipeline]
       |
[YOLOv8n - Bird Detection] ── Hailo NPU (COCO class 14 = "bird")
       |
[Crop Bird ROI]
       |
[MobileNetV2 - Species Classification] ── Hailo NPU (fine-tuned on NABirds)
       |
[FastAPI Backend]
  ├── SQLite Database (detections, species, weather)
  ├── Image Storage (crops + thumbnails on SD card)
  ├── Open-Meteo Weather Correlation
  └── REST API (for future dashboard)
```

---

## Phases

### Phase 1: ML Foundations (PyTorch Learning)
> **Goal:** Learn PyTorch fundamentals by building the bird species classifier from scratch.

| Step | Task | Who | Why |
|------|------|-----|-----|
| 1.1 | Project scaffolding, config, dependencies | Claude | Boilerplate — no ML learning value |
| 1.2 | Download & explore NABirds dataset | You | Understand your data before modeling. Look at class distributions, image sizes, quality variation |
| 1.3 | Write the PyTorch Dataset class | You | **Core learning.** You need to understand how PyTorch loads data — `__init__`, `__len__`, `__getitem__`, image loading, label mapping |
| 1.4 | Write data augmentation transforms | You | **Core learning.** Understand why augmentation prevents overfitting — RandomHorizontalFlip, RandomRotation, ColorJitter, Normalize. Learn torchvision.transforms |
| 1.5 | Write the DataLoader setup | You | Learn batch sizes, shuffling, num_workers, train/val/test splits |
| 1.6 | Write the transfer learning model setup | You | **Core learning.** Load pretrained MobileNetV2, freeze backbone, replace final classifier head. Understand what freezing layers means and why |
| 1.7 | Write the training loop | You | **THE most important step.** Forward pass, loss computation (CrossEntropyLoss), backward pass, optimizer.step(), learning rate scheduling. This is the heart of PyTorch |
| 1.8 | Write the evaluation/metrics code | You | Accuracy, precision, recall, confusion matrix, per-species accuracy. Understand what these metrics mean |
| 1.9 | Experiment: unfreeze backbone, lower LR, retrain | You | Learn fine-tuning vs feature extraction. See how unfreezing improves accuracy |
| 1.10 | Export trained model to ONNX | Claude + You | Semi-mechanical, but you should understand the export signature |

**Deliverables:** A trained MobileNetV2 bird classifier in ONNX format, and solid PyTorch fundamentals.

### Phase 2: Edge Deployment (Hailo NPU)
> **Goal:** Learn model quantization and edge deployment.

| Step | Task | Who | Why |
|------|------|-----|-----|
| 2.1 | Set up Hailo DFC environment (x86 Linux) | Claude | Environment setup — follow docs |
| 2.2 | Convert ONNX → HEF (Hailo format) | You + Claude | You should understand the quantization pipeline: parsing, calibration, INT8 conversion, compilation |
| 2.3 | Validate HEF accuracy vs original | You | Compare INT8 quantized accuracy against your FP32 model. Understand the accuracy/speed tradeoff |
| 2.4 | Download pre-compiled YOLOv8n HEF from Hailo Model Zoo | Claude | Detection model — no need to train, COCO already has "bird" class |

**Deliverables:** YOLOv8n HEF (detection) + MobileNetV2 HEF (classification) ready for Hailo NPU.

### Phase 3: RTSP Pipeline & Detection
> **Goal:** Build the real-time capture and inference pipeline.

| Step | Task | Who | Why |
|------|------|-----|-----|
| 3.1 | RTSP capture module (GStreamer/OpenCV) | Claude | Plumbing — GStreamer syntax is not ML learning |
| 3.2 | Hailo inference wrapper (detection) | Claude | Hardware integration boilerplate |
| 3.3 | Bird ROI cropping + tracking logic | Claude | Computer vision utility code |
| 3.4 | Hailo inference wrapper (classification) | Claude | Same pattern as detection wrapper |
| 3.5 | Integration: RTSP → detect → classify → output | You + Claude | You wire it together. Understand the data flow between stages |

**Deliverables:** Working pipeline that reads camera, detects birds, classifies species, and prints results to console.

### Phase 4: Backend & Storage
> **Goal:** Build the persistence and API layer.

| Step | Task | Who | Why |
|------|------|-----|-----|
| 4.1 | SQLite schema + models (SQLAlchemy) | Claude | DB boilerplate |
| 4.2 | Image saving service (crops + thumbnails) | Claude | File I/O utility |
| 4.3 | Detection storage service | Claude | CRUD operations |
| 4.4 | Open-Meteo weather integration | Claude | API client code |
| 4.5 | FastAPI REST endpoints | Claude | Standard API patterns |
| 4.6 | Main application entry point | You + Claude | You wire pipeline → backend. Understand how detection events flow into storage |

**Deliverables:** Full backend that stores every bird sighting with species, confidence, image crop, and weather conditions.

### Phase 5: Audio Classification (Optional)
> **Goal:** Add BirdNET audio identification and cross-reference with visual detections.

| Step | Task | Who | Why |
|------|------|-----|-----|
| 5.1 | BirdNET integration module | Claude | Library wrapper |
| 5.2 | Audio capture from USB mic | Claude | Hardware I/O |
| 5.3 | Cross-reference audio + visual detections | You + Claude | Interesting ML problem — how to fuse two modalities |

### Phase 6: Dashboard (Future)
> **Goal:** Visualize bird activity, seasonal trends, species diversity.

| Step | Task | Who | Why |
|------|------|-----|-----|
| 6.1 | Grafana + TimescaleDB migration | Claude | Infrastructure |
| 6.2 | Dashboard design and queries | You + Claude | You decide what questions to answer with data |

---

## What You'll Learn

By the end of Phase 1, you'll understand:
- PyTorch Dataset, DataLoader, transforms
- Transfer learning (feature extraction vs fine-tuning)
- Training loops (forward, loss, backward, optimize)
- Learning rate scheduling
- Model evaluation metrics
- ONNX export

By the end of Phase 2, you'll understand:
- Model quantization (FP32 → INT8)
- Edge deployment constraints
- Accuracy vs latency tradeoffs

By the end of all phases:
- Full ML pipeline from training to production
- Real-time inference on edge hardware
- Backend architecture for ML systems
- Time-series data analysis

---

## Project Structure

```
bird-feeder-ai/
├── PLAN.md                          # This file
├── README.md                        # Project overview (create when ready)
├── pyproject.toml                   # Dependencies
├── config/
│   └── settings.py                  # Camera URL, model paths, thresholds
├── data/
│   └── nabirds/                     # NABirds dataset (gitignored)
├── models/
│   ├── bird-classifier/             # Species classifier checkpoints (gitignored)
│   ├── onnx/                        # Exported ONNX models (gitignored)
│   └── hef/                         # Compiled Hailo HEF files (gitignored)
├── notebooks/
│   └── 01_explore_dataset.ipynb     # Dataset exploration
├── src/
│   ├── training/
│   │   ├── dataset.py               # YOU WRITE: PyTorch Dataset class
│   │   ├── transforms.py            # YOU WRITE: Data augmentation
│   │   ├── model.py                 # YOU WRITE: MobileNetV2 transfer learning setup
│   │   ├── train.py                 # YOU WRITE: Training loop
│   │   ├── evaluate.py              # YOU WRITE: Metrics and evaluation
│   │   └── export_onnx.py           # Export to ONNX format
│   ├── inference/
│   │   ├── detector.py              # YOLOv8n Hailo detection wrapper
│   │   ├── classifier.py            # MobileNetV2 Hailo classification wrapper
│   │   └── tracker.py               # Bird centroid tracking
│   ├── pipeline/
│   │   ├── camera.py                # RTSP capture (GStreamer/OpenCV)
│   │   └── pipeline.py              # Detection → Classification → Storage
│   ├── backend/
│   │   ├── database.py              # SQLite/SQLAlchemy models and schema
│   │   ├── storage.py               # Image crop/thumbnail saving
│   │   ├── weather.py               # Open-Meteo integration
│   │   └── api.py                   # FastAPI REST endpoints
│   └── audio/
│       └── birdnet.py               # BirdNET audio classification
├── tests/
│   └── ...
└── scripts/
    ├── download_nabirds.sh          # Download NABirds dataset
    └── convert_hailo.py             # ONNX → HEF conversion helper
```

---

## Key Resources

### Models
- Hailo Model Zoo: https://github.com/hailo-ai/hailo_model_zoo
- Hailo RPi5 Examples: https://github.com/hailo-rpi5-examples
- HuggingFace EfficientNetB2 (validation): https://huggingface.co/dennisjooo/Birds-Classifier-EfficientNetB2

### Datasets
- NABirds (555 NA species): https://dl.allaboutbirds.org/nabirds
- CUB-200-2011 (benchmark): https://www.vision.caltech.edu/datasets/cub_200_2011/
- Birds 525 (Kaggle): https://www.kaggle.com/datasets/gpiosenka/100-bird-species

### Audio
- BirdNET Python: https://pypi.org/project/birdnet/
- BirdNET GitHub: https://github.com/birdnet-team/birdnet

### Reference Projects
- rpi5-birdcam-hailo-bioclip: https://github.com/ekstremedia/rpi5-birdcam-hailo-bioclip
- birdwatch-ai: https://github.com/louistrue/birdwatch-ai
- BirdNET-Go: https://github.com/tphakala/birdnet-go

### Frisco TX Birds
- Frisco Birding Checklist: https://www.friscotexas.gov/DocumentCenter/View/21288/Frisco-Birding-Checklist
- BirdNET geo model can predict species by week at coordinates (33.1507, -96.8236)

---

## Current Status
- [x] Research and architecture design
- [ ] Phase 1: ML Foundations — **START HERE**
- [ ] Phase 2: Edge Deployment
- [ ] Phase 3: RTSP Pipeline
- [ ] Phase 4: Backend & Storage
- [ ] Phase 5: Audio Classification
- [ ] Phase 6: Dashboard
- [ ] Phase 7 (Bonus): EfficientNetB2 Upgrade

---

### Phase 7 (Bonus): EfficientNetB2 Upgrade
> **Goal:** Repeat the same training process with EfficientNetB2 for higher accuracy. Since the process is identical to Phase 1 (only the model load and head dimensions change), this reinforces everything you learned while producing a more accurate model (~99% vs ~94%).

| Step | Task | Who | Notes |
|------|------|-----|-------|
| 7.1 | Swap model from MobileNetV2 to EfficientNetB2 | You | `models.efficientnet_b2(weights=...)`, head becomes `Linear(1408, 555)` |
| 7.2 | Retrain on NABirds with same two-stage freeze/unfreeze approach | You | Same dataset, transforms, training loop — only the model changes |
| 7.3 | Compare accuracy against your MobileNetV2 results | You | Side-by-side evaluation: confusion matrices, per-species metrics |
| 7.4 | Export to ONNX and convert to Hailo HEF | You | Same process as Phase 2 |
| 7.5 | Deploy the better model to the Pi | You | Swap the HEF file in config, done |

**Reference repo:** [dennisjooo/Birds-Classifier-EfficientNetB2](https://github.com/dennisjooo/Birds-Classifier-EfficientNetB2) — `development.ipynb` has their full training code on Birds 525. They fine-tuned all layers (no freeze), used Adam (lr=1e-3), ReduceLROnPlateau scheduler, and hit 99.1% test accuracy in ~25 epochs on a Kaggle P100.
