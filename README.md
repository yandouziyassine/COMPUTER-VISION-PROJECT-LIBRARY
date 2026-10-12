# Computer Vision Project Library

A hands-on library of **20 computer vision projects**, from classical image processing and
real-time tracking to segmentation, 3D and foundation models, built to master the field
end to end.

Each project lives in its own folder with its own `README`, `requirements.txt` and demo,
so every one can be read, installed and run on its own.

## Projects

**Status:** ✅ done · 🚧 in progress · 📋 planned

### Level 1: Foundations
| # | Project | What it does | Key skills | Status |
|---|---|---|---|---|
| 01 | [Virtual Painter](01-virtual-painter/) | Draw in the air with hand gestures | MediaPipe hands, real-time video, FastAPI streaming | ✅ |
| 02 | Solar Panel Defect Classifier | Classify panels into 6 defect types | Transfer learning, class imbalance, Grad-CAM | 📋 |
| 03 | Document Scanner + Receipt OCR | Photo of a receipt → flattened scan → structured JSON | Contours, perspective transform, OCR | 📋 |
| 04 | AI Gym Coach | Count reps and flag bad squat/push-up form | Pose estimation, joint angles, state machines | 📋 |
| 05 | Sign Language Translator | Webcam ASL letters/words → text | Landmark features, sequence models | 📋 |

### Level 2: Detection & Tracking
| # | Project | What it does | Key skills | Status |
|---|---|---|---|---|
| 06 | License Plate Recognition | Read plates from video | YOLO detection + OCR pipeline | 📋 |
| 07 | PPE Safety Monitor | Detect missing hard hats / vests and alert | Custom YOLO training, labeling, mAP | 📋 |
| 08 | Traffic Analytics | Vehicle counting, speed estimation, heatmaps | Multi-object tracking, homography | 📋 |
| 09 | Retail Shelf Monitor | Detect empty shelf gaps, count products | Dense small-object detection | 📋 |
| 10 | Face Recognition Attendance | Log arrivals with anti-spoofing | Face embeddings, vector search, liveness | 📋 |

### Level 3: Segmentation, 3D & Specialized Domains
| # | Project | What it does | Key skills | Status |
|---|---|---|---|---|
| 11 | Medical Image Segmentation | Outline tumors / skin lesions | U-Net, Dice/IoU | 📋 |
| 12 | Satellite Land-Cover Mapping | Segment buildings, roads, water, forest | Geospatial tiling, semantic segmentation | 📋 |
| 13 | Unsupervised Anomaly Detection | Find factory defects with no defect examples | PatchCore, MVTec AD | 📋 |
| 14 | Monocular Depth → 3D | One photo → depth map → point cloud | Depth estimation, Open3D, camera geometry | 📋 |
| 15 | Fall Detection | Alert when a person falls on video | Action recognition, temporal models | 📋 |

### Level 4: Foundation Models & Production
| # | Project | What it does | Key skills | Status |
|---|---|---|---|---|
| 16 | Text-to-Image Search Engine | Search photos by typing a description | CLIP embeddings, FAISS, web app | 📋 |
| 17 | Auto-Labeling Tool | Text prompt → boxes + masks for a whole dataset | Grounding DINO + SAM 2, zero-shot | 📋 |
| 18 | Video Q&A Assistant | Ask questions about a video, get timestamped answers | Vision-language models, retrieval | 📋 |
| 19 | Edge Deployment Benchmark | Same detector on PyTorch vs ONNX vs TensorRT/INT8 | Quantization, latency/FPS profiling | 📋 |
| 20 | Football Analytics (capstone) | Player/ball tracking, team ID, 2D tactical map | Detection + tracking + clustering + homography | 📋 |

## Repository layout

```
COMPUTER-VISION-PROJECT-LIBRARY/
├── README.md               ← you are here
├── common/                 ← small utilities shared across projects
├── _template/              ← starting point for every new project
├── 01-virtual-painter/
├── 02-…/
└── …
```

## Conventions

- **One folder per project**, named `NN-short-name`, with its own `README.md`, `requirements.txt` and isolated `.venv`.
- **Every README shows a demo GIF or image first**, then results (metrics, FPS), then how to run it.
- **No datasets or model weights in git.** Each project documents where to get its data (or ships a download script). See `.gitignore`.
- **Report results honestly**, including what didn't work and why.

## Getting started with a project

```bash
cd 01-virtual-painter
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Then follow that project's README.

## License

[MIT](LICENSE) © Yassine Yandouzi
