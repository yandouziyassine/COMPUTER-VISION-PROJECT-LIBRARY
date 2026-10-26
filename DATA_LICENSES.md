# Data & Model Licenses

The **code** in this repository is MIT-licensed (see [LICENSE](LICENSE)). The **datasets and pretrained
models** it uses are not part of the repository (they're downloaded to `../cv-data/`) and keep their own
licenses, listed below. Several are **non-commercial** or **research-only**: fine for a learning portfolio,
but check before reusing any of this commercially.

Legend: ✅ permissive · ⚠️ non-commercial / research-only / copyleft · ❓ not verified, check the source

## Datasets

| Project | Dataset | License | Notes |
|---|---|---|---|
| 02 | Faulty Solar Panel images (Kaggle) | ❓ | License not recorded when downloaded; check the Kaggle dataset page |
| 03 | ICDAR 2019 SROIE receipts (HF mirror `jsdnrs/ICDAR2019-SROIE`) | ✅ CC BY 4.0 | Attribution: Huang et al., ICDAR 2019 |
| 05 | Sign Language MNIST (Kaggle `datamunge/sign-language-mnist`) | ✅ CC0 1.0 | |
| 05 | MediaPipe Hand Landmarker model (Google) | ✅ Apache-2.0 | Downloaded by `app.py` |
| 06 | Vehicle Registration Plates (Roboflow / Augmented Startups, HF `keremberke/license-plate-object-detection`) | ✅ CC BY 4.0 | |
| 06 | UK Car Plate VRN dataset (HF `spawn99/UK-Car-Plate-VRN-Dataset`) | ✅ MIT | Synthetic plates |
| 07 | Hard Hat Workers (Roboflow, HF `keremberke/hard-hat-detection`) | ✅ CC BY 4.0 | |
| 08 | `vehicles.mp4` sample video (Roboflow `supervision` assets) | ❓ | Distributed as a library sample asset; check Roboflow's terms before republishing the video |
| 09 | SKU-110K (Goldman et al., CVPR 2019; HF `finedet/sku110k`) | ⚠️ CC BY-NC 2.0 | Non-commercial |
| 10 | LFW, Labeled Faces in the Wild (UMass Amherst) | ⚠️ research benchmark | No explicit license; photos of public figures. Don't redistribute images |
| 11 | Kvasir-SEG (Simula, Jha et al. MMM 2020) | ⚠️ research / education | Cite the paper |
| 12 | EuroSAT RGB (Helber et al. 2019; HF `timm/eurosat-rgb`) | ✅ MIT | |
| 13 | MVTec AD (HF mirror `foersben/mvtec-ad`) | ⚠️ CC BY-NC-SA 4.0 | Non-commercial, share-alike |
| 15, 18 | UR Fall Detection Dataset (Kwolek & Kepski 2014) | ⚠️ research use | Cite the paper |
| 16, 14, 17, 18 | COCO val2017 images + annotations | ✅ CC BY 4.0 annotations | Images: Flickr, under their original licenses |
| 18 | VQAv2 questions/answers (visualqa.org) | ✅ CC BY 4.0 | |
| 20 | Football players & ball (Roboflow, HF `keremberke/football-object-detection`) | ✅ CC BY 4.0 | |
| 20 | Football pitch keypoints (Kaggle `danielmachniak/football-pitch-keypoints-detection`) | ✅ MIT | 32-landmark layout as in Roboflow's open `sports` toolkit |
| 20 | Metrica Sports sample data, games 1–2 (github.com/metrica-sports/sample-data) | ❓ no formal licence | README asks for responsible use and acknowledgement of Metrica Sports; validation only, not redistributed |
| 20 | StatsBomb open data (github.com/statsbomb/open-data), shots of 597 men's matches | ⚠️ StatsBomb Public Data User Agreement | Research/analysis use, no redistribution, no commercial use; published analysis must credit StatsBomb with its logo |
| 14 | NYU Depth V2 labeled subset (Silberman et al., ECCV 2012) | ⚠️ research use | Evaluated locally, never redistributed; figures show a few frames |
| 14 | Illusion photos (6, Wikimedia Commons) | ✅ CC BY 2.0 / CC BY-SA 4.0 / CC0 | Credits in `14-monocular-depth-3d/docs/illusion_credits.md` |

## Pretrained models

| Project | Model | License | Notes |
|---|---|---|---|
| 06, 07, 08, 09, 15, 17, 19, 20 | Ultralytics YOLO11n / YOLO11n-pose | ⚠️ AGPL-3.0 | Copyleft: code that uses it and is offered as a service must be released under AGPL, or an Ultralytics enterprise license is needed |
| 06 | `morsetechlab/yolov11-license-plate-detection` | ⚠️ AGPL-3.0 | Used only as a comparison baseline |
| 01 | MediaPipe Hands | ✅ Apache-2.0 | |
| 02, 12 | timm ImageNet weights (EfficientNet-B0, ResNet-18) | ✅ Apache-2.0 (timm) | |
| 03, 06 | RapidOCR / PaddleOCR models | ✅ Apache-2.0 | |
| 10 | OpenCV zoo YuNet / SFace | ✅ MIT / Apache-2.0 | |
| 11 | segmentation_models_pytorch + ResNet-18 ImageNet weights | ✅ MIT | |
| 13 | torchvision WideResNet-50 ImageNet weights | ✅ BSD-3 (torchvision) | |
| 14 | Depth Anything V2 Small / DPT-Hybrid | ✅ Apache-2.0 | |
| 14 | Depth Anything V2 Base | ⚠️ CC BY-NC 4.0 | Non-commercial; benchmark only |
| 14 | Depth Anything V2 Metric-Indoor Small | ❓ | No license on the model card |
| 12, 16 | OpenCLIP ViT-B/32 (`laion2b_s34b_b79k`) | ✅ MIT | |
| 17 | Grounding DINO tiny, SAM ViT-B | ✅ Apache-2.0 | |
| 18 | SmolVLM2-500M-Video-Instruct | ✅ Apache-2.0 | |

## What this means for publishing the repo

- Publishing **code + small result figures** for learning/portfolio purposes is fine.
- Don't commit or redistribute the datasets themselves (`.gitignore` already blocks data and weights).
- Figures showing LFW faces (project 10) are photos of public figures from a research benchmark; keep them
  as they are, but don't build a public face-search service on top of this.
- Anything commercial would need: replacing AGPL YOLO (or a license), and replacing the ⚠️ datasets.
