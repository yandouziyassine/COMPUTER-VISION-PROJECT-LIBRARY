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
| 14 v2 | KITTI Eigen test (652 frames + projected LiDAR, Geiger et al. / Uhrig et al.) | ⚠️ CC BY-NC-SA 3.0 | Streamed on Kaggle from the Marigold authors' public evaluation archive (646 MB), never stored or redistributed; no signed agreement involved |
| 14 v2 | DIODE val (771 images, Vasiljevic et al. 2019) | ✅ MIT | Streamed on Kaggle from the Marigold evaluation archive (6 GB) |
| 14 v2 | ETH3D high-res multi-view training DSLR images + laser depth (454) | ⚠️ CC BY-NC-SA 4.0 | Streamed on Kaggle from the Marigold evaluation archive (46 GB), only 1/4-resolution arrays kept |
| 14 v2 | iBims-1 core set (100 images, Koch et al. 2018, TUM mediaTUM) | ✅ CC BY 4.0 | 170 MB, public FTP listed on mediaTUM. Frames are shown in the public W9 pair set (`14-monocular-depth-3d/docs/w9_public_img/`) with attribution |
| 14 v2 | W9 mirror and illusion photos (11, Wikimedia Commons) | ✅ Public domain / CC BY 2.0 / CC BY-SA 4.0 | Credits per image in `14-monocular-depth-3d/docs/w9_public_pairs.json`; poster composites on iBims-1 walls include CC BY-SA / CC0 Commons photos (share-alike applies to those composites) |
| 14 v2 | MoGe evaluation benchmarks (HF `Ruicheng/monocular-geometry-evaluation`: NYUv2, KITTI) | ❓ not stated | Re-processed research datasets; downloaded on Kaggle only to run MoGe's official evaluation (E1), never displayed or redistributed |
| 14 v2 | Booster (Ramirez et al., CVPR 2022, Univ. Bologna) | ⚠️ CC BY-NC 4.0 | **Not used yet**: the monocular test split (1.7 GB) has no public ground truth (online benchmark); the labeled set is a 19 GB zip whose server did not answer range requests. Decision recorded 2026-10-25 |

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
| 14 v2 | Depth Anything V2 Large | ⚠️ CC BY-NC 4.0 | Non-commercial; benchmark only |
| 14 v2 | Depth Pro (`apple/DepthPro-hf`) | ⚠️ Apple ML Research Model licence | Non-commercial research use; benchmark only |
| 14 v2 | MoGe-2 ViT-L (`Ruicheng/moge-2-vitl`, Microsoft) | ✅ MIT | |
| 14 v2 | Marigold Depth LCM v1.0 (`prs-eth/marigold-depth-lcm-v1-0`) | ✅ Apache-2.0 | Built on Stable Diffusion 2 |
| 12, 16 | OpenCLIP ViT-B/32 (`laion2b_s34b_b79k`) | ✅ MIT | |
| 17 | Grounding DINO tiny, SAM ViT-B | ✅ Apache-2.0 | |
| 18 | SmolVLM2-500M-Video-Instruct | ✅ Apache-2.0 | |

## What this means for publishing the repo

- Publishing **code + small result figures** for learning/portfolio purposes is fine.
- Don't commit or redistribute the datasets themselves (`.gitignore` already blocks data and weights).
- **Dataset images are only shown when their licence allows redistribution** (CC0 / public domain / CC BY /
  CC BY-SA / MIT). Figures with photos from ⚠️ or ❓ sources (02, 08, 09, 10, 11, 13, 14 NYU/KITTI/ETH3D,
  15, 16/17/18 COCO, 20 broadcast frames, 07 watermarked photos) were removed on 2026-10-26; READMEs show
  own charts or a note instead. Older versions remain in git history.
- Anything commercial would need: replacing AGPL YOLO (or a license), and replacing the ⚠️ datasets.
