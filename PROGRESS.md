# Overnight Progress Log

Owner rules for this run: **no git commits, no pushes**. Everything is saved as files on the
USB key. Finish one project fully before starting the next. Data lives in `D:\cv-data\`.

## ☀️ Morning summary (2026-10-12, ~12:00)

**10 of 20 projects done** (01 from before + 9 tonight): 02, 03, 08, 11, 13, 14, 16, 17, 19.
Every one has code that ran end to end, real metrics, a README with figures, and a
`requirements.txt`. **Nothing was committed or pushed**: `git status` shows all new work as untracked.

| # | Project | Headline result |
|---|---|---|
| 02 | Solar defect classifier | 82.9 % test accuracy, Grad-CAM reveals shortcut learning |
| 03 | Receipt scanner + OCR | page found 68 % on a table; total 78 % on clean scans (after the amount-parser fix); OCR 11× faster via OpenVINO |
| 08 | Traffic analytics | 19 vehicles tracked, speeds via homography, 3.4 FPS at 4K on CPU |
| 11 | Polyp segmentation | Dice 0.897 / IoU 0.833 (U-Net, CPU) |
| 13 | PatchCore (from scratch) | image AUROC 1.000 / 0.999 / 0.977, matches the paper |
| 14 | Depth → 3D | one photo → point cloud → new viewpoints |
| 16 | CLIP image search | R@10 75.2 % on COCO 5k, zero-shot |
| 17 | Auto-labeling | zero-shot mAP 0.558 vs 0.473 for supervised YOLO11n (but 118× slower) |
| 19 | Edge benchmark | 79 ms → 4.6 ms (iGPU/NPU via OpenVINO), INT8 −0.1 mAP on held-out COCO val2017 |

**Suggested testing order** (fast to slow): 16 (`python app.py`), 02 (`python app.py`),
03 (`python app.py`), 14 (`python depth3d.py --demo`), 19 (`python benchmark.py --skip-accuracy`).
Activate the env first: `D:\COMPUTER-VISION-PROJECT-LIBRARY\.venv\Scripts\activate`.

**Not done:** 15 Fall Detection, 06 License Plates (ran out of night). 04 Gym Coach deferred
(needs videos with known rep counts; you could film a few sets). 05/07/09/12/18/20 blocked as
planned (login, dataset size, GPU or API key).

**Things to know:**
- All data, weights and caches: `D:\cv-data\` (~6 GB). Repo folder has only code + small figures.
- `08-traffic-analytics/docs/demo.gif` is 2.0 MB (slightly over the 2 MB guideline).
- Global git config got one line: `safe.directory D:/COMPUTER-VISION-PROJECT-LIBRARY` (FAT32 needs it).
- Obsidian notes are up to date in `SuperVault/Projets/CV-Project-Library/`.

## Plan (in order)

| # | Project | Status | Notes |
|---|---|---|---|
| — | Setup: repo copied to USB, shared `.venv`, CLAUDE.md, common/paths.py | ✅ | `.venv` on the key works: torch 2.14.1+cpu, cv2 5.0.0 (install took ~2 h on FAT32) |
| 02 | Solar Defect Classifier | ✅ | Test acc 82.9 %, macro F1 0.830. README, figures, Gradio app tested |
| 13 | Unsupervised Anomaly Detection (PatchCore) | ✅ | Image AUROC bottle 1.000 / metal_nut 0.999 / grid 0.977, pixel AUROC 0.977–0.985, matches the paper. Data from HF mirror `foersben/mvtec-ad` |
| 08 | Traffic Analytics | ✅ | 19 vehicles tracked, 2 in / 5 out, median 119 km/h, 3.4 FPS at 4K on CPU. Ran with the USB `.venv` |
| 19 | Edge Deployment Benchmark (CPU / iGPU / NPU) | ✅ | PyTorch CPU 79 ms → OpenVINO iGPU 4.7 ms / NPU 4.6 ms (~17×). INT8: −0.1 mAP on 500 held-out COCO val2017 images (first version wrongly used COCO128 = train data; fixed after review). Found and fixed a hybrid-CPU thread trap in ONNX Runtime |
| 16 | Text-to-Image Search | ✅ | COCO 5k text→image R@1 39.1 % / R@5 65.5 % / R@10 75.2 % (matches published zero-shot). Gradio app tested |
| 14 | Monocular Depth → 3D | ✅ | **2026-10-14 upgrade (★★★):** NYU benchmark of 4 models (DA2-Base AbsRel 0.058), failure map by surface (mirrors 7× walls; sensor coverage reported), poster illusion test (DA2 not fooled, DPT is), illusion gallery, metric measurement (17 % → 10 % median error with one known length), see-vs-infer figure, Gradio measuring app. Before: Depth Anything V2 Small ~650 ms/img on CPU, .ply clouds, z-buffer novel-view renders + GIF. No ground-truth metric (no labeled depth data); documented |
| 03 | Document Scanner + Receipt OCR | ✅ | 100 SROIE receipts, synthetic phone photos. Table: page found 68 % (0.6 % median corner error); clean-scan date 86 % / total 78 % (re-run after the review fix; was 79 %). RapidOCR via OpenVINO 11× faster than ONNX Runtime. Fails on white/busy backgrounds (documented) |
| 04 | AI Gym Coach (video input) | ⏭️ | Deferred: needs exercise videos with true rep counts to measure anything. RepCount/Countix videos are YouTube-derived (not downloaded). Owner could record a few clips and count reps by hand → then build it |
| 11 | Medical Segmentation (Kvasir-SEG) | ✅ | U-Net ResNet-18 on CPU: test Dice 0.897 / IoU 0.833. Training job was killed in epoch 20 (best ckpt from epoch 14 kept; history rebuilt from log; train.py now saves history every epoch) |
| 15 | Fall Detection | ✅ | UR Fall (30 falls / 40 ADL). Rules: sens 80 % / spec 95 %. Learned (grouped 5-fold CV): sens 100 % / spec 80 % at thr 0.5; 96.7 % / 100 % at thr 0.9 (post-hoc, flagged as optimistic). Latency ~0.6–0.7 s |
| 17 | Auto-Labeling Tool | ✅ | 200 COCO images, 10 classes: Grounding DINO-T zero-shot mAP 0.558 vs YOLO11n supervised 0.473, but 9.4 s vs 0.08 s per image. Open-vocab demo with SAM masks |
| 06 | License Plate Recognition | ✅ | YOLO11n fine-tuned on CPU (1,500 imgs, 12 ep, 2.4 h): test mAP50 0.983 vs public model 0.993. OCR on 300 synthetic UK plates: exact 89.7 % clean / 80.3 % augmented (after padding fix + UK format rules; was 0 % before the padding fix) |

## Next queue (after 06), with login-free data found

| Order | # | Project | Plan / data |
|---|---|---|---|
| ✅ | 10 | Face Recognition | DONE: YuNet + SFace (OpenCV zoo). LFW 10-fold 99.35 % ± 0.37, TAR@FAR0.1 % 98.8 %. Attendance CLI tested (2 enrolled + 1 unknown). Ethics section. Anti-spoofing not implemented |
| ✅ | 12 | Satellite Land-Cover | DONE: EuroSAT RGB. ResNet-18 fine-tuned 98.07 % test (macro F1 0.981) vs CLIP zero-shot 47.4 % (domain shift) |
| ✅ | 07 | PPE Safety Monitor | DONE: YOLO11n fine-tuned on CPU (1,500 imgs, 10 ep). Test mAP50 0.828. Site alerts @0.25: recall 90.8 % / precision 87.8 % |
| ✅ | 05 | Sign Language | DONE 2026-10-14 (Kaggle CLI download, CC0). CNN from scratch: test 99.9 % (aug) / 100 % (no aug) vs pixel logreg ~70 %. Leakage audit (dataset = 1,704 photos × 50 variations; 1/3 of test is near-copy of train). Robustness: aug doubles accuracy at 30° / 6 px shift, but is worse under blur (not in the augmentation). MediaPipe webcam app (not measured: no labeled webcam data) |
| ✅ | 18 | Video Q&A | DONE: SmolVLM2-500M on CPU. VQAv2 (300 q): 67.0 % with the standard prompt (58.3 % with my first prompt). Zero-shot "did the person fall?": sensitivity 3–13 % vs ~97–100 % for project 15 |
| ✅ | 09 | Retail Shelf Monitor | DONE: YOLO11n on 219 SKU-110K images (CPU, 30 ep, 960 px, max_det 1000). Test (488 imgs, 76k products): mAP50 0.838 / mAP50-95 0.487; count MAE 22.1 (13.3 %), under-counts. Gap rule mostly flags missed products (documented) |
| ✅ | 20 | Football Analytics | DONE: YOLO11n CPU fine-tune (859 frames, 15 ep, 960 px). Test: player AP50 0.980, ball 0.438. Teams by jersey color, silhouette 0.80. Possession in 41/130 frames. **2026-10-14:** tactical map added: YOLO11n-pose on 32 pitch landmarks (Kaggle, MIT, 222 frames, 1 h CPU) + RANSAC homography: 24/24 test frames, landmark error median 21 px, pitch position error median 2.5 m (label floor 0.8 m); settings chosen on validation |
| — | 04 | Gym Coach | still blocked (no videos with known rep counts) |

**2026-10-13:** the USB key disconnected mid-run (project 18 eval + downloads failed). Reconnected as D:, env verified; work restarted.

## 2026-10-13 evening (while 20 trains)

- **Root README rewritten** for recruiters: highlights with images, all projects with headline result and a
  depth rating (★ demo / ★★ measured / ★★★ in-depth), skills matrix, setup.
- **DATA_LICENSES.md** added: every dataset/model license; non-commercial ones (MVTec, SKU-110K, LFW,
  Kvasir, UR Fall) and AGPL (Ultralytics YOLO) flagged; two items marked "to verify" (solar Kaggle set,
  supervision sample video).
- **Code review** of all projects (reviewer agent, read-only): 0 high / 8 medium / 18 low findings.
  Fixed: 03 amount parser (thousands separators), 15 frame rate no longer hard-coded + rule debounce
  disclosed + lie-down count computed, 11 Python `random` now seeded (README notes the reported run was
  unseeded), 06/10/17 download steps documented, 18 matplotlib requirement + results merge + "simplified VQA
  accuracy" wording, 12 faiss requirement + cross-project dependency noted, 08 homography off-by-one,
  07/20 class mapping by name, 02 dataset counts (869 images, not 885) and results.json naming, 10 enrollment
  accepts all image types, README data paths (`../../cv-data`). Two findings were checked and rejected:
  02 "129 validation images" is correct; 09 image sizes match their metadata.
- **Dependency bug found while re-running 19:** installing `rapidocr-openvino` (project 03) had silently
  downgraded OpenVINO 2026.4.1 → 2024.0 (it pins `openvino<=2024.0`), which broke INT8 on the iGPU.
  Fix: OpenVINO pinned back to 2026.4.1, `rapidocr-openvino` installed with `--no-deps`, and
  `common/ov_compat.py` aliases the removed `openvino.runtime` module so the OCR still runs (verified).
- **Re-runs done (2026-10-13 night):** 03 → clean-scan total 78 % (was 79 %); 19 → INT8 −0.1 mAP held-out,
  all 8 runtime/device combos working on OpenVINO 2026.4.1. Previously listed as pending: 03 `evaluate.py --n 100` (new amount parser) and
  19 `benchmark.py` (accuracy now on 500 held-out COCO val2017 images instead of COCO128, which is
  train data; explicit FP16 precision on GPU/NPU).

## Engineering upgrade (2026-10-14)

- Added: `tests/` (pytest unit tests on synthetic inputs for 03, 06, 09, 13, 15, 18, 20 and `common/paths.py`; ~5 s,
  no data/network), `pyproject.toml` (ruff lint + format config, pytest config), `.github/workflows/ci.yml` (ruff +
  pytest on Ubuntu / Python 3.10 / CPU torch) with `tests/requirements-ci.txt`, Dockerfiles for 13 and 15 + root
  `.dockerignore` + "Run with Docker" in both READMEs, "Engineering" section in the root README.
- Whole repo ran through `ruff check --fix` + `ruff format`: formatting only (ASTs verified identical) plus 10
  unused imports removed and one unused loop variable renamed (`conf` → `_conf` in 03 `receipt.py`).
  Import sorting is disabled for project scripts on purpose (import order is load-bearing: `common.paths` before
  ultralytics/transformers, `common.ov_compat` before rapidocr). No result, number or figure changed.
- Test found a real limitation (kept as a strict `xfail`, code unchanged): 03 `normalize_amount` strips spaces
  before matching, so "3.00 6.00" reads as 3006.00.
- **Not verified:** Docker images were not built (C: space), CI hasn't run on GitHub yet (repo not public).
  The CI badge line in the root README is commented out until then.

## Notes for the owner

- Git on the USB key needed `git config --global --add safe.directory D:/COMPUTER-VISION-PROJECT-LIBRARY`
  (FAT32 doesn't record file ownership). Already done.
- Solar dataset: 885 images extracted, not 891. Some filenames in `archive.zip` differ only by
  upper/lower case and overwrite each other on Windows (Bird-drop: 192 instead of 208).
- Shared env: `D:\COMPUTER-VISION-PROJECT-LIBRARY\.venv` (Python 3.10, CPU PyTorch). If the
  USB key gets a different drive letter, re-run `setup_env.ps1`.

## Log

- Setup started. Repo copied from OneDrive folder to `D:\COMPUTER-VISION-PROJECT-LIBRARY`.
- The `.venv` install on the FAT32 key is very slow (FAT32 handles thousands of small files badly),
  so a **temporary** copy of the env was built in the session scratchpad on C: to develop and
  train in the meantime. Each project gets re-checked with the USB `.venv` once it's ready, and
  the temporary copy is deleted at the end.
- USB `.venv` finished installing. Project 02 re-verified with it (same 82.9 %); the temporary
  env on C: has been **deleted**. All projects since 08 ran directly on the USB env.
- 02 done: EfficientNet-B0, 15 epochs on CPU (~15 min). Grad-CAM shows shortcut learning on
  "Dusty" (looks at the roof), which is documented in its README.
