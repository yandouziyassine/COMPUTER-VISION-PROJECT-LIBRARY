# 13 · Unsupervised Anomaly Detection (PatchCore from scratch)

Finds and localizes manufacturing defects **without ever seeing a defect during training**.
A from-scratch PyTorch implementation of **PatchCore** (Roth et al., *Towards Total Recall in
Industrial Anomaly Detection*, CVPR 2022), about 150 lines with no anomaly-detection library,
evaluated on the MVTec AD benchmark on a laptop CPU.

**▶ Try it live:** [Hugging Face demo](https://huggingface.co/spaces/yandyassine/patchcore-defect-detection), running entirely in your browser (ONNX Runtime Web, WebGPU or WebAssembly; no server).
Model files and card: [yandyassine/patchcore-mvtec-ad](https://huggingface.co/yandyassine/patchcore-mvtec-ad).

![metal_nut results](docs/metal_nut_examples.png)

*Top: defective test parts. Middle: ground-truth defect masks. Bottom: PatchCore anomaly heatmap.
The model was fitted on defect-free nuts only.*

## Results

| Category | Image AUROC | Pixel AUROC | Paper (image / pixel)* | Fit time (CPU) | Inference |
|---|---|---|---|---|---|
| bottle | **1.000** | **0.984** | 1.000 / 0.986 | 10.4 min | 394 ms/img |
| metal_nut | **0.999** | **0.985** | 1.000 / 0.984 | 9.6 min | 566 ms/img |
| grid | **0.977** | **0.977** | ~0.98 / 0.987 | 14.1 min | 530 ms/img |

\*PatchCore-10 % with a WideResNet-50 backbone at 224 px, as reported in the paper.

- **Image AUROC**: can it tell defective parts from good ones? 1.0 = perfect ranking.
- **Pixel AUROC**: does the heatmap light up the right pixels (compared with the ground-truth masks)?

The from-scratch implementation reproduces the published numbers within ~1 point.

<p float="left">
  <img src="docs/bottle_examples.png" width="49%" />
  <img src="docs/grid_examples.png" width="49%" />
</p>

## How it works

Defect data is rare and expensive in real factories, but images of *good* parts are plentiful.
PatchCore only needs those.

1. **Patch features**: each image goes through an ImageNet-pretrained WideResNet-50 (frozen, no
   training). Feature maps from layers 2 and 3 are concatenated: mid-level features that are local
   enough for small scratches and semantic enough to ignore pixel noise. Each position is averaged
   with its 3×3 neighborhood, giving one 1,536-d vector per 8×8-pixel patch (784 per image).
2. **Memory bank**: patch vectors from all good training images (~165k–207k) are collected.
3. **Coreset subsampling**: greedy farthest-point sampling (on a random 128-d projection) keeps
   10 % of the patches while still covering the whole feature space. That makes the bank 10× smaller
   and nearest-neighbor search 10× faster, with almost no accuracy loss.
4. **Scoring**: each test patch's anomaly score is the distance to its nearest neighbor in the bank.
   A patch that looks like nothing seen on good parts scores high. The image score is the max patch
   score; the patch scores, upsampled and Gaussian-smoothed (σ = 4), form the heatmap.

Code: [`patchcore.py`](patchcore.py) (the method) · [`run.py`](run.py) (data loading, evaluation, figures).

## Data

- **Source:** [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad), license
  CC BY-NC-SA 4.0 (non-commercial). Downloaded from the Hugging Face mirror `foersben/mvtec-ad`
  (original folder layout + license), because the original per-category links returned 404.
- **Categories used:** bottle (209 train / 83 test), metal_nut (220 / 115), grid (264 / 78).
  Test sets mix good parts and several real defect types, with pixel-level masks.
- **Location:** `../../cv-data/mvtec/<category>/` (not committed).

## Run it

```bash
.venv\Scripts\activate            # shared environment, from the repo root
cd 13-anomaly-detection-patchcore
python run.py --categories bottle metal_nut grid
```

Get the data (≈ 400 MB for these three categories):

```bash
python -c "from huggingface_hub import snapshot_download as d; d('foersben/mvtec-ad', repo_type='dataset', local_dir='../../cv-data/mvtec', allow_patterns=['bottle/**','metal_nut/**','grid/**','license.txt'])"
```

## Browser demo (no server)

The [live demo](https://huggingface.co/spaces/yandyassine/patchcore-defect-detection) runs the model on the
visitor's own device with ONNX Runtime Web (WebGPU when available, WebAssembly otherwise), so it is free to host
and uploaded photos never leave the browser.

- [`web/build.py`](web/build.py) exports two ONNX graphs: the backbone (image → 784 patch features), and one
  nearest-neighbor graph per category with the memory bank baked in, written as one matrix multiplication
  (`|q|² + |b|² − 2 q·b`). Weights are stored as float16 and cast back to float32 inside the graph: ~50 MB per
  file instead of ~100 MB, with float32 math. It also picks each category's threshold (best F1 on the test set).
- [`web/validate.py`](web/validate.py) re-runs the full test set through the ONNX files: **image AUROC and F1
  are identical to PyTorch** (1.000 / 0.999 / 0.977; [`docs/onnx_validation.json`](docs/onnx_validation.json)).
- [`web/index.html`](web/index.html) is the whole demo page. In the browser, one image takes ~0.3 s with WebGPU
  and ~2 s with WebAssembly on one thread (laptop measurements).

```bash
python web/build.py && python web/validate.py     # writes to ../../cv-data/mvtec/web/
```

## What I learned / what didn't work

- **Grid is the hardest** (0.977): it's a repeating texture, so a defect patch can look locally
  similar to a *different* part of a normal grid. Textures need more context per patch than objects.
- **The coreset loop dominates fit time** (~10 min of each fit). It's a Python loop of ~16k–21k
  iterations, each computing distances to all patches. A batched or GPU version would cut this to
  seconds; at inference it doesn't matter.
- **Image AUROC isn't the whole story**: a factory needs a *threshold*. Picking one means trading
  missed defects against false alarms on the validation set, which is the next step for deployment.

## Next steps

- Choose an operating threshold per category (e.g. for 99 % defect recall) and report the false-alarm rate.
- Run all 15 MVTec categories.
- Export the backbone to OpenVINO (see project 19) to bring desktop inference from ~500 ms to real time
  (the browser demo already does it in ~0.3 s with WebGPU).

## Tech stack

Python · PyTorch · torchvision (WideResNet-50) · scikit-learn (AUROC) · Matplotlib

## Run with Docker

CPU-only image (`python:3.10-slim` + CPU PyTorch). The dataset is not baked in: the `cv-data` folder is mounted
at `/data`, and the WideResNet-50 weights are cached there on first run. Commands from this folder:

```bash
docker build -f Dockerfile -t cv-patchcore ..          # build context = repo root (needs common/)
docker run --rm \
  -v "$(pwd)/../../cv-data:/data" \
  -v "$(pwd)/../../cv-data/mvtec/outputs/docker-docs:/app/13-anomaly-detection-patchcore/docs" \
  cv-patchcore --categories bottle
```

The second mount collects `results.json` and the heatmap figures outside the repo (so the committed `docs/`
isn't overwritten). On Windows PowerShell, use `${PWD}` instead of `$(pwd)`.
