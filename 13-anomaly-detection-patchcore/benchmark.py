"""Desktop latency of PatchCore inference: PyTorch on the CPU vs OpenVINO on the CPU, the Arc integrated GPU and
the NPU of an Intel Core Ultra 7 258V, plus an accuracy check of every configuration (run after run.py).

The inference graph is split in two OpenVINO models, so each can run on its own device and precision:
- backbone: image (1,3,224,224) -> patch features (784, 1536)          [WideResNet-50 layers 1-3]
- knn:      features -> anomaly map (224,224) + image score            [memory bank baked in; one matrix
            multiplication |q|^2 + |b|^2 - 2 q.b, then bilinear upsampling + Gaussian blur]

Latency is batch 1, model inference only (image decoding / resizing excluded), median of `--runs` timed runs
after warm-up. Accuracy: image and pixel AUROC on the full test set of a few categories, to catch precision
problems (FP16 distances), compared with the PyTorch scores.

Usage:
    python benchmark.py                                  # latency on bottle's memory bank + accuracy on 3 categories
    python benchmark.py --accuracy-categories bottle     # quicker

Outputs:
    docs/latency.json, docs/latency.png, ../cv-data/mvtec/outputs/openvino/*.xml|bin (not committed)
"""

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openvino as ov
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from patchcore import FeatureExtractor, PatchCore, gaussian_blur
from run import OUT, SIZE, MVTecSplit

DOCS = Path(__file__).resolve().parent / "docs"
OV_DIR = OUT / "openvino"
GRID = SIZE // 8  # 28x28 patches


class Backbone(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.f = FeatureExtractor(backbone)

    def forward(self, x):
        feats = self.f(x)
        return feats.permute(0, 2, 3, 1).reshape(-1, feats.shape[1])  # (784, 1536)


class Knn(nn.Module):
    """Nearest-neighbor distance to the memory bank (same result as torch.cdist(...).min(1))."""

    def __init__(self, bank):
        super().__init__()
        self.register_buffer("bank", bank)
        self.register_buffer("bank_sq", (bank**2).sum(1))

    def forward(self, q):
        d2 = (q * q).sum(1, keepdim=True) + self.bank_sq[None] - 2 * q @ self.bank.T
        dists = d2.min(dim=1).values.clamp_min(0).sqrt()
        maps = dists.reshape(1, 1, GRID, GRID)
        score = maps.max()
        maps = F.interpolate(maps, size=(SIZE, SIZE), mode="bilinear", align_corners=False)
        return gaussian_blur(maps, sigma=4).reshape(SIZE, SIZE), score


def convert(module, example, path):
    """Static input shape: the NPU compiler rejects dynamic dimensions."""
    if not path.exists():
        m = ov.convert_model(module.eval(), example_input=example, input=[tuple(example.shape)])
        ov.save_model(m, str(path), compress_to_fp16=False)
    return path


def compiles(path, device, precision):
    """Try compiling in a child process first: a failing NPU compile can abort the whole process (LLVM error)."""
    cfg = {"PERFORMANCE_HINT": "LATENCY", **({"INFERENCE_PRECISION_HINT": precision} if precision else {})}
    code = f"import openvino as ov; ov.Core().compile_model(r'{path}', '{device}', {cfg!r})"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=900)
    return r.returncode == 0, (r.stderr.strip().splitlines() or ["?"])[-1][:300]


# name, backbone (device, precision), knn (device, precision). Precision None = device default
# (CPU: FP32; NPU: always FP16, it rejects the precision hint).
CONFIGS = [
    ("OpenVINO CPU FP32", ("CPU", "f32"), ("CPU", "f32")),
    ("OpenVINO iGPU FP16 (kNN FP32)", ("GPU", "f16"), ("GPU", "f32")),
    ("OpenVINO iGPU FP16", ("GPU", "f16"), ("GPU", "f16")),
    ("OpenVINO NPU (kNN on iGPU FP32)", ("NPU", None), ("GPU", "f32")),
    ("OpenVINO NPU", ("NPU", None), ("NPU", None)),
]


def compile_model(core, path, device, precision):
    cfg = {"PERFORMANCE_HINT": "LATENCY"}
    if precision is not None:
        cfg["INFERENCE_PRECISION_HINT"] = precision
    return core.compile_model(str(path), device, cfg)


def timed(fn, warmup, runs):
    for _ in range(warmup):
        fn()
    t = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        t.append((time.perf_counter() - t0) * 1000)
    return round(statistics.median(t), 2), round(float(np.percentile(t, 90)), 2)


def accuracy(predict, cat):
    """Image and pixel AUROC of one predict(x) -> (map, score) function on a category's test set."""
    scores, maps, labels, masks = [], [], [], []
    for x, y, m in DataLoader(MVTecSplit(cat, "test"), batch_size=1):
        a, s = predict(x)
        scores.append(float(s))
        maps.append(np.asarray(a, dtype=np.float32).reshape(SIZE, SIZE))
        labels.append(int(y))
        masks.append(m[0].numpy())
    scores = np.array(scores)
    img = roc_auc_score(labels, scores)
    pix = roc_auc_score(np.stack(masks).ravel().astype(np.uint8), np.stack(maps).ravel())
    return round(float(img), 4), round(float(pix), 4), scores


def run_config(core, bb_xml, banks, bdev, bprec, kdev, kprec, x0n, args):
    """Latency (backbone, kNN, total) and test-set accuracy of one device / precision configuration."""
    bb_req = compile_model(core, bb_xml, bdev, bprec).create_infer_request()
    knn_req = {c: compile_model(core, OV_DIR / f"knn_{c}.xml", kdev, kprec).create_infer_request() for c in banks}

    def predict(x, cat):
        feats = bb_req.infer({0: x})[0]
        out = knn_req[cat].infer({0: feats})
        return out[0], out[1]

    cat = args.latency_category
    feats0 = bb_req.infer({0: x0n})[0].copy()
    bb_ms = timed(lambda: bb_req.infer({0: x0n}), args.warmup, args.runs)
    knn_ms = timed(lambda: knn_req[cat].infer({0: feats0}), args.warmup, args.runs)
    total_ms = timed(lambda: predict(x0n, cat), args.warmup, args.runs)
    r = {"backbone_ms": bb_ms[0], "knn_ms": knn_ms[0], "total_ms": total_ms[0], "total_p90_ms": total_ms[1]}
    for c in args.accuracy_categories:
        r[f"{c}_image_auroc"], r[f"{c}_pixel_auroc"], _ = accuracy(lambda x, c=c: predict(x.numpy(), c), c)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latency-category", default="bottle", help="memory bank used for the latency runs")
    ap.add_argument("--accuracy-categories", nargs="*", default=["bottle", "grid", "screw"])
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--threads", type=int, nargs="+", default=[4, 8], help="PyTorch CPU thread counts to time")
    ap.add_argument("--plot-only", action="store_true", help="redraw docs/latency.png from docs/latency.json")
    args = ap.parse_args()
    if args.plot_only:
        return plot(json.loads((DOCS / "latency.json").read_text()), DOCS / "latency.png")
    OV_DIR.mkdir(exist_ok=True)
    torch.manual_seed(0)

    model = PatchCore()
    backbone = Backbone(model.backbone).eval()
    x0 = torch.randn(1, 3, SIZE, SIZE)
    bb_xml = convert(backbone, x0, OV_DIR / "backbone.xml")
    banks = {}
    for cat in {args.latency_category, *args.accuracy_categories}:
        banks[cat] = torch.load(OUT / f"{cat}_memory_bank.pt", map_location="cpu")["memory_bank"].float()
        convert(Knn(banks[cat]), torch.randn(GRID * GRID, 1536), OV_DIR / f"knn_{cat}.xml")

    core = ov.Core()
    core.set_property({"CACHE_DIR": str(OV_DIR / "cache")})
    devices = core.available_devices
    results = {
        "machine": {
            "cpu": platform.processor(),
            "openvino": ov.get_version(),
            "torch": torch.__version__,
            "devices": {d: core.get_property(d, "FULL_DEVICE_NAME") for d in devices},
        },
        "latency_category": args.latency_category,
        "memory_bank_size": int(banks[args.latency_category].shape[0]),
        "runs": args.runs,
        "configs": {},
    }
    print(results["machine"], flush=True)

    # PyTorch reference (the code path run.py uses: model.predict, batch 1).
    lat_bank = banks[args.latency_category]
    for n in args.threads:
        torch.set_num_threads(n)
        model.memory_bank = lat_bank
        with torch.no_grad():
            bb_ms = timed(lambda: model._features(x0), args.warmup, args.runs)
            total_ms = timed(lambda: model.predict(x0, SIZE), args.warmup, args.runs)
        name = f"PyTorch CPU FP32 ({n} threads)"
        results["configs"][name] = {"backbone_ms": bb_ms[0], "total_ms": total_ms[0], "total_p90_ms": total_ms[1]}
        print(name, results["configs"][name], flush=True)
    torch.set_num_threads(max(args.threads))

    for name, (bdev, bprec), (kdev, kprec) in CONFIGS:
        if bdev not in devices or kdev not in devices:
            results["configs"][name] = {"error": f"device not available ({devices})"}
            continue
        bad = [
            f"{p.name} on {d}: {msg}"
            for p, d, pr in [(bb_xml, bdev, bprec), (OV_DIR / f"knn_{args.latency_category}.xml", kdev, kprec)]
            for ok, msg in [compiles(p, d, pr)]
            if not ok
        ]
        if bad:
            results["configs"][name] = {"error": "compile failed: " + " | ".join(bad)}
            print(name, results["configs"][name], flush=True)
            continue
        try:
            results["configs"][name] = run_config(core, bb_xml, banks, bdev, bprec, kdev, kprec, x0.numpy(), args)
        except Exception as e:  # e.g. an op the NPU compiler doesn't support
            results["configs"][name] = {"error": f"failed: {str(e).strip().splitlines()[0][:300]}"}
        print(name, results["configs"][name], flush=True)

    # PyTorch accuracy reference, same code path as the OpenVINO models.
    ref = results["configs"][f"PyTorch CPU FP32 ({max(args.threads)} threads)"]
    for c in args.accuracy_categories:
        model.memory_bank = banks[c]

        def torch_predict(x):
            s, m = model.predict(x, SIZE)
            return m[0].numpy(), s[0]

        ref[f"{c}_image_auroc"], ref[f"{c}_pixel_auroc"], _ = accuracy(torch_predict, c)
    print("PyTorch reference", ref, flush=True)

    DOCS.mkdir(exist_ok=True)
    (DOCS / "latency.json").write_text(json.dumps(results, indent=1))
    plot(results, DOCS / "latency.png")


def broken(r, ref, tol=0.02):
    """True when a configuration's image AUROC drops by more than `tol` vs PyTorch on any checked category."""
    keys = [k for k in r if k.endswith("_image_auroc") and k in ref]
    return any(ref[k] - r[k] > tol for k in keys)


def plot(results, path):
    ref = next(r for n, r in results["configs"].items() if n.startswith("PyTorch") and "bottle_image_auroc" in r)
    rows = [(n, r) for n, r in results["configs"].items() if "total_ms" in r]
    rows.sort(key=lambda nr: -nr[1]["total_ms"])
    bad = [broken(r, ref) for _, r in rows]
    names = [n + ("  [wrong output]" if b else "") for (n, _), b in zip(rows, bad)]
    bb = np.array([r["backbone_ms"] for _, r in rows])
    total = np.array([r["total_ms"] for _, r in rows])
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(rows) + 1.8))
    alpha = [0.35 if b else 1.0 for b in bad]
    for i in range(len(rows)):
        ax.barh(i, bb[i], color="#4f81bd", alpha=alpha[i], label="backbone (WideResNet-50)" if i == 0 else None)
        ax.barh(
            i, total[i] - bb[i], left=bb[i], color="#f79646", alpha=alpha[i],
            label="kNN + upsample + blur" if i == 0 else None,
        )  # fmt: skip
        ax.text(total[i], i, f" {total[i]:.0f} ms", va="center", fontsize=9)
    ax.set_yticks(y, names)
    ax.set_xlabel("ms per image (batch 1, median of %d runs)" % results["runs"])
    ax.set_xlim(0, total.max() * 1.15)
    ax.set_title(
        "PatchCore inference, Intel Core Ultra 7 258V\n"
        f"(memory bank: {results['memory_bank_size']:,} patches; faded = accuracy collapsed, not usable)",
        fontsize=10,
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
