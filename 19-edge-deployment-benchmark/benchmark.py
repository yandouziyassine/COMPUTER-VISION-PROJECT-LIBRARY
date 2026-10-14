"""Benchmark one detector (YOLO11n, 640x640) across runtimes, precisions and devices of an
Intel Core Ultra laptop: PyTorch vs ONNX Runtime vs OpenVINO on CPU, integrated GPU and NPU,
in FP32 / FP16 / INT8. Also measures the accuracy cost of INT8 quantization.

Usage:
    python benchmark.py                 # export models, benchmark latency, validate accuracy
    python benchmark.py --runs 200      # more timed iterations

Outputs:
    docs/results.json, docs/latency.png, docs/results.md
"""

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import data_dir, output_dir  # noqa: E402  (must come before ultralytics)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import openvino as ov  # noqa: E402
import torch  # noqa: E402
from ultralytics import YOLO, settings  # noqa: E402

OUT = output_dir("edge")
DOCS = Path(__file__).resolve().parent / "docs"
IMGSZ = 640
settings.update({"datasets_dir": str(data_dir("ultralytics-datasets")), "runs_dir": str(OUT / "runs")})


def coco_val_subset(n=500):
    """Held-out accuracy set: the first n COCO val2017 images (YOLO11n was trained on train2017) as a
    YOLO-format dataset. COCO category ids map to YOLO's 0..79 in sorted-id order (standard coco80)."""
    root = data_dir("edge") / f"coco_val{n}"
    yaml = root / "data.yaml"
    if yaml.exists():
        return yaml
    from pycocotools.coco import COCO

    coco_dir = data_dir("clip-search")  # COCO val2017 images + annotations (projects 16/17)
    gt = COCO(str(coco_dir / "annotations" / "instances_val2017.json"))
    cat_ids = sorted(gt.getCatIds())
    to_yolo = {c: i for i, c in enumerate(cat_ids)}
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    for img_id in sorted(gt.getImgIds())[:n]:
        info = gt.loadImgs(img_id)[0]
        w, h = info["width"], info["height"]
        lines = [
            f"{to_yolo[a['category_id']]} {(a['bbox'][0] + a['bbox'][2] / 2) / w:.6f} "
            f"{(a['bbox'][1] + a['bbox'][3] / 2) / h:.6f} {a['bbox'][2] / w:.6f} {a['bbox'][3] / h:.6f}"
            for a in gt.loadAnns(gt.getAnnIds(imgIds=img_id, iscrowd=False))
        ]
        (root / "images" / info["file_name"]).write_bytes((coco_dir / "val2017" / info["file_name"]).read_bytes())
        (root / "labels" / f"{Path(info['file_name']).stem}.txt").write_text("\n".join(lines))
    names = "".join(f"  {i}: {gt.loadCats(c)[0]['name']}\n" for i, c in enumerate(cat_ids))
    yaml.write_text(f"path: {root.as_posix()}\ntrain: images\nval: images\nnames:\n{names}")
    return yaml


def export_all():
    """Export once; later runs reuse the files (INT8 calibration takes a few minutes).
    INT8 is calibrated on COCO128 (train2017 images), disjoint from the val2017 accuracy set."""
    pt = OUT / "yolo11n.pt"
    onnx, ov_fp32, ov_int8 = OUT / "yolo11n.onnx", OUT / "yolo11n_openvino_model", OUT / "yolo11n_int8_openvino_model"
    if not onnx.exists():
        onnx = Path(YOLO(str(pt)).export(format="onnx", imgsz=IMGSZ, simplify=True, verbose=False))
    if not ov_fp32.exists():
        ov_fp32 = Path(YOLO(str(pt)).export(format="openvino", imgsz=IMGSZ, verbose=False))
    if not ov_int8.exists():
        ov_int8 = Path(
            YOLO(str(pt)).export(format="openvino", imgsz=IMGSZ, int8=True, data="coco128.yaml", verbose=False)
        )
    return pt, onnx, ov_fp32, ov_int8


def time_fn(fn, x, warmup, runs):
    for _ in range(warmup):
        fn(x)
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        fn(x)
        times.append((time.perf_counter() - t) * 1000)
    return {
        "mean_ms": round(statistics.mean(times), 2),
        "p50_ms": round(statistics.median(times), 2),
        "p90_ms": round(float(np.percentile(times, 90)), 2),
        "fps": round(1000 / statistics.mean(times), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--skip-accuracy", action="store_true")
    ap.add_argument(
        "--threads",
        type=int,
        default=4,
        help="CPU threads for PyTorch / ONNX Runtime. The Core Ultra 7 258V has 4 performance "
        "+ 4 low-power cores; spreading one small model over all 8 is much slower.",
    )
    args = ap.parse_args()
    DOCS.mkdir(exist_ok=True)
    torch.set_num_threads(args.threads)

    pt, onnx, ov_fp32, ov_int8 = export_all()
    x = np.random.rand(1, 3, IMGSZ, IMGSZ).astype(np.float32)
    core = ov.Core()
    devices = core.available_devices
    print("OpenVINO devices:", {d: core.get_property(d, "FULL_DEVICE_NAME") for d in devices}, flush=True)

    results = []

    def add(runtime, precision, device, fn):
        try:
            r = time_fn(fn, x, args.warmup, args.runs)
            r.update(runtime=runtime, precision=precision, device=device)
            results.append(r)
            print(f"{runtime:12s} {precision:5s} {device:4s} {r['mean_ms']:7.2f} ms  {r['fps']:6.1f} FPS", flush=True)
        except Exception as e:  # a device/precision combo may be unsupported — record and move on
            results.append({"runtime": runtime, "precision": precision, "device": device, "error": str(e)[:200]})
            print(f"{runtime:12s} {precision:5s} {device:4s} FAILED: {str(e)[:120]}", flush=True)

    # PyTorch eager (the baseline most people deploy by default)
    torch_model = YOLO(str(pt)).model.float().eval()
    xt = torch.from_numpy(x)
    with torch.inference_mode():
        add("PyTorch", "FP32", "CPU", lambda _: torch_model(xt))

    # ONNX Runtime
    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    sess = ort.InferenceSession(str(onnx), so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    add("ONNX Runtime", "FP32", "CPU", lambda a: sess.run(None, {name: a}))

    # OpenVINO: the FP32 IR runs in FP32 on the CPU and in FP16 on the GPU/NPU (set explicitly, not left
    # to plugin defaults), plus the INT8 IR on every device.
    xml_fp32 = next(ov_fp32.glob("*.xml"))
    xml_int8 = next(ov_int8.glob("*.xml"))
    for dev in [d for d in ("CPU", "GPU", "NPU") if d in devices]:
        float_prec = "FP32" if dev == "CPU" else "FP16"
        for precision, xml in ((float_prec, xml_fp32), ("INT8", xml_int8)):
            config = {"PERFORMANCE_HINT": "LATENCY"}
            # CPU and GPU accept an explicit precision; the NPU doesn't (it always computes in FP16).
            if precision in ("FP32", "FP16") and dev != "NPU":
                config["INFERENCE_PRECISION_HINT"] = precision.lower().replace("fp", "f")
            try:
                compiled = core.compile_model(str(xml), dev, config)
            except Exception as e:
                results.append({"runtime": "OpenVINO", "precision": precision, "device": dev, "error": str(e)[:200]})
                print(f"OpenVINO     {precision:5s} {dev:4s} compile FAILED: {str(e)[:120]}", flush=True)
                continue
            req = compiled.create_infer_request()
            add("OpenVINO", precision, dev, lambda a, r=req: r.infer({0: a}))

    accuracy = {}
    previous = DOCS / "results.json"
    if args.skip_accuracy and previous.exists():  # keep the last accuracy results on latency-only runs
        accuracy = json.loads(previous.read_text()).get("accuracy", {})
    if not args.skip_accuracy:
        # mAP on 500 held-out COCO val2017 images. (A first version used COCO128, which is part of
        # train2017 — YOLO11n's own training data — and also the INT8 calibration set: not held out.)
        val_yaml = coco_val_subset(500)
        for label, path in (
            ("PyTorch FP32", pt),
            ("ONNX FP32", onnx),
            ("OpenVINO FP32", ov_fp32),
            ("OpenVINO INT8", ov_int8),
        ):
            m = YOLO(str(path), task="detect").val(
                data=str(val_yaml), imgsz=IMGSZ, batch=1, device="cpu", plots=False, verbose=False
            )
            accuracy[label] = {"mAP50-95": round(float(m.box.map), 4), "mAP50": round(float(m.box.map50), 4)}
            print(f"accuracy {label}: mAP50-95 {m.box.map:.3f}", flush=True)

    info = {
        "cpu": platform.processor(),
        "openvino_devices": {d: core.get_property(d, "FULL_DEVICE_NAME") for d in devices},
        "torch": torch.__version__,
        "onnxruntime": ort.__version__,
        "openvino": ov.__version__,
        "model": "yolo11n",
        "input": [1, 3, IMGSZ, IMGSZ],
        "runs": args.runs,
        "cpu_threads": args.threads,
        "accuracy_set": "first 500 COCO val2017 images (held out)",
        "int8_calibration_set": "COCO128 (train2017)",
    }
    with open(DOCS / "results.json", "w") as f:
        json.dump({"system": info, "latency": results, "accuracy": accuracy}, f, indent=1)

    ok = [r for r in results if "error" not in r]
    base = next(r["mean_ms"] for r in ok if r["runtime"] == "PyTorch")
    lines = [
        "| Runtime | Precision | Device | Mean latency (ms) | FPS | Speed-up vs PyTorch |",
        "|---|---|---|---|---|---|",
    ]
    for r in ok:
        lines.append(
            f"| {r['runtime']} | {r['precision']} | {r['device']} | {r['mean_ms']} | {r['fps']} | {base / r['mean_ms']:.1f}x |"
        )
    for r in results:
        if "error" in r:
            lines.append(f"| {r['runtime']} | {r['precision']} | {r['device']} | failed | — | — |")
    if accuracy:
        lines += ["", "| Model | mAP50-95 | mAP50 |", "|---|---|---|"]
        lines += [f"| {k} | {v['mAP50-95']} | {v['mAP50']} |" for k, v in accuracy.items()]
    (DOCS / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    labels = [f"{r['runtime']}\n{r['precision']} · {r['device']}" for r in ok]
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(ok) + 1.2))
    bars = ax.barh(labels, [r["mean_ms"] for r in ok], color="#4C78A8")
    ax.bar_label(bars, labels=[f"{r['mean_ms']:.1f} ms · {r['fps']:.0f} FPS" for r in ok], padding=4, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean latency per image, batch 1 (ms) — lower is better")
    ax.set_title("YOLO11n 640x640 on Intel Core Ultra 7 258V")
    ax.set_xlim(0, max(r["mean_ms"] for r in ok) * 1.35)
    fig.tight_layout()
    fig.savefig(DOCS / "latency.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
