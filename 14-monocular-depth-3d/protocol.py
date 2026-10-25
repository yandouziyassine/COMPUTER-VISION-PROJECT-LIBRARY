"""v2 evaluation protocol (configs/protocol.yaml): masks, alignment per model kind, metrics, bootstrap CIs,
prediction-cache manifest. Pure NumPy + standard library, so the same code runs on Kaggle and in CI."""

import hashlib
import json
import re
from pathlib import Path

import numpy as np

CONFIG = Path(__file__).resolve().parent / "configs" / "protocol.yaml"


# ---------- config ----------


def _scalar(s):
    s = s.strip()
    if s in ("null", "~", ""):
        return None
    if s in ("true", "false"):
        return s == "true"
    if s.startswith("[") and s.endswith("]"):
        return [_scalar(x) for x in s[1:-1].split(",") if x.strip()]
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s.strip("'\"")


def parse_yaml(text):
    """Minimal block-YAML reader (nested mappings, scalars, [flow, lists], # comments): enough for
    protocol.yaml and keeps PyYAML out of the CI dependencies."""
    root, stack = {}, [(-1, None)]
    stack[0] = (-1, root)
    for raw in text.splitlines():
        line = re.sub(r"\s+#.*$", "", raw) if not raw.lstrip().startswith("#") else ""
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        key, _, value = line.strip().partition(":")
        while stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]
        if value.strip():
            parent[key] = _scalar(value)
        else:
            parent[key] = {}
            stack.append((indent, parent[key]))
    return root


def load_protocol(path=CONFIG):
    return parse_yaml(Path(path).read_text(encoding="utf-8"))


# ---------- masks ----------


def crop_mask(shape, crop):
    """Evaluation region for a dataset crop name (on the GT resolution)."""
    h, w = shape
    m = np.zeros(shape, bool)
    if crop in (None, "none"):
        m[:] = True
    elif crop == "eigen_nyu":
        assert shape == (480, 640), shape
        m[45:471, 41:601] = True
    elif crop == "eigen_kitti":
        m[int(0.3324324 * h) : int(0.91351351 * h), int(0.0359477 * w) : int(0.96405229 * w)] = True
    elif crop == "garg_kitti":
        m[int(0.40810811 * h) : int(0.99189189 * h), int(0.03594771 * w) : int(0.96405229 * w)] = True
    else:
        raise ValueError(f"unknown crop {crop!r}")
    return m


def kb_crop(a):
    """KITTI benchmark crop (352 x 1216, bottom-centred), for images (H, W, C) or depth (H, W)."""
    h, w = a.shape[:2]
    top, left = h - 352, (w - 1216) // 2
    return a[top : top + 352, left : left + 1216]


def valid_mask(gt, ds, extra=None):
    """Pixels scored for one image: finite GT inside (min_depth, max_depth), inside the crop."""
    lo, hi = ds["min_depth"], ds["max_depth"]
    m = np.isfinite(gt) & (gt > lo)
    if hi is not None:
        m &= gt < hi
    m &= crop_mask(gt.shape, ds.get("crop"))
    if extra is not None:
        m &= extra
    return m


# ---------- alignment ----------


def _lsq(x, y):
    A = np.stack([x, np.ones_like(x)], axis=1)
    (s, t), *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(s), float(t)


def align(pred, gt, mask, method, max_depth=None, min_depth=None, clip=True):
    """Turn a raw prediction into depth (meters, same scale as gt) following one protocol method."""
    p = pred.astype(np.float64)
    if method == "none":
        out = p
    elif method == "median":
        out = p * (np.median(gt[mask]) / np.median(p[mask]))
    elif method == "lsq_depth":
        s, t = _lsq(p[mask], gt[mask].astype(np.float64))
        out = s * p + t
    elif method == "lsq_disparity":
        s, t = _lsq(p[mask], 1.0 / gt[mask].astype(np.float64))
        disp = s * p + t
        floor = 1.0 / max_depth if max_depth else 1e-3
        out = 1.0 / np.clip(disp, floor, None)
    else:
        raise ValueError(f"unknown alignment {method!r}")
    if clip:
        out = np.clip(out, min_depth if min_depth else 1e-6, max_depth if max_depth else None)
    return out


# ---------- metrics ----------


def image_metrics(depth, gt, mask):
    """AbsRel and delta1 on one image (pixels averaged within the image)."""
    d, g = np.maximum(depth[mask], 1e-6), gt[mask]
    ratio = np.maximum(d / g, g / d)
    return {"absrel": float(np.mean(np.abs(d - g) / g)), "delta1": float(np.mean(ratio < 1.25))}


def score_image(pred, gt, ds, kind, protocol, extra_mask=None):
    """All alignments listed for this model kind -> {alignment: {metric: value}}. None if no valid pixel."""
    mask = valid_mask(gt, ds, extra_mask)
    if mask.sum() < 10:
        return None
    out = {}
    for method in protocol["alignment"][kind]:
        d = align(pred, gt, mask, method, ds["max_depth"], ds["min_depth"], protocol.get("clip_after_alignment", True))
        out[method] = image_metrics(d, gt, mask)
    return out


def bootstrap_ci(values, resamples=1000, confidence=0.95, seed=0):
    """Mean over images and a percentile bootstrap CI (images resampled with replacement)."""
    v = np.asarray(values, np.float64)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (resamples, len(v)))].mean(axis=1)
    a = (1 - confidence) / 2
    return float(v.mean()), float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def summarize(per_image, protocol):
    """per_image: list of score_image outputs -> {alignment: {metric: [mean, lo, hi]}, n}."""
    b = protocol["bootstrap"]
    rows = [r for r in per_image if r is not None]
    out = {"n_images": len(rows)}
    for method in rows[0]:
        out[method] = {
            m: [
                round(x, 5)
                for x in bootstrap_ci([r[method][m] for r in rows], b["resamples"], b["confidence"], b["seed"])
            ]
            for m in rows[0][method]
        }
    return out


# ---------- prediction cache manifest ----------


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def update_manifest(manifest_path, cache_file, model, dataset, revision, protocol_version, n_images):
    """Record one cache file (hash, model revision, protocol version) in the JSON manifest."""
    manifest_path = Path(manifest_path)
    man = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    man[Path(cache_file).name] = {
        "model": model,
        "dataset": dataset,
        "revision": revision,
        "protocol_version": protocol_version,
        "n_images": n_images,
        "bytes": Path(cache_file).stat().st_size,
        "sha256": sha256(cache_file),
    }
    manifest_path.write_text(json.dumps(man, indent=1, sort_keys=True))
    return man


def verify_manifest(manifest_path, folder):
    """Names of cache files whose hash no longer matches the manifest (empty list = all good)."""
    man = json.loads(Path(manifest_path).read_text())
    return [n for n, e in man.items() if not (Path(folder) / n).exists() or sha256(Path(folder) / n) != e["sha256"]]
