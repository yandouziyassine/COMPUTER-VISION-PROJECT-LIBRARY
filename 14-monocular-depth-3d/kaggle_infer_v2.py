"""Kaggle T4 job for v2 (W2/W3): run the models on one or more datasets, cache float16 predictions, score
every image with protocol.py, write a manifest. Pushed by kaggle_push_v2.py, which prepends a JOB = {...} line.

Outputs (/kaggle/working): preds/<model>__<dataset>__<chunk>.npz (float16, keyed by image id),
scores/<model>__<dataset>.json (per-image metrics for every alignment + seconds per image),
preds_manifest.json (sha256 per cache file, model revision, protocol version)."""

import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

JOB = globals().get("JOB") or json.loads(os.environ.get("JOB", "{}"))
REPO = "https://github.com/yandouziyassine/COMPUTER-VISION-PROJECT-LIBRARY.git"
PIP = [
    "transformers==4.53.0",
    "diffusers==0.34.0",
    "accelerate",
    "git+https://github.com/microsoft/MoGe.git@74fbce054ebed49800de42d0ad0e83495065719a",
]
WORK, TMP = Path("/kaggle/working"), Path("/tmp/v2")
CHUNK = 100


def sh(*cmd):
    subprocess.run(list(cmd), check=True)


def setup():
    sh(sys.executable, "-m", "pip", "install", "-q", *PIP)
    src = TMP / "repo"
    if not src.exists():
        sh("git", "clone", "-q", REPO, str(src))
    sh("git", "-C", str(src), "checkout", "-q", JOB["commit"])
    os.environ["CV_DATA"] = str(TMP / "cv-data")
    proj = src / "14-monocular-depth-3d"
    sys.path[:0] = [str(proj), str(src)]
    return proj


def stage(name, ds_cfg):
    """Decode a dataset once into per-image .npz files so each model can re-read it quickly."""
    import numpy as np

    import datasets_v2 as D

    out = TMP / "stage" / name
    done = out / "_done"
    if done.exists():
        return sorted(out.glob("*.npz"))
    out.mkdir(parents=True, exist_ok=True)
    root = Path(os.environ["CV_DATA"]) / "depth" / name
    if name == "nyu":
        root.mkdir(parents=True, exist_ok=True)
        for url, fn in [(D.NYU_URL, "nyu_depth_v2_labeled.mat"), (D.NYU_SPLIT_URL, "splits.mat")]:
            if not (root / fn).exists():
                sh("curl", "-sL", "-o", str(root / fn), url)
    t0, n = time.time(), 0
    for k, (img_id, rgb, gt, extra) in enumerate(D.LOADERS[name](ds_cfg, root)):
        kw = {} if extra is None else {"extra": extra}
        np.savez(out / f"{k:05d}.npz", id=img_id, rgb=rgb, gt=gt.astype(np.float32), **kw)
        n += 1
    done.write_text(str(n))
    print(f"staged {name}: {n} images in {time.time() - t0:.0f}s", flush=True)
    return sorted(out.glob("*.npz"))


def run(model, name, files, proto):
    import numpy as np
    import torch

    import models_v2 as M
    import protocol as P

    ds_cfg = proto["datasets"][name]
    predict = M.load(model, "cuda")
    scores, chunk, seconds = {}, {}, []
    (WORK / "preds").mkdir(exist_ok=True, parents=True)
    (WORK / "scores").mkdir(exist_ok=True)

    def flush(idx):
        if chunk:
            f = WORK / "preds" / f"{model}__{name}__{idx:03d}.npz"
            np.savez_compressed(f, **chunk)
            P.update_manifest(
                WORK / "preds_manifest.json", f, model, name, M.MODELS[model][2], proto["version"], len(chunk)
            )
            chunk.clear()

    for k, f in enumerate(files):
        s = np.load(f)
        img_id, rgb, gt = str(s["id"]), s["rgb"], s["gt"]
        torch.cuda.synchronize()
        t0 = time.time()
        pred = predict(rgb)
        torch.cuda.synchronize()
        seconds.append(time.time() - t0)
        pred = np.asarray(pred, np.float32)
        assert pred.shape == gt.shape, (pred.shape, gt.shape)
        chunk[img_id] = pred.astype(np.float16)
        scores[img_id] = P.score_image(pred, gt, ds_cfg, M.kind(model), proto, s["extra"] if "extra" in s else None)
        if (k + 1) % CHUNK == 0:
            flush(k // CHUNK)
    flush(len(files) // CHUNK)
    res = {
        "model": model,
        "dataset": name,
        "revision": M.MODELS[model][2],
        "commit": JOB["commit"],
        "seconds_per_image_t4": round(float(np.median(seconds[1:] or seconds)), 3),
        "per_image": scores,
        "summary": P.summarize(list(scores.values()), proto),
    }
    (WORK / "scores" / f"{model}__{name}.json").write_text(json.dumps(res))
    print(model, name, json.dumps(res["summary"]), flush=True)
    del predict
    gc.collect()
    torch.cuda.empty_cache()


def main():
    setup()
    import protocol as P

    proto = P.load_protocol()
    for name in JOB["datasets"]:
        files = stage(name, proto["datasets"][name])[: JOB.get("limit")]  # limit: smoke tests only
        for model in JOB["models"]:
            if (WORK / "scores" / f"{model}__{name}.json").exists():
                continue
            try:
                run(model, name, files, proto)
            except Exception as e:  # one broken model must not lose the others' results
                print(f"FAILED {model} {name}: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
