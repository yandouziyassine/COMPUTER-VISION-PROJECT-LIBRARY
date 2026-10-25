"""Push one v2 inference job to Kaggle (T4, internet on, private). Usage:
    python kaggle_push_v2.py --name depth-v2-nyu --datasets nyu ibims --commit <sha> [--limit 3]
Results: kaggle kernels output <user>/<name> -p <folder>."""

import argparse
import json
import subprocess
from pathlib import Path

import models_v2 as M

KAGGLE = r"C:/Users/yando/.kaggle-cli/Scripts/kaggle.exe"
USER = "yassineyandouzi"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", default=list(M.MODELS))
    ap.add_argument("--commit", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", default="../../work/kaggle14")
    a = ap.parse_args()
    job = {"commit": a.commit, "datasets": a.datasets, "models": a.models, "limit": a.limit}
    folder = Path(a.out) / a.name
    folder.mkdir(parents=True, exist_ok=True)
    src = Path(__file__).with_name("kaggle_infer_v2.py").read_text(encoding="utf-8")
    (folder / "run.py").write_text(f"JOB = {json.dumps(job)}\n" + src, encoding="utf-8")
    meta = {
        "id": f"{USER}/{a.name}",
        "title": a.name,
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    subprocess.run([KAGGLE, "kernels", "push", "-p", str(folder)], check=True)


if __name__ == "__main__":
    main()
