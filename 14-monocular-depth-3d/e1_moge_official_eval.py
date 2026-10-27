"""E1 · Credibility check for the failed reproduction gate: run MoGe's *official* evaluation code
(microsoft/MoGe, moge/scripts/eval_baseline.py) on its own processed NYUv2 and KITTI benchmarks
(Hugging Face Ruicheng/monocular-geometry-evaluation) for MoGe-2 and Depth Pro, metric depth without GT intrinsics.
If these reproduce the MoGe-2 paper (NYU AbsRel 0.0733 / 0.107) the gap in our gate is a protocol difference
(data processing), not a broken model wrapper. Kaggle T4.

Usage (Kaggle): python kaggle_push_v2.py --name depth-e1-moge-eval --script e1_moge_official_eval.py --commit <sha>
"""

import json
import os
import subprocess
import sys
from pathlib import Path

JOB = globals().get("JOB") or json.loads(os.environ.get("JOB", "{}"))
MOGE_COMMIT = "74fbce054ebed49800de42d0ad0e83495065719a"
BENCH = ["NYUv2", "KITTI"]

DEPTH_PRO_BASELINE = """
from typing import *
import click
import torch
import torch.nn.functional as F
from moge.test.baseline import MGEBaselineInterface


class Baseline(MGEBaselineInterface):
    def __init__(self, device):
        from transformers import DepthProForDepthEstimation, DepthProImageProcessorFast
        self.proc = DepthProImageProcessorFast.from_pretrained("apple/DepthPro-hf")
        self.net = DepthProForDepthEstimation.from_pretrained("apple/DepthPro-hf", torch_dtype=torch.float16).to(device).eval()
        self.device = device

    @click.command()
    @click.option("--device", type=str, default="cuda")
    @staticmethod
    def load(device):
        return Baseline(device)

    @torch.inference_mode()
    def infer(self, image, intrinsics=None):
        h, w = image.shape[-2:]
        img = (image.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
        inputs = self.proc(images=img, return_tensors="pt").to(self.device, torch.float16)
        post = self.proc.post_process_depth_estimation(self.net(**inputs), target_sizes=[(h, w)])[0]
        depth = post["predicted_depth"].float()
        if intrinsics is not None:  # oracle mode: rescale by the true focal (Depth Pro depth is proportional to f)
            fx = float(intrinsics[0, 0]) * w
            depth = depth * (fx / float(post["focal_length"]))
        return {"depth_metric": depth.to(self.device)}
"""


def sh(*cmd, cwd=None):
    subprocess.run(list(cmd), check=True, cwd=cwd)


def main():
    work = Path("/tmp/moge")
    if not work.exists():
        sh("git", "clone", "-q", "https://github.com/microsoft/MoGe.git", str(work))
    sh("git", "checkout", "-q", MOGE_COMMIT, cwd=work)
    sh(sys.executable, "-m", "pip", "install", "-q", "-e", ".", "transformers==4.53.0", "huggingface_hub", cwd=work)
    data = work / "data" / "eval"
    data.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download

    for b in BENCH:
        if not (data / b).exists():
            z = hf_hub_download(
                "Ruicheng/monocular-geometry-evaluation", f"{b}.zip", repo_type="dataset", local_dir=str(data)
            )
            sh("unzip", "-q", "-o", z, "-d", str(data))
    full = json.loads((work / "configs" / "eval" / "moge2.json").read_text())
    cfg = {k: v for k, v in full.items() if k in BENCH}
    (work / "eval_cfg.json").write_text(json.dumps(cfg))
    (work / "baselines" / "depth_pro_hf.py").write_text(DEPTH_PRO_BASELINE)
    out = Path("/kaggle/working")
    runs = {
        "moge2-vitl": ["--baseline", "baselines/moge.py", "--version", "v2", "--pretrained", "Ruicheng/moge-2-vitl"],
        "moge2-vitl-normal": [
            "--baseline",
            "baselines/moge.py",
            "--version",
            "v2",
            "--pretrained",
            "Ruicheng/moge-2-vitl-normal",
        ],
        "depth-pro": ["--baseline", "baselines/depth_pro_hf.py"],
    }
    summary = {"moge_commit": MOGE_COMMIT, "config": cfg, "runs": {}}
    for name, args in runs.items():
        for oracle in (False, True):
            tag = f"{name}{'-gt-intrinsics' if oracle else ''}"
            o = out / f"e1_{tag}.json"
            cmd = [
                sys.executable,
                "moge/scripts/eval_baseline.py",
                "--config",
                "eval_cfg.json",
                "--output",
                str(o),
                "--mg",
                "metric",
                *args,
            ]
            if oracle:
                cmd.append("--oracle")
            try:
                sh(*cmd, cwd=work)
                summary["runs"][tag] = json.loads(o.read_text())
            except subprocess.CalledProcessError as e:
                summary["runs"][tag] = f"failed: {e}"
            (out / "e1_summary.json").write_text(json.dumps(summary, indent=1))
            print(tag, "done", flush=True)


if __name__ == "__main__":
    main()
