"""Build the in-browser demo's model files from the trained memory banks (run after run.py).

Writes to ../cv-data/mvtec/web/ (never committed):
- models/backbone.onnx         image (1,3,224,224) -> patch features (784, 1536)       [loaded once]
- models/knn_<category>.onnx   features -> anomaly map (224,224) + image score         [memory bank baked in]
- models/thresholds.json       per-category threshold that maximizes F1 on the MVTec test set
- examples/*.png               one good and one defective test image per category

Large weights are stored as float16 and cast back to float32 inside the graph: half the download, float32 math.
Every export is checked against the PyTorch model on the example images; validate.py checks the full test set.
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnx import helper, numpy_helper
from sklearn.metrics import precision_recall_curve
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1]))
from common.paths import DATA_ROOT  # noqa: E402
from patchcore import PatchCore, gaussian_blur  # noqa: E402
from run import SIZE, MVTecSplit  # noqa: E402

MVTEC = DATA_ROOT / "mvtec"
BANKS = MVTEC / "outputs"
OUT = MVTEC / "web"
CATS = ["bottle", "metal_nut", "grid"]


class Backbone(nn.Module):
    """PatchCore._features as a module, for export."""

    def __init__(self, b):
        super().__init__()
        self.b = b

    def forward(self, x):
        b = self.b
        x = b.maxpool(b.relu(b.bn1(b.conv1(x))))
        f2 = b.layer2(b.layer1(x))
        f3 = b.layer3(f2)
        f2 = F.avg_pool2d(f2, 3, 1, 1)
        f3 = F.avg_pool2d(f3, 3, 1, 1)
        f3 = F.interpolate(f3, size=(28, 28), mode="bilinear", align_corners=False)
        feats = torch.cat([f2, f3], dim=1)
        return feats.permute(0, 2, 3, 1).reshape(-1, feats.shape[1])  # (784, 1536)


class Knn(nn.Module):
    """Nearest-neighbor distance to the memory bank, via |q|^2 + |b|^2 - 2 q.b (same as cdist)."""

    def __init__(self, bank_fp16):
        super().__init__()
        self.register_buffer("bank", bank_fp16)
        self.register_buffer("bank_sq", (bank_fp16.float() ** 2).sum(1))

    def forward(self, q):
        bank = self.bank.float()
        d2 = (q * q).sum(1, keepdim=True) + self.bank_sq[None] - 2 * q @ bank.T
        dists = d2.min(dim=1).values.clamp_min(0).sqrt()
        maps = dists.reshape(1, 1, 28, 28)
        score = maps.max()
        maps = F.interpolate(maps, size=(SIZE, SIZE), mode="bilinear", align_corners=False)
        return gaussian_blur(maps, sigma=4).reshape(SIZE, SIZE), score


def store_fp16(path):
    """Keep large float32 weights as float16 in the file; one Cast node each restores float32 at load time."""
    m = onnx.load(str(path))
    g = m.graph
    casts = []
    for init in list(g.initializer):
        if init.data_type != onnx.TensorProto.FLOAT or np.prod(init.dims) < 1024:
            continue
        arr = numpy_helper.to_array(init).astype(np.float16)
        g.initializer.remove(init)
        g.initializer.append(numpy_helper.from_array(arr, init.name + "_fp16"))
        casts.append(helper.make_node("Cast", [init.name + "_fp16"], [init.name], to=onnx.TensorProto.FLOAT))
    nodes = casts + list(g.node)
    del g.node[:]
    g.node.extend(nodes)
    onnx.save(m, str(path))


def export(module, args, path, names_in, names_out):
    torch.onnx.export(
        module, args, str(path), input_names=names_in, output_names=names_out, opset_version=17, dynamo=False
    )
    store_fp16(path)
    print(f"{path.name}: {path.stat().st_size / 1e6:.1f} MB", flush=True)


def best_f1_threshold(model, cat):
    scores, labels = [], []
    for x, y, _ in DataLoader(MVTecSplit(cat, "test"), batch_size=8):
        s, _ = model.predict(x, SIZE)
        scores += s.tolist()
        labels += y.tolist()
    p, r, t = precision_recall_curve(labels, scores)
    f1 = 2 * p * r / np.maximum(p + r, 1e-9)
    k = int(np.argmax(f1[:-1]))
    return {
        "threshold": round(float(t[k]), 4),
        "f1": round(float(f1[k]), 4),
        "precision": round(float(p[k]), 4),
        "recall": round(float(r[k]), 4),
    }


def main():
    (OUT / "models").mkdir(parents=True, exist_ok=True)
    (OUT / "examples").mkdir(exist_ok=True)
    model = PatchCore()
    export(
        Backbone(model.backbone).eval(),
        (torch.zeros(1, 3, SIZE, SIZE),),
        OUT / "models" / "backbone.onnx",
        ["image"],
        ["features"],
    )
    sess_bb = ort.InferenceSession(str(OUT / "models" / "backbone.onnx"))

    thresholds = {}
    for cat in CATS:
        bank = torch.load(BANKS / f"{cat}_memory_bank.pt", map_location="cpu")["memory_bank"].half()
        model.memory_bank = bank.float()  # score with exactly what the demo ships
        thresholds[cat] = best_f1_threshold(model, cat)
        print(cat, thresholds[cat], flush=True)
        export(
            Knn(bank).eval(),
            (torch.zeros(784, 1536),),
            OUT / "models" / f"knn_{cat}.onnx",
            ["features"],
            ["map", "score"],
        )
        sess = ort.InferenceSession(str(OUT / "models" / f"knn_{cat}.onnx"))

        good = sorted((MVTEC / cat / "test" / "good").glob("*.png"))[0]
        bad_dir = sorted(d for d in (MVTEC / cat / "test").iterdir() if d.is_dir() and d.name != "good")[0]
        bad = sorted(bad_dir.glob("*.png"))[0]
        ds = MVTecSplit(cat, "test")
        paths = [item[0] for item in ds.items]
        for src, name in [(good, f"{cat}_good"), (bad, f"{cat}_{bad_dir.name}")]:
            shutil.copy(src, OUT / "examples" / f"{name}.png")
            x = ds[paths.index(src)][0][None]
            s_ref, m_ref = model.predict(x, SIZE)
            m, s = sess.run(None, {"features": sess_bb.run(None, {"image": x.numpy()})[0]})
            print(
                f"  {name:24s} torch {float(s_ref[0]):.4f}  onnx {float(s):.4f}  "
                f"map max|diff| {np.abs(m - m_ref[0].numpy()).max():.4f}",
                flush=True,
            )
    (OUT / "models" / "thresholds.json").write_text(json.dumps(thresholds, indent=1))


if __name__ == "__main__":
    main()
