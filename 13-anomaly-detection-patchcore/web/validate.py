"""Full MVTec test set through the exported ONNX models (run after build.py): image AUROC and F1 at the shipped
threshold, to confirm the browser demo scores exactly like the PyTorch model."""

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1]))
from common.paths import DATA_ROOT  # noqa: E402
from run import MVTecSplit  # noqa: E402

MODELS = DATA_ROOT / "mvtec" / "web" / "models"
THR = json.loads((MODELS / "thresholds.json").read_text())
opts = ort.SessionOptions()
opts.intra_op_num_threads = 4
bb = ort.InferenceSession(str(MODELS / "backbone.onnx"), opts)
results = {}
for cat in THR:
    knn = ort.InferenceSession(str(MODELS / f"knn_{cat}.onnx"), opts)
    scores, labels = [], []
    for x, y, _ in MVTecSplit(cat, "test"):
        feats = bb.run(None, {"image": x.unsqueeze(0).numpy()})[0]
        scores.append(float(knn.run(None, {"features": feats})[1]))
        labels.append(int(y))
    s, y = np.array(scores), np.array(labels)
    pred = s > THR[cat]["threshold"]
    f1 = 2 * int((pred & (y == 1)).sum()) / max(int(pred.sum() + y.sum()), 1)
    results[cat] = {"n": len(y), "auroc": round(float(roc_auc_score(y, s)), 4), "f1_at_threshold": round(f1, 4),
                    "f1_pytorch": THR[cat]["f1"]}
    print(cat, results[cat], flush=True)
(HERE.parent / "docs" / "onnx_validation.json").write_text(json.dumps(results, indent=1))
