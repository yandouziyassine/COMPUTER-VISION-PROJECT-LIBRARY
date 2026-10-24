"""Reference pre/post-processing for the PPE hard-hat YOLO11n ONNX model.

This is the ground truth the browser (JavaScript, onnxruntime-web) port is
checked against. It reproduces Ultralytics 8.4.175 ``predict`` behaviour for a
static 640x640 export with NO built-in NMS.

Model I/O
---------
input  ``images``  float32 [1, 3, 640, 640]  (NCHW, RGB, values in [0, 1])
output ``output0`` float32 [1, 6, 8400]

Output layout (channel-major: row ``c`` of 8400 anchors)::

    out[0, 0, i] = cx  box centre x  (pixels in the 640x640 letterboxed input)
    out[0, 1, i] = cy  box centre y
    out[0, 2, i] = w   box width
    out[0, 3, i] = h   box height
    out[0, 4, i] = score for class 0 "hardhat"     (already sigmoid, 0..1)
    out[0, 5, i] = score for class 1 "no-hardhat"  (already sigmoid, 0..1)

There is no objectness channel. In a flat JS ``Float32Array`` the value for
channel ``c`` and anchor ``i`` is at ``data[c * 8400 + i]``.
The 8400 anchors are the 80x80 + 40x40 + 20x20 grids (strides 8/16/32); the
decode below does not need to know that, boxes are already in pixels.

Preprocessing (letterbox, identical to Ultralytics ``LetterBox(auto=False)``)
---------------------------------------------------------------------------
1. Decode image to RGB (not BGR), shape (H, W).
2. ``r = min(640 / H, 640 / W)``; ``new_w = round(W * r)``, ``new_h = round(H * r)``.
3. Resize to (new_w, new_h) with bilinear interpolation (skip if same size).
4. ``dw = (640 - new_w) / 2``, ``dh = (640 - new_h) / 2``;
   ``left = round(dw - 0.1)``, ``top = round(dh - 0.1)``.
   Paste into a 640x640 canvas filled with grey (114, 114, 114) at (left, top).
5. Divide by 255, convert HWC -> CHW, add batch dim.

Postprocessing
--------------
1. For each anchor: ``cls = argmax(s0, s1)``, ``score = max(s0, s1)``.
   Keep if ``score > conf`` (conf = 0.25 for site alerts).
2. xywh -> xyxy: ``x1 = cx - w/2`` ... ``y2 = cy + h/2``.
3. Class-aware NMS, IoU threshold 0.45, sort by score desc, max 300 boxes.
   (Ultralytics implements class-aware NMS by offsetting boxes per class; the
   per-class NMS here is equivalent.)
4. Undo letterbox: ``x = (x - left) / r``, ``y = (y - top) / r``, then clip to
   [0, W] and [0, H].

Usage::

    python ppe_onnx_reference.py ppe_yolo11n.onnx image.jpg [more.jpg ...]
"""

from __future__ import annotations

import json
import sys

import numpy as np

INPUT_SIZE = 640
CLASS_NAMES = ["hardhat", "no-hardhat"]
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
MAX_DET = 300
PAD_VALUE = 114


def letterbox(rgb: np.ndarray, size: int = INPUT_SIZE):
    """Return (tensor[1,3,size,size] float32, meta dict) for an RGB uint8 HxWx3 image."""
    import cv2

    h, w = rgb.shape[:2]
    r = min(size / h, size / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    if (new_w, new_h) != (w, h):
        rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dw, dh = (size - new_w) / 2, (size - new_h) / 2
    left, top = int(round(dw - 0.1)), int(round(dh - 0.1))
    canvas = np.full((size, size, 3), PAD_VALUE, dtype=np.uint8)
    canvas[top : top + new_h, left : left + new_w] = rgb
    x = canvas.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    meta = {"orig_w": w, "orig_h": h, "ratio": r, "pad_left": left, "pad_top": top}
    return np.ascontiguousarray(x), meta


def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area + areas - inter + 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    """Greedy NMS; returns kept indices sorted by descending score."""
    order = np.argsort(-scores, kind="stable")
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        order = rest[_iou(boxes[i], boxes[rest]) <= iou_thr]
    return keep


def decode(
    output: np.ndarray,
    meta: dict,
    conf: float = CONF_THRESHOLD,
    iou: float = IOU_THRESHOLD,
    max_det: int = MAX_DET,
) -> list[dict]:
    """Decode raw (1, 6, 8400) output into detections in original-image pixels."""
    pred = output[0].T  # (8400, 6)
    cls_scores = pred[:, 4:]
    cls = cls_scores.argmax(1)
    score = cls_scores.max(1)
    m = score > conf
    xywh, cls, score = pred[m, :4], cls[m], score[m]
    boxes = np.stack(
        [
            xywh[:, 0] - xywh[:, 2] / 2,
            xywh[:, 1] - xywh[:, 3] / 2,
            xywh[:, 0] + xywh[:, 2] / 2,
            xywh[:, 1] + xywh[:, 3] / 2,
        ],
        1,
    )
    keep: list[int] = []
    for c in np.unique(cls):
        idx = np.nonzero(cls == c)[0]
        keep += [int(idx[k]) for k in nms(boxes[idx], score[idx], iou)]
    keep = sorted(keep, key=lambda k: -score[k])[:max_det]

    r, left, top = meta["ratio"], meta["pad_left"], meta["pad_top"]
    dets = []
    for k in keep:
        x1, y1, x2, y2 = boxes[k]
        x1 = min(max((x1 - left) / r, 0), meta["orig_w"])
        x2 = min(max((x2 - left) / r, 0), meta["orig_w"])
        y1 = min(max((y1 - top) / r, 0), meta["orig_h"])
        y2 = min(max((y2 - top) / r, 0), meta["orig_h"])
        dets.append(
            {
                "class_id": int(cls[k]),
                "label": CLASS_NAMES[int(cls[k])],
                "score": round(float(score[k]), 5),
                "box_xyxy": [round(float(v), 2) for v in (x1, y1, x2, y2)],
            }
        )
    return dets


def run(model_path: str, image_path: str, conf: float = CONF_THRESHOLD) -> list[dict]:
    """Load image (as RGB), run the ONNX model with onnxruntime, decode."""
    import onnxruntime as ort
    from PIL import Image

    rgb = np.asarray(Image.open(image_path).convert("RGB"))
    x, meta = letterbox(rgb)
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    return decode(out, meta, conf=conf)


if __name__ == "__main__":
    model = sys.argv[1]
    for path in sys.argv[2:]:
        print(json.dumps({"image": path, "detections": run(model, path)}, indent=2))
