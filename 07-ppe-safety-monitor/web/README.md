# Hard-hat detector: in-browser (ONNX) package

This folder has what you need to run the project 07 model (YOLO11n, 2 classes) in the browser with
onnxruntime-web. Decode and NMS are left to JavaScript.

> In this repo, only the reference script and this README are committed. The `.onnx` files, samples and golden JSON are not in git (no model weights in the repo); to rebuild them, export `best.pt` as described below.

## Files

| File | What |
|---|---|
| `ppe_yolo11n.onnx` | FP32, static input 1x3x640x640, opset 17, simplified (onnxslim), **no NMS**. 10.7 MB (10,664,397 B) |
| `ppe_yolo11n_fp16.onnx` | FP16 weights, **float32 input/output** (casts inside the graph). 5.4 MB (5,365,843 B) |
| `labels.json` | `{"0": "hardhat", "1": "no-hardhat"}` |
| `ppe_onnx_reference.py` | Python reference letterbox + decode + NMS. The JS port is checked against it |
| `samples/*.jpg` | 5 real images: 4 from the Hard Hat Workers **test** split (held out), plus one 960x540 crop of a test image to exercise letterbox padding |
| `expected_detections.json` | Golden vectors: letterbox params, input-tensor checksums, one raw output column, and final detections per sample (FP32 model) |

Source: `best.pt` from the Kaggle kernel `yassineyandouzi/cvlib-07-ppe-full` (Ultralytics 8.4.175, trained at imgsz 640
on the full train split). It was exported with `YOLO(best.pt).export(format="onnx", imgsz=640, opset=17, simplify=True, nms=False, dynamic=False)`.
The FP16 file was made with `onnxconverter_common.float16.convert_float_to_float16(keep_io_types=True)`, after
clearing the stale `graph.value_info` (without that step onnxruntime rejects the Resize casts).

## Input

- Name `images`, float32 `[1, 3, 640, 640]`, NCHW, **RGB** order, values `pixel / 255`.
- Letterbox (matches Ultralytics `LetterBox(auto=False)`):
  1. `r = min(640 / H, 640 / W)`, `newW = round(W * r)`, `newH = round(H * r)`
  2. resize bilinearly to `newW x newH`
  3. `left = round((640 - newW) / 2 - 0.1)`, `top = round((640 - newH) / 2 - 0.1)`
  4. paste onto a 640x640 canvas filled with `rgb(114,114,114)`

  In JS: `ctx.fillStyle = "rgb(114,114,114)"; ctx.fillRect(0,0,640,640); ctx.drawImage(img, left, top, newW, newH)`,
  then `getImageData` and write `R/255` into plane 0, `G/255` into plane 1 and `B/255` into plane 2. Ignore alpha.
  Use JS `Math.round` and note it differs from Python's banker's rounding at exact .5. That only matters when
  `(640 - newW)/2 - 0.1` lands on .5, which can't happen because of the -0.1 offset.

## Output

`output0`, float32 `[1, 6, 8400]`, channel-major. For anchor `i` (`N = 8400`):

```
cx = d[0*N+i]  cy = d[1*N+i]  w = d[2*N+i]  h = d[3*N+i]   // pixels in the 640x640 letterboxed image
s0 = d[4*N+i]  // hardhat score, already sigmoid
s1 = d[5*N+i]  // no-hardhat score, already sigmoid
```

There is no objectness channel. `class = argmax(s0, s1)`, `score = max(s0, s1)`.

## Postprocessing

1. Keep anchors with `score > conf` (**conf 0.25** is the site-alert operating point).
2. `x1 = cx - w/2, y1 = cy - h/2, x2 = cx + w/2, y2 = cy + h/2`.
3. Greedy **class-aware** NMS, **IoU 0.45**: sort by score descending, then drop any box whose IoU with an
   already-kept box of the same class is above 0.45. Keep at most 300 boxes.
4. Map back to the original image: `x = (x - left) / r`, `y = (y - top) / r`, then clip to `[0, W] x [0, H]`.
5. Site alert: the image is flagged when it has at least one `no-hardhat` detection at conf >= 0.25.

## Measured accuracy (source: `07-ppe-safety-monitor/docs/results_full.json` in the library repo)

These numbers were measured on the Hard Hat Workers test split (2,001 images, 413 with a violation), using the same weights through Ultralytics:

- mAP50 **0.939**, mAP50-95 0.595 (hardhat AP50 0.941, no-hardhat AP50 0.938)
- Site alerts at conf 0.25: recall **97.6 %** (403/413 violations caught), precision **96.2 %** (16 false alarms)

## Numerical parity (this export, onnxruntime 1.31 CPU vs PyTorch 2.14 CPU)

| Check, over the 5 samples | Result |
|---|---|
| Raw output, ONNX FP32 vs PyTorch on the same tensor | max abs diff 0.0013 (on box pixels; scores agree far more closely) |
| Decoded detections, ONNX FP32 vs PyTorch, same tensor | identical counts and classes, box diff <= 0.01 px, score diff 0 (to 5 dp) |
| ONNX FP32 + reference decode vs `YOLO(best.pt).predict(conf=.25, iou=.45)`, 640x640 images | identical (4/7/1/22 boxes), box diff <= 0.005 px, score diff <= 5e-6 |
| Same, 960x540 sample | same 3 boxes and classes, box diff <= 2.8 px, score diff <= 0.043. This is expected: Ultralytics' `.pt` predict uses a minimal-rectangle letterbox (640x384 input), while the static ONNX always gets 640x640 |
| FP16 vs FP32 ONNX | identical counts and classes, box diff <= 0.22 px, score diff <= 0.001 (raw score diff <= 0.009) |

The FP16 file is half the size and matches FP32 to within display precision, so it is the better download
for the web. The FP32 file is the reference for exact golden comparisons.
For a JS port, bilinear resampling in canvas is not identical to OpenCV `INTER_LINEAR`. On resized images,
expect small differences (around 1 px and around 0.01 in score). On the 640x640 samples (no resize) the
JS result should match `expected_detections.json` almost exactly.
