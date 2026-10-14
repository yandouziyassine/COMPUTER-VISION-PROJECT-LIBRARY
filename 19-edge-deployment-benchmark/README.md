# 19 · Edge Deployment Benchmark: CPU vs iGPU vs NPU

How fast can one object detector run on a modern laptop *without* an NVIDIA GPU? This project
exports **YOLO11n** to ONNX and OpenVINO (FP32 / FP16 / INT8), then benchmarks it on all three
compute units of an Intel Core Ultra 7 258V (CPU, Arc 140V integrated GPU, AI Boost NPU). It also
checks that INT8 quantization doesn't hurt accuracy.

![latency chart](docs/latency.png)

*Chart: the latest run (CPU busy with other apps, see below).*

## Results

YOLO11n, 640×640, batch 1, model inference only (100 timed runs after 15 warm-up runs), OpenVINO 2026.4.

| Runtime | Precision | Device | Idle CPU (2026-10-12) | CPU busy with other apps (2026-10-13) |
|---|---|---|---|---|
| PyTorch (eager) | FP32 | CPU | 79.0 ms (1.0×) | 141.2 ms (1.0×) |
| ONNX Runtime | FP32 | CPU | 42.7 ms (1.9×) | 61.2 ms (2.3×) |
| OpenVINO | FP32 | CPU | 34.6 ms (2.3×) | 47.2 ms (3.0×) |
| OpenVINO | INT8 | CPU | 16.7 ms (4.7×) | 21.5 ms (6.6×) |
| OpenVINO | FP16 | **iGPU** (Arc 140V) | **4.7 ms (16.9×)** | 5.9 ms (23.9×) |
| OpenVINO | INT8 | iGPU | 4.9 ms (16.2×) | 5.4 ms (26.3×) |
| OpenVINO | FP16 | **NPU** (AI Boost) | **4.6 ms (17.3×)** | 6.4 ms (22.2×) |
| OpenVINO | INT8 | NPU | 5.4 ms (14.6×) | 6.2 ms (23.0×) |

The second run happened while Docker and Chrome were using the CPU heavily. **CPU paths got
1.3–1.8× slower; the iGPU and NPU barely moved.** Offloading inference also protects it from
whatever else the machine is doing.

**Accuracy**: 500 COCO val2017 images, never seen by YOLO11n in training:

| Model | mAP50-95 | mAP50 |
|---|---|---|
| PyTorch FP32 | 0.418 | 0.576 |
| ONNX FP32 | 0.424 | 0.579 |
| OpenVINO FP32 | 0.424 | 0.579 |
| OpenVINO INT8 | **0.423** | 0.577 |

### Takeaways
- **Moving from PyTorch on the CPU to OpenVINO on the iGPU or NPU is ~17× faster** on an idle machine,
  taking the same model from 13 FPS to over 200 FPS, and the gap grows when the CPU is busy.
- **INT8 halves CPU latency** (34.6 → 16.7 ms) for a **0.1-point** mAP drop (0.424 → 0.423) on held-out
  images. On the iGPU and NPU it brings little or no speed-up: FP16 is already native there, and for a
  model this small, fixed overheads dominate.
- **The iGPU and NPU are effectively tied**; their ranking flips between runs. The NPU's advantage is
  **power efficiency**, which this benchmark doesn't measure, not raw speed.
- PyTorch scores ~0.006 mAP *below* the exported models. Likely small differences in how each format is
  pre/post-processed during validation; not investigated.

### Corrections after a code review
- The first version measured accuracy on **COCO128**, which is part of YOLO11n's own *training* data and
  was also the INT8 calibration set, so it wasn't a held-out result (it reported a 0.4-point INT8 drop).
  Accuracy is now measured on 500 unseen val2017 images; calibration stays on COCO128 (disjoint).
- GPU precision is now set explicitly (`INFERENCE_PRECISION_HINT`) instead of relying on plugin
  defaults; the NPU doesn't accept that option and always computes in FP16.
- Installing project 03's OCR package had silently **downgraded OpenVINO to 2024.0**, which made INT8 on the
  iGPU fail to compile. OpenVINO is now pinned to 2026.4.1 (see `common/ov_compat.py` and `setup_env.ps1`).

## How it works

1. **Export**: Ultralytics exports YOLO11n to ONNX (graph simplified with onnxslim) and to
   OpenVINO IR. INT8 uses NNCF post-training quantization, calibrated on COCO128 (train2017) images. No retraining.
2. **Latency**: each runtime gets the same random 1×3×640×640 tensor; we time only the model call,
   after warm-up, with the OpenVINO `LATENCY` performance hint.
3. **Accuracy**: each exported model is validated on the first 500 COCO val2017 images (converted to
   YOLO format by `coco_val_subset()`) with the same pipeline, so the numbers isolate the effect of the
   runtime and precision.

## A real-world gotcha: hybrid CPUs

The first run showed ONNX Runtime **2× slower than PyTorch** (170 ms), which made no sense.
Testing thread counts in isolation:

| ONNX Runtime threads | Latency |
|---|---|
| default (standalone process) | 34.8 ms |
| 4 | 39.8 ms |
| 8 | 166.0 ms |

The Core Ultra 7 258V has **4 performance + 4 low-power cores**. With 8 threads, the slow cores
hold everyone back at every synchronization point. In the benchmark process (after PyTorch had
already run), ORT ended up in the 8-thread regime. Pinning PyTorch and ORT to 4 threads
(`--threads 4`, now the default) fixed it. **Lesson: on hybrid CPUs, more threads can be 4× slower,
so always benchmark thread counts.**

## Run it

```bash
.venv\Scripts\activate            # shared environment, from the repo root
cd 19-edge-deployment-benchmark
python benchmark.py                 # export (first run only) + latency + accuracy
python benchmark.py --runs 300 --skip-accuracy
```

Exports are cached in `../../cv-data/edge/outputs/`. Results go to `docs/results.json`,
`docs/results.md` and `docs/latency.png`. The NPU needs the Intel NPU driver (installed with
recent Windows updates on Core Ultra laptops); unsupported devices are skipped and reported.

## Limitations

- **Model-only latency.** Pre-processing (resize/normalize) and post-processing (NMS) aren't
  included; in a full pipeline they add a few ms and can become the bottleneck on the iGPU or NPU.
- **Batch 1, single stream.** Throughput-oriented settings (batching, `THROUGHPUT` hint,
  several streams) would raise FPS further on the iGPU.
- **500 validation images, not all 5,000**: enough to show INT8 doesn't hurt accuracy, not a full COCO evaluation.
- **Latency depends on what else runs.** Two runs a day apart differ by up to 1.8× on the CPU (background
  apps); both are reported rather than picking the better one.
- **No power measurement.** The NPU's real selling point (performance per watt) isn't captured here.

## Next steps

- End-to-end pipeline benchmark (decode → pre-process → infer → NMS) and plug it into project 08
  (traffic analytics ran at 3.4 FPS on CPU).
- Measure power draw per inference (CPU vs iGPU vs NPU).

## Tech stack

Python · Ultralytics YOLO11 · PyTorch · ONNX / ONNX Runtime · OpenVINO 2026.4 · NNCF (INT8)
