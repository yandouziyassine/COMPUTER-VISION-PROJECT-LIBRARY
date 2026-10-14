| Runtime | Precision | Device | Mean latency (ms) | FPS | Speed-up vs PyTorch |
|---|---|---|---|---|---|
| PyTorch | FP32 | CPU | 141.23 | 7.1 | 1.0x |
| ONNX Runtime | FP32 | CPU | 61.22 | 16.3 | 2.3x |
| OpenVINO | FP32 | CPU | 47.16 | 21.2 | 3.0x |
| OpenVINO | INT8 | CPU | 21.52 | 46.5 | 6.6x |
| OpenVINO | FP16 | GPU | 5.92 | 169.0 | 23.9x |
| OpenVINO | INT8 | GPU | 5.38 | 185.9 | 26.3x |
| OpenVINO | FP16 | NPU | 6.35 | 157.4 | 22.2x |
| OpenVINO | INT8 | NPU | 6.15 | 162.5 | 23.0x |

| Model | mAP50-95 | mAP50 |
|---|---|---|
| PyTorch FP32 | 0.4181 | 0.5762 |
| ONNX FP32 | 0.4241 | 0.5791 |
| OpenVINO FP32 | 0.424 | 0.5793 |
| OpenVINO INT8 | 0.4231 | 0.5768 |
