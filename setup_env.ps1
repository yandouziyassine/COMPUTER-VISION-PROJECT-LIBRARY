# Creates the shared deep-learning environment in .venv next to this script.
# Caches (pip, Hugging Face, torch hub) go to ..\cv-data\.cache so nothing large lands on C:.
#
# Usage (PowerShell, from anywhere):
#   powershell -ExecutionPolicy Bypass -File setup_env.ps1
#
# Re-run it if the USB key gets a different drive letter: venvs store absolute paths.

$ErrorActionPreference = "Stop"
$root  = $PSScriptRoot
$cache = Join-Path (Split-Path $root -Parent) "cv-data\.cache"
New-Item -ItemType Directory -Force $cache | Out-Null
$env:PIP_CACHE_DIR = Join-Path $cache "pip"

$py = "py"; $pyArgs = @("-3.10")
& $py @pyArgs -m venv (Join-Path $root ".venv")
$venvPy = Join-Path $root ".venv\Scripts\python.exe"

& $venvPy -m pip install --upgrade pip
& $venvPy -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
& $venvPy -m pip install -r (Join-Path $root "requirements-shared.txt")
# OCR for projects 03/06: --no-deps so its openvino<=2024 pin can't downgrade OpenVINO (see common/ov_compat.py)
& $venvPy -m pip install --no-deps rapidocr-openvino==1.4.4
& $venvPy -c "import torch, cv2, timm, ultralytics, openvino; print('env ok | torch', torch.__version__, '| cv2', cv2.__version__, '| openvino', openvino.__version__)"
