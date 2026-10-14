# PatchCore anomaly detection (project 13), CPU only.
#
# Build from the repo root (the image needs common/ too):
#   docker build -f 13-anomaly-detection-patchcore/Dockerfile -t cv-patchcore .
# Run with the datasets mounted at /data (see README, "Run with Docker"):
#   docker run --rm -v /path/to/cv-data:/data cv-patchcore --categories bottle

FROM python:3.10-slim

# common/paths.py reads datasets from $CV_DATA and keeps model caches in $CV_DATA/.cache
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    CV_DATA=/data

# CPU-only PyTorch first (the default wheels pull ~2 GB of CUDA libraries), then the rest.
# Versions match the environment the README numbers were measured with.
RUN pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu

WORKDIR /app
COPY 13-anomaly-detection-patchcore/requirements.txt 13-anomaly-detection-patchcore/requirements.txt
RUN pip install -r 13-anomaly-detection-patchcore/requirements.txt

COPY common/ common/
COPY 13-anomaly-detection-patchcore/ 13-anomaly-detection-patchcore/
WORKDIR /app/13-anomaly-detection-patchcore

# MVTec AD lives in /data/mvtec/<category>/; memory banks go to /data/mvtec/outputs/,
# the WideResNet-50 weights are downloaded once into /data/.cache/torch/.
VOLUME ["/data"]
ENTRYPOINT ["python", "run.py"]
CMD ["--categories", "bottle", "metal_nut", "grid"]
