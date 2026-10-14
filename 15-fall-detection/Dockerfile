# Fall detection (project 15) on a video file, CPU only.
#
# Build from the repo root (the image needs common/ too):
#   docker build -f 15-fall-detection/Dockerfile -t cv-fall .
# Run on a video stored in the mounted data folder (see README, "Run with Docker"):
#   docker run --rm -v /path/to/cv-data:/data cv-fall /data/fall/videos/fall-01-cam0.mp4 --ur-fall
#
# Webcam mode (`detect.py 0`) needs a display and a camera device: not supported in this image.

FROM python:3.10-slim

# common/paths.py reads data from $CV_DATA; Ultralytics settings go to $CV_DATA/.cache/ultralytics
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    CV_DATA=/data

# OpenCV (pulled in by ultralytics as opencv-python) needs these shared libraries on a slim image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch first, so ultralytics doesn't pull the CUDA build.
# Versions match the environment the README numbers were measured with.
RUN pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu

WORKDIR /app
COPY 15-fall-detection/requirements.txt 15-fall-detection/requirements.txt
# scikit-learn pinned: fall_model.joblib is a pickle and must be loaded with the version that wrote it.
RUN pip install ultralytics==8.4.175 scikit-learn==1.7.2 -r 15-fall-detection/requirements.txt

COPY common/ common/
COPY 15-fall-detection/ 15-fall-detection/
WORKDIR /app/15-fall-detection

# Expects /data/fall/outputs/fall_model.joblib (written by `python evaluate.py`) and
# yolo11n-pose.pt in the same folder (downloaded there on first use).
# The annotated video is written to /data/fall/outputs/detected.mp4 unless --out is given.
VOLUME ["/data"]
ENTRYPOINT ["python", "detect.py"]
CMD ["--help"]
