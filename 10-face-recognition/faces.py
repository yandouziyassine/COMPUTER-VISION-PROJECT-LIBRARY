"""Face detection, alignment and embedding with OpenCV's model zoo (no extra dependencies).

    image → YuNet (face box + 5 landmarks) → align & crop to 112x112 → SFace → 128-d embedding

Two faces belong to the same person when the cosine similarity of their embeddings is above a
threshold. Recognition = comparing embeddings, never storing or comparing raw photos.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import data_dir, output_dir  # noqa: E402

DATA = data_dir("faces")
OUT = output_dir("faces")
DOCS = Path(__file__).resolve().parent / "docs"
YUNET = OUT / "face_detection_yunet_2023mar.onnx"
SFACE = {"fp32": OUT / "face_recognition_sface_2021dec.onnx", "int8": OUT / "face_recognition_sface_2021dec_int8.onnx"}


class FaceEmbedder:
    def __init__(self, precision="fp32", score_threshold=0.6):
        self.detector = cv2.FaceDetectorYN.create(str(YUNET), "", (320, 320), score_threshold, 0.3, 50)
        self.recognizer = cv2.FaceRecognizerSF.create(str(SFACE[precision]), "")

    def detect(self, img):
        h, w = img.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(img)
        return faces if faces is not None else np.empty((0, 15), np.float32)

    def embed(self, img, face=None):
        """Embedding of `face` (a YuNet row), or of the face closest to the image center."""
        if face is None:
            faces = self.detect(img)
            if len(faces) == 0:
                return None
            h, w = img.shape[:2]
            centers = faces[:, :2] + faces[:, 2:4] / 2
            face = faces[np.argmin(np.linalg.norm(centers - [w / 2, h / 2], axis=1))]
        aligned = self.recognizer.alignCrop(img, face)
        f = self.recognizer.feature(aligned).ravel()
        return f / np.linalg.norm(f)


def cosine(a, b):
    return float(np.dot(a, b))
