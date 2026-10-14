"""Measure the real world from one photo: metric depth + camera intrinsics + two pixels -> meters."""

import numpy as np
from PIL import Image

import models as M
from metrics import distance

DEFAULT_FOV = 60.0


def intrinsics_from_fov(w, h, fov_deg=DEFAULT_FOV):
    f = 0.5 * w / np.tan(np.radians(fov_deg) / 2)
    return (float(f), float(f), w / 2, h / 2)


def intrinsics_from_35mm(w, h, focal_35mm):
    """EXIF FocalLengthIn35mmFilm: a full-frame sensor is 36 mm wide."""
    f = focal_35mm / 36.0 * w
    return (float(f), float(f), w / 2, h / 2)


def intrinsics(img):
    """From EXIF when the photo has it, else the default field of view. Returns (K, source)."""
    focal = img.getexif().get_ifd(0x8769).get(0xA405)            # FocalLengthIn35mmFilm
    if focal:
        return intrinsics_from_35mm(*img.size, float(focal)), f"EXIF ({focal} mm equivalent)"
    return intrinsics_from_fov(*img.size), f"assumed {DEFAULT_FOV:.0f}° field of view"


class Measurer:
    def __init__(self):
        self.est = M.load("da2-metric-indoor")

    def depth(self, img):
        return M.predict(self.est, np.asarray(img.convert("RGB")))

    def measure(self, img: Image.Image, p1, p2, depth=None, reference=None):
        """Distance in meters between pixels p1 and p2. reference = (q1, q2, true_meters) rescales the depth
        so a known length in the photo comes out exact."""
        depth = self.depth(img) if depth is None else depth
        K, source = intrinsics(img)
        scale = 1.0
        if reference:
            q1, q2, true_m = reference
            scale = true_m / distance(q1, q2, depth, K)
        return distance(p1, p2, depth * scale, K), source
