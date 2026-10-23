"""NYU Depth V2 labeled subset (1,449 Kinect RGB-D images, 894 object classes): data access.

The .mat file is HDF5 with transposed arrays; everything returned here is (480, 640)."""

import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import data_dir  # noqa: E402

ROOT = data_dir("depth") / "nyu"
K = (518.8579, 519.4696, 325.5824, 253.7362)  # RGB camera intrinsics (official NYU toolbox)


class NYU:
    def __init__(self):
        self.f = h5py.File(ROOT / "nyu_depth_v2_labeled.mat", "r")
        refs = self.f["names"][0]
        self.names = ["".join(chr(c) for c in self.f[r][:].ravel()) for r in refs]

    def test_indices(self):
        """The official 654 test images, as 0-based indices."""
        return [int(i) - 1 for i in loadmat(ROOT / "splits.mat")["testNdxs"].ravel()]

    def rgb(self, i):
        return np.ascontiguousarray(self.f["images"][i].transpose(2, 1, 0))  # (480, 640, 3) uint8

    def depth(self, i):
        return np.ascontiguousarray(self.f["depths"][i].T)  # hole-filled, meters

    def raw(self, i):
        return np.ascontiguousarray(self.f["rawDepths"][i].T)  # sensor only, 0 = no data

    def labels(self, i):
        return np.ascontiguousarray(self.f["labels"][i].T)  # 0 = unlabeled

    def class_id(self, name):
        return self.names.index(name) + 1
