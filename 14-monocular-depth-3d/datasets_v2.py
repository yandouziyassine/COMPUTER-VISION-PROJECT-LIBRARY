"""v2 evaluation datasets: each loader yields (image_id, rgb uint8 (H, W, 3), gt float32 meters (H, W), extra_mask).

GT is returned at the protocol's evaluation resolution (configs/protocol.yaml `downsample`), and RGB at the same
size, so predictions, GT and masks line up pixel for pixel. KITTI, DIODE and ETH3D are streamed from the
evaluation archives prepared by the Marigold authors (same file lists as their published numbers), so the
big archives never have to be stored: each member is decoded in memory and only the needed arrays are kept.
"""

import io
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np

MARIGOLD = "https://share.phys.ethz.ch/~pf/bingkedata/marigold/evaluation_dataset"
SPLITS = "https://raw.githubusercontent.com/prs-eth/Marigold/main/data_split"
NYU_URL = "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"
NYU_SPLIT_URL = "http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat"
IBIMS_FTP = "ftp://m1455541:m1455541@dataserv.ub.tum.de/ibims1_core_mat.zip"  # public login printed on mediaTUM
ETH3D_HW = (4032, 6048)


def _lines(url):
    return [ln.split() for ln in urllib.request.urlopen(url).read().decode().splitlines() if ln.strip()]


def _stream_tar(url):
    """Iterate (name, bytes) over a remote tar without saving it."""
    with urllib.request.urlopen(url) as r, tarfile.open(fileobj=r, mode="r|") as tar:
        for m in tar:
            if m.isfile():
                yield m.name.lstrip("./"), tar.extractfile(m).read()


def _png(b, flags=cv2.IMREAD_UNCHANGED):
    return cv2.imdecode(np.frombuffer(b, np.uint8), flags)


def _rgb(b):
    return cv2.cvtColor(_png(b, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def _shrink_rgb(rgb, f):
    return rgb if f == 1 else cv2.resize(rgb, (rgb.shape[1] // f, rgb.shape[0] // f), interpolation=cv2.INTER_AREA)


def _paired(stream, wanted_rgb, wanted_gt, decode_rgb, decode_gt):
    """Tar members arrive in archive order; pair RGB and GT members of the same sample as they appear."""
    rgbs, gts = {}, {}
    for name, b in stream:
        if name in wanted_rgb:
            rgbs[wanted_rgb[name]] = decode_rgb(b)
        elif name in wanted_gt:
            gts[wanted_gt[name]] = decode_gt(b)
        else:
            continue
        for k in set(rgbs) & set(gts):
            yield k, rgbs.pop(k), gts.pop(k)


def nyu(ds, root):
    """Official 654 test images from the labeled .mat (same loader as v1)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from nyu import NYU

    d = NYU()
    for i in d.test_indices():
        yield f"nyu_{i:04d}", d.rgb(i), d.depth(i).astype(np.float32), None


def kitti(ds, root):
    from protocol import kb_crop

    rows = [r for r in _lines(f"{SPLITS}/kitti_depth/eigen_test_files_with_gt.txt") if r[1] != "None"]
    want_rgb = {r[0]: r[0] for r in rows}
    want_gt = {r[1]: r[0] for r in rows}
    for k, rgb, gt in _paired(
        _stream_tar(f"{MARIGOLD}/kitti/kitti_eigen_split_test.tar"),
        want_rgb,
        want_gt,
        _rgb,
        lambda b: _png(b).astype(np.float32) / 256.0,
    ):
        yield k, kb_crop(rgb), kb_crop(gt), None


def diode(ds, root):
    rows = _lines(f"{SPLITS}/diode_depth/diode_val_all_filename_list.txt")
    masks = {}
    want_rgb = {r[0]: r[0] for r in rows}
    want_gt = {r[1]: r[0] for r in rows}
    want_mask = {r[2]: r[0] for r in rows}

    def stream():
        for name, b in _stream_tar(f"{MARIGOLD}/diode/diode_val.tar"):
            if name in want_mask:
                masks[want_mask[name]] = np.load(io.BytesIO(b)).squeeze().astype(bool)
            else:
                yield name, b

    pending = []
    for k, rgb, gt in _paired(stream(), want_rgb, want_gt, _rgb, lambda b: np.load(io.BytesIO(b)).squeeze()):
        pending.append((k, rgb, gt.astype(np.float32)))
        for item in [p for p in pending if p[0] in masks]:
            pending.remove(item)
            yield item[0], item[1], item[2], masks.pop(item[0])
    for k, rgb, gt in pending:  # masks that came after both images
        yield k, rgb, gt, masks.pop(k, None)


def eth3d(ds, root):
    f = ds["downsample"]
    rows = _lines(f"{SPLITS}/eth3d_depth/eth3d_filename_list.txt")
    want_rgb = {r[0]: r[0] for r in rows}
    want_gt = {r[1]: r[0] for r in rows}

    def gt_decode(b):
        d = np.frombuffer(b, np.float32).reshape(ETH3D_HW)[::f, ::f].copy()
        d[~np.isfinite(d)] = 0.0
        return d

    for k, rgb, gt in _paired(
        _stream_tar(f"{MARIGOLD}/eth3d/eth3d.tar"), want_rgb, want_gt, lambda b: _shrink_rgb(_rgb(b), f), gt_decode
    ):
        yield k, rgb[: gt.shape[0], : gt.shape[1]], gt, None


def ibims(ds, root):
    """iBims-1 core set (100 images, 640 x 480, laser scan GT); `mask_invalid` is 1 on usable pixels, 0 elsewhere."""
    from scipy.io import loadmat

    path = Path(root) / "ibims1_core_mat.zip"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(IBIMS_FTP, path)
    with zipfile.ZipFile(path) as z:
        for name in sorted(n for n in z.namelist() if n.endswith(".mat")):
            d = loadmat(io.BytesIO(z.read(name)))["data"]
            rgb, gt = d["rgb"][0, 0], d["depth"][0, 0].astype(np.float32)
            yield Path(name).stem, rgb, gt, d["mask_invalid"][0, 0].astype(bool)


LOADERS = {"nyu": nyu, "kitti": kitti, "diode": diode, "eth3d": eth3d, "ibims": ibims}
