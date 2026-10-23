"""Depth evaluation math: alignment, standard metrics, pinhole geometry. Pure NumPy, unit-tested."""

import numpy as np

EIGEN_CROP = (slice(45, 471), slice(41, 601))  # standard NYU evaluation crop (480 x 640 images)
MIN_DEPTH, MAX_DEPTH = 1e-3, 10.0


def eval_mask(gt):
    """Valid ground-truth pixels inside the Eigen crop."""
    mask = np.zeros(gt.shape, bool)
    mask[EIGEN_CROP] = True
    return mask & (gt > MIN_DEPTH) & (gt < MAX_DEPTH)


def align_disparity(pred_disp, gt_depth, mask):
    """Relative models predict disparity up to scale and shift. Fit s, t (least squares) so that
    s * pred + t ~ 1 / gt on the mask, then return the aligned *depth* for every pixel."""
    x = pred_disp[mask].astype(np.float64)
    y = 1.0 / gt_depth[mask].astype(np.float64)
    A = np.stack([x, np.ones_like(x)], axis=1)
    (s, t), *_ = np.linalg.lstsq(A, y, rcond=None)
    disp = s * pred_disp + t
    return 1.0 / np.clip(disp, 1.0 / MAX_DEPTH, None)


def abs_rel(pred, gt, mask):
    return float(np.mean(np.abs(pred[mask] - gt[mask]) / gt[mask]))


def delta1(pred, gt, mask):
    r = np.maximum(pred[mask] / gt[mask], gt[mask] / pred[mask])
    return float(np.mean(r < 1.25))


def backproject(u, v, z, K):
    """Pixels (u, v) with depth z (meters) -> 3D points (N, 3) in the camera frame. K = (fx, fy, cx, cy)."""
    fx, fy, cx, cy = K
    u, v, z = (np.asarray(a, np.float64) for a in (u, v, z))
    return np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], axis=-1)


def distance(p1, p2, depth, K):
    """3D distance (meters) between two pixels (u, v) of a depth map."""
    a = backproject(p1[0], p1[1], depth[int(p1[1]), int(p1[0])], K)
    b = backproject(p2[0], p2[1], depth[int(p2[1]), int(p2[0])], K)
    return float(np.linalg.norm(a - b))


def fit_plane(points):
    """Least-squares plane n . p + d = 0 (unit normal) through 3D points (N, 3)."""
    c = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - c, full_matrices=False)
    n = vt[-1]
    return n, -float(n @ c)


def plane_depth(u, v, n, d, K):
    """Depth at which each pixel's viewing ray hits the plane."""
    fx, fy, cx, cy = K
    rx, ry = (np.asarray(u, np.float64) - cx) / fx, (np.asarray(v, np.float64) - cy) / fy
    return -d / (n[0] * rx + n[1] * ry + n[2])
