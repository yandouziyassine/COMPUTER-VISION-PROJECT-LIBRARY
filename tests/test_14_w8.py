"""Project 14 W8 (mirror fine-tune): loss and planarity metric."""

import pytest

np = pytest.importorskip("numpy")

from conftest import load_module  # noqa: E402

w8 = load_module("14", "w8_mirror_finetune.py")


def test_ssi_loss_ignores_scale_and_shift():
    torch = pytest.importorskip("torch")
    t = torch.rand(2, 20, 30) + 0.5
    m = torch.ones_like(t, dtype=torch.bool)
    assert w8.ssi_l1(3 * t + 0.2, t, m).item() == pytest.approx(0, abs=1e-4)
    assert w8.ssi_l1(torch.rand(2, 20, 30), t, m).item() > 0.05


def test_planarity_is_zero_for_a_plane_and_positive_for_a_bump():
    load_module("14", "metrics.py")
    K = (500.0, 500.0, 320.0, 240.0)
    vv, uu = np.mgrid[0:100, 0:100]
    region = np.ones((100, 100), bool)
    flat = 1.0 / (1 / 3.0 + 0.0002 * (uu - 50))
    assert w8.planarity(flat, region, K) < 1e-6
    bump = flat + 0.3 * np.exp(-((uu - 50) ** 2 + (vv - 50) ** 2) / 200)
    assert w8.planarity(bump, region, K) > 0.005
