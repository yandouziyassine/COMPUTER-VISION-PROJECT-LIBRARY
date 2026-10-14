"""Project 13: PatchCore pieces that don't need the pretrained backbone.

PatchCore.__init__ downloads Wide-ResNet50 weights, so the object is built with __new__ instead."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from conftest import load_module  # noqa: E402

pc = load_module("13", "patchcore.py")


def bare_patchcore(coreset_ratio, proj_dim=16):
    model = pc.PatchCore.__new__(pc.PatchCore)
    model.coreset_ratio, model.proj_dim, model.device = coreset_ratio, proj_dim, "cpu"
    return model


def test_gaussian_blur_keeps_shape_and_constant_maps():
    maps = torch.full((2, 1, 40, 40), 3.0)
    out = pc.gaussian_blur(maps, sigma=4)
    assert out.shape == maps.shape
    assert torch.allclose(out, maps, atol=1e-5)  # kernel sums to 1 + reflect padding


def test_gaussian_blur_spreads_impulse_and_keeps_its_mass():
    maps = torch.zeros(1, 1, 64, 64)
    maps[0, 0, 32, 32] = 1.0
    out = pc.gaussian_blur(maps, sigma=2)
    assert out.sum().item() == pytest.approx(1.0, abs=1e-4)  # far from borders: mass preserved
    assert out[0, 0].argmax().item() == 32 * 64 + 32  # peak stays in place
    assert out.max().item() < 1.0
    assert torch.allclose(out[0, 0, 32, 30], out[0, 0, 32, 34])  # symmetric
    assert torch.allclose(out[0, 0, 30, 32], out[0, 0, 32, 30])  # isotropic


def test_greedy_coreset_size_and_subset():
    torch.manual_seed(0)
    feats = torch.randn(200, 64)
    bank = bare_patchcore(0.1)._greedy_coreset(feats)
    assert bank.shape == (20, 64)
    # Every selected row is one of the input rows, and no row is picked twice.
    same = (bank[:, None, :] == feats[None, :, :]).all(dim=-1)  # (20, 200) exact row matches
    assert same.any(dim=1).all()
    assert len(set(same.float().argmax(dim=1).tolist())) == 20


def test_greedy_coreset_keeps_at_least_one():
    assert bare_patchcore(0.001)._greedy_coreset(torch.randn(10, 8)).shape == (1, 8)


def test_greedy_coreset_covers_every_cluster():
    """Farthest-point sampling must visit all well-separated clusters, even a tiny one."""
    torch.manual_seed(0)
    centers = torch.eye(5, 32) * 50
    sizes = [300, 300, 300, 300, 5]
    feats = torch.cat([c + torch.randn(n, 32) for c, n in zip(centers, sizes)])
    labels = torch.cat([torch.full((n,), i) for i, n in enumerate(sizes)])
    bank = bare_patchcore(5 / len(feats), proj_dim=32)._greedy_coreset(feats)
    picked = labels[torch.cdist(bank, feats).argmin(dim=1)]
    assert sorted(picked.tolist()) == [0, 1, 2, 3, 4]
