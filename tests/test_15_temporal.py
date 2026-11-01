"""Project 15: inputs and causality of the temporal models (1D-CNN / GRU on keypoint sequences)."""

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from conftest import load_module  # noqa: E402

tm = load_module("15", "temporal_model.py")


def skeleton_clip(T=40, seed=0):
    rng = np.random.default_rng(seed)
    kps = np.concatenate([rng.uniform(0, 240, (T, 17, 2)), np.ones((T, 17, 1))], axis=2).astype(np.float32)
    boxes = np.tile(np.array([100, 40, 140, 220], np.float32), (T, 1))
    return kps, boxes


def test_ffill_time_forward_then_backward():
    a = np.array([[np.nan, 5.0], [1.0, np.nan], [np.nan, np.nan], [3.0, 7.0], [np.nan, np.nan]])
    np.testing.assert_array_equal(tm.ffill_time(a), [[1, 5], [1, 5], [1, 5], [3, 7], [3, 7]])
    assert (tm.ffill_time(np.full((3, 2), np.nan)) == 0).all()


def test_mirror_twice_is_identity_and_swaps_sides():
    kps, boxes = skeleton_clip()
    mk, mb = tm.mirror(kps, boxes, frame_w=320)
    assert mk[0, 5, 0] == pytest.approx(320 - kps[0, 6, 0])  # left shoulder <- mirrored right shoulder
    assert (mb[:, 0] <= mb[:, 2]).all()
    k2, b2 = tm.mirror(mk, mb, frame_w=320)
    np.testing.assert_allclose(k2, kps, atol=1e-4)
    np.testing.assert_allclose(b2, boxes, atol=1e-4)


def test_sequence_inputs_shape_and_missing_frames():
    kps, boxes = skeleton_clip()
    kps[10:15], boxes[10:15] = np.nan, np.nan
    x = tm.sequence_inputs(kps, boxes)
    assert x.shape == (40, 94) and np.isfinite(x).all()
    present = x[:, 34 + 17 + 4]
    assert present[10:15].sum() == 0 and present[:10].all()


@pytest.mark.parametrize("arch", ["tcn", "gru"])
def test_models_are_causal(arch):
    """Changing the future must not change past outputs: the models can run online."""
    torch.manual_seed(0)
    model = tm.build(arch, 94).eval()
    x = torch.randn(1, 80, 94)
    x2 = x.clone()
    x2[:, 50:] += 5.0
    with torch.no_grad():
        a, b = model(x), model(x2)
    assert a.shape == (1, 80)
    torch.testing.assert_close(a[:, :50], b[:, :50])
    assert not torch.allclose(a[:, 50:], b[:, 50:])


def test_train_learns_a_trivial_signal():
    """A positive frame is announced by feature 0 being high: a few hundred steps must pick it up."""
    rng = np.random.default_rng(0)
    seqs = []
    for _ in range(8):
        x = rng.normal(size=(60, 4)).astype(np.float32)
        y = (x[:, 0] > 1.0).astype(np.float32)
        seqs.append((x, y))
    model = tm.train("tcn", seqs, steps=150, batch=8, length=60, noise=0.0)
    p = tm.predict(model, seqs[0][0])
    y = seqs[0][1]
    assert p[y == 1].mean() > p[y == 0].mean() + 0.3
