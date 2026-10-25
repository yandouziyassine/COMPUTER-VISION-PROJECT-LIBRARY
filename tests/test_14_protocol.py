"""Project 14 v2: evaluation protocol (config, masks, alignment, bootstrap, manifest) on synthetic data."""

import pytest

np = pytest.importorskip("numpy")

from conftest import load_module  # noqa: E402

p = load_module("14", "protocol.py")
PROTO = p.load_protocol()


def test_protocol_yaml_parses_with_the_expected_structure():
    assert PROTO["bootstrap"]["resamples"] == 1000 and PROTO["bootstrap"]["confidence"] == 0.95
    assert PROTO["alignment"]["disparity"] == ["lsq_disparity"]
    assert PROTO["alignment"]["metric"][0] == "none"
    assert PROTO["datasets"]["nyu"]["max_depth"] == 10.0 and PROTO["datasets"]["nyu"]["crop"] == "eigen_nyu"
    assert PROTO["datasets"]["eth3d"]["max_depth"] is None
    assert PROTO["datasets"]["kitti"]["kb_crop"] is True


def test_minimal_yaml_reader_handles_nesting_comments_and_lists():
    d = p.parse_yaml("a:\n  b: 1  # c\n  c: [x, 2.5]\n# full comment\nd: null\ne: text with spaces\n")
    assert d == {"a": {"b": 1, "c": ["x", 2.5]}, "d": None, "e": "text with spaces"}


def test_crops_cover_the_published_regions():
    nyu = p.crop_mask((480, 640), "eigen_nyu")
    assert nyu.sum() == (471 - 45) * (601 - 41) and nyu[45, 41] and not nyu[44, 41]
    k = p.crop_mask((352, 1216), "eigen_kitti")
    assert k[int(0.3324324 * 352), 100] and not k[0, 100] and not k[200, 0]
    assert p.kb_crop(np.zeros((375, 1242, 3))).shape == (352, 1216, 3)


def test_valid_mask_applies_range_crop_and_nonfinite():
    ds = {"min_depth": 0.6, "max_depth": 350.0, "crop": "none"}
    gt = np.array([[0.5, 1.0], [np.inf, 400.0]])
    assert p.valid_mask(gt, ds).tolist() == [[False, True], [False, False]]


@pytest.mark.parametrize(
    "method, make_pred",
    [
        ("lsq_disparity", lambda gt: (1 / gt - 0.05) / 3.0),
        ("lsq_depth", lambda gt: 2.0 * gt + 0.7),
        ("median", lambda gt: 0.37 * gt),
        ("none", lambda gt: gt.copy()),
    ],
)
def test_each_alignment_recovers_its_own_ambiguity(method, make_pred):
    gt = np.random.default_rng(0).uniform(1, 8, (40, 50))
    mask = np.ones_like(gt, bool)
    d = p.align(make_pred(gt), gt, mask, method, max_depth=10.0, min_depth=1e-3)
    assert np.allclose(d, gt, rtol=1e-6)
    assert p.image_metrics(d, gt, mask)["absrel"] == pytest.approx(0, abs=1e-6)


def test_alignment_clips_to_the_depth_cap():
    gt = np.full((4, 4), 5.0)
    gt[0, 0] = 9.0
    d = p.align(gt * 3, gt, np.ones_like(gt, bool), "none", max_depth=10.0, min_depth=1e-3)
    assert d.max() == 10.0


def test_score_image_reports_every_alignment_for_metric_models():
    gt = np.random.default_rng(1).uniform(1, 7, (480, 640))  # x1.27 stays under the 10 m cap
    s = p.score_image(gt * 1.27, gt, PROTO["datasets"]["nyu"], "metric", PROTO)
    assert list(s) == ["none", "median", "lsq_depth"]
    assert s["none"]["absrel"] == pytest.approx(0.27, abs=1e-6) and s["none"]["delta1"] == 0.0
    assert s["median"]["absrel"] == pytest.approx(0, abs=1e-9)


def test_bootstrap_ci_contains_the_mean_and_shrinks_with_more_images():
    rng = np.random.default_rng(2)
    small, big = rng.normal(0.1, 0.02, 50), rng.normal(0.1, 0.02, 5000)
    m, lo, hi = p.bootstrap_ci(small)
    assert lo < m < hi
    _, lo2, hi2 = p.bootstrap_ci(big)
    assert hi2 - lo2 < (hi - lo) / 5
    assert p.bootstrap_ci(small) == p.bootstrap_ci(small)  # seeded: reproducible


def test_manifest_records_and_verifies_hashes(tmp_path):
    f = tmp_path / "m_nyu_000.npz"
    np.savez(f, a=np.arange(10, dtype=np.float16))
    man = tmp_path / "manifest.json"
    p.update_manifest(man, f, "m", "nyu", "abc123", 1, 10)
    assert p.verify_manifest(man, tmp_path) == []
    np.savez(f, a=np.arange(11, dtype=np.float16))
    assert p.verify_manifest(man, tmp_path) == ["m_nyu_000.npz"]
