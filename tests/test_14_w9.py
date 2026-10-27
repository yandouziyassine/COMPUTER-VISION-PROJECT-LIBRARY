"""Project 14 W9 (human vs machine study): pair construction and analysis helpers on synthetic data."""

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from conftest import load_defs, load_module  # noqa: E402

an = load_module("14", "w9_analyze.py")
m = load_module("14", "metrics.py")
K = (500.0, 500.0, 320.0, 240.0)
b = load_defs(
    "14",
    "w9_build_study.py",
    ["MIN_SEP", "ring", "surface_depth_from_ring", "local_std", "pick_pair", "make_pair"],
    {"np": np, "cv2": cv2, "backproject": m.backproject, "fit_plane": m.fit_plane, "plane_depth": m.plane_depth},
)


def test_fleiss_kappa_perfect_chance_and_split():
    assert an.fleiss_kappa([[5, 0], [0, 5], [5, 0]]) == pytest.approx(1.0)
    # every item split as evenly as possible -> agreement below chance
    assert an.fleiss_kappa([[2, 2], [2, 2]]) < 0
    with pytest.raises(ValueError):
        an.fleiss_kappa([[3, 0], [1, 1]])


def test_majority_pairwise_agreement_and_outcomes():
    assert an.majority(["a", "a", "b"]) == "a" and an.majority(["a", "b"]) is None
    assert an.pairwise_agreement([{"p1": "a", "p2": "b"}, {"p1": "a", "p2": "a"}]) == pytest.approx(0.5)
    assert an.outcome(True, False) == "humans win" and an.outcome(False, False) == "both fail"


def test_bootstrap_mean_brackets_the_mean():
    mean, lo, hi = an.bootstrap_mean([1, 0, 1, 1, 0, 1, 1, 1])
    assert lo <= mean <= hi and mean == pytest.approx(0.75)


def test_mirror_surface_depth_is_the_plane_of_the_frame_not_the_reflection():
    vv, uu = np.mgrid[0:480, 0:640]
    wall = 1.0 / (1 / 3.0 + 0.0002 * (uu - K[2]))  # tilted wall: 1/depth linear in pixels
    raw = wall.copy()
    mirror = np.zeros((480, 640), bool)
    mirror[150:300, 250:400] = True
    raw[mirror] = 6.0  # the sensor measures the reflected room, twice as far
    plane = b.surface_depth_from_ring(mirror, raw, K)
    assert np.allclose(plane[mirror], wall[mirror], rtol=1e-4)
    assert b.ring(mirror, 10).sum() > 0 and not (b.ring(mirror, 10) & mirror).any()


def test_pick_pair_respects_ratio_and_separation():
    rng = np.random.default_rng(0)
    depth = np.tile(np.linspace(1, 4, 640), (480, 1))
    cand = np.nonzero(np.ones((480, 640), bool))
    a, bb, za, zb = b.pick_pair(rng, cand, cand, depth, depth, 1.5)
    assert max(za, zb) / min(za, zb) >= 1.5 and np.hypot(a[0] - bb[0], a[1] - bb[1]) >= b.MIN_SEP
    pair = b.make_pair("x", "normal", "img.jpg", a, bb, za, zb, "why", rng)
    near = pair["a"] if pair["truth"] == "a" else pair["b"]
    assert depth[near[1], near[0]] == pytest.approx(min(za, zb))


def test_local_std_is_zero_on_flat_regions():
    g = np.full((60, 60), 100, np.uint8)
    g[:, 30:] = np.random.default_rng(1).integers(0, 255, (60, 30))
    s = b.local_std(g)
    assert s[30, 5] < 1e-3 and s[30, 50] > 20


def test_error_consistency_kappa():
    same = [True, False, True, True, False, True]
    assert an.error_kappa(same, same) == pytest.approx(1.0)
    # independent errors with the same accuracies -> about 0
    a = [True, True, False, False]
    b = [True, False, True, False]
    assert an.error_kappa(a, b) == pytest.approx(0.0)
    assert an.error_kappa([True, False], [False, True]) == pytest.approx(-1.0)


def test_game_sessions_are_kept_as_their_own_group(tmp_path):
    import json

    game = {"session": "g1", "n": 12, "source": "portfolio-game", "plan": {"exposure": "free", "transform": "none"},
            "answers": [{"id": "p001", "choice": "a", "ms": 900, "exposure": "free", "transform": "none"}]}  # fmt: skip
    full = {"session": "s1", "answers": [{"id": f"p{i:03d}", "choice": "b", "ms": 800} for i in range(300)]}
    short = {"session": "s2", "answers": [{"id": "p001", "choice": "b", "ms": 800}]}
    for name, s in (("g.json", game), ("answers_s1.json", full), ("answers_s2.json", short)):
        (tmp_path / name).write_text(json.dumps(s))
    study, games = an.load_humans(tmp_path, 250)
    assert [r["session"] for r in study] == ["s1"] and [r["session"] for r in games] == ["g1"]


def test_public_licences_are_normalised():
    import re

    pub = load_defs("14", "w9_public.py", ["ALLOWED_LICENCES", "normalise_licence"], {"re": re})
    assert pub.normalise_licence("CC BY-SA 4.0") == "CC BY-SA 4.0" and pub.normalise_licence("CC0 1.0") == "CC0"
    with pytest.raises(ValueError):
        pub.normalise_licence("CC BY-NC 4.0")
