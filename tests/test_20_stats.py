"""Project 20, stage S4: statistics engine on synthetic tracks (speed, sprints, possession, passes, formation)."""

import sys

import pytest

np = pytest.importorskip("numpy")

from conftest import project_dir  # noqa: E402

sys.path.insert(0, str(project_dir("20")))
from stats import noise, physical, tactics  # noqa: E402
from stats import possession as P  # noqa: E402
from stats.tracks import Tracks  # noqa: E402

FPS = 25.0


def test_constant_speed_is_recovered_including_edges():
    t = np.arange(250) / FPS
    xy = np.c_[5.0 * t, np.full_like(t, 30.0)]  # 5 m/s along x for 10 s
    _, speed = physical.velocity(xy, FPS)
    assert np.allclose(speed, 5.0, atol=1e-6)
    assert abs(np.sum(speed) / FPS - 5.0 * 10) < 0.5


def test_speed_is_capped_and_gaps_are_bridged():
    xy = np.c_[np.zeros(200), np.zeros(200)]
    xy[100:, 0] = 30.0  # 30 m teleport = tracking error
    _, speed = physical.velocity(xy, FPS)
    assert speed.max() <= physical.MAX_SPEED + 1e-9
    gappy = np.c_[np.arange(100.0), np.zeros(100)]
    gappy[40:50] = np.nan  # 10-frame gap (< 1 s): interpolated
    gappy[70:] = np.nan
    filled = physical.fill_gaps(gappy, 25)
    assert np.allclose(filled[40:50, 0], np.arange(40.0, 50.0))
    assert np.isnan(filled[70:, 0]).all()  # trailing gap: nothing to interpolate towards


def test_sprints_need_one_second_above_seven_m_s():
    speed = np.full(300, 3.0)
    speed[50:80] = 8.0  # 1.2 s -> sprint
    speed[150:170] = 8.0  # 0.8 s -> too short
    speed[200:260] = 7.5  # 2.4 s -> sprint
    n, dist = physical.sprints(speed, FPS)
    assert n == 2
    assert abs(dist - (30 * 8.0 + 60 * 7.5) / FPS) < 1e-9


def make_tracks(players, teams, ball):
    """players: (T, N, 2), ball: (T, 2) -> Tracks at 25 fps."""
    return Tracks(xy=np.asarray(players, float), team=np.asarray(teams), ball=np.asarray(ball, float), fps=FPS)


def static_players(T, positions):
    return np.repeat(np.asarray(positions, float)[None], T, axis=0)


def test_possession_hysteresis_ignores_brief_touches():
    A, B = (10.0, 10.0), (30.0, 10.0)
    xy = static_players(100, [A, B])
    ball = np.empty((100, 2))
    ball[:40] = A  # team 0 controls
    ball[40:43] = B  # 3 frames near team 1: below the 5-frame hysteresis
    ball[43:60] = A
    ball[60:] = B  # team 1 takes over for good
    tr = make_tracks(xy, [0, 1], ball)
    poss = P.possession(tr, P.ball_owner(tr, 1.0), min_frames=5)
    assert (poss[:4] == -1).all()  # nobody has had the ball for 5 frames yet
    assert (poss[4:64] == 0).all()  # the 3-frame touch by team 1 is ignored
    assert (poss[64:] == 1).all()  # switch after 5 control frames
    assert abs(P.possession_share(poss) - 60 / 96) < 1e-9


def flight(p0, p1, n):
    w = np.linspace(0, 1, n + 2)[1:-1, None]
    return np.asarray(p0) + w * (np.subtract(p1, p0))


def test_pass_detection_on_synthetic_play():
    A, B, C = (10.0, 10.0), (40.0, 10.0), (25.0, 30.0)  # A, B team 0; C team 1
    seq = [
        np.repeat([A], 20, 0),
        flight(A, B, 25),  # A -> B in flight for 1 s
        np.repeat([B], 20, 0),
        flight(B, C, 25),  # B -> C: interception, not a pass
        np.repeat([C], 20, 0),
    ]
    ball = np.concatenate(seq)
    tr = make_tracks(static_players(len(ball), [A, B, C]), [0, 0, 1], ball)
    passes = P.detect_passes(tr, P.ball_owner(tr, 1.0), min_control=2)
    assert len(passes) == 1
    p = passes[0]
    assert (p["from"], p["to"], p["team"]) == (0, 1, 0)
    assert p["start"] == 19 and p["end"] == 45

    truth = [{"start": 21, "team": 0, "from": 0, "to": 1}, {"start": 200, "team": 0, "from": 1, "to": 0}]
    m = P.match_passes(passes, truth, tol_frames=5)
    assert (m["tp"], m["fp"], m["fn"]) == (1, 0, 1)
    assert abs(m["f1"] - 2 / 3) < 1e-9


def test_ball_brushing_past_an_opponent_does_not_break_the_pass():
    A, B, D = (10.0, 10.0), (40.0, 10.0), (25.0, 10.5)  # defender D right on the ball's path
    ball = np.concatenate([np.repeat([A], 20, 0), flight(A, B, 25), np.repeat([B], 20, 0)])
    tr = make_tracks(static_players(len(ball), [A, B, D]), [0, 0, 1], ball)
    own = P.ball_owner(tr, 1.0)
    assert (own == 2).sum() in (1, 2)  # the defender is "in control" for a frame or two
    assert len(P.detect_passes(tr, own, min_control=1)) == 0  # naive: the pass is cut
    assert len(P.detect_passes(tr, own, min_control=3)) == 1  # short control runs ignored


def test_hull_area():
    square = [(0, 0), (10, 0), (10, 10), (0, 10), (5, 5), (2, 7)]
    assert abs(tactics.hull_area(square) - 100.0) < 1e-9
    assert tactics.hull_area([(0, 0), (1, 1)]) == 0.0


@pytest.mark.parametrize("name", list(tactics.FORMATIONS))
def test_formation_matching_recovers_each_template(name):
    pytest.importorskip("scipy")
    rng = np.random.default_rng(1)
    tpl = tactics.FORMATIONS[name]
    avg = tpl * [45.0, 60.0] + [15.0, 4.0] + rng.normal(0, 0.8, tpl.shape)  # metres, shuffled players
    found, costs = tactics.match_formation(avg[rng.permutation(10)])
    assert found == name, costs


def test_formation_from_tracks_with_direction_and_goalkeeper():
    pytest.importorskip("scipy")
    rng = np.random.default_rng(2)
    home = np.vstack([[3.0, 34.0], tactics.FORMATIONS["4-4-2"] * [50.0, 60.0] + [10.0, 4.0]])  # GK + 10, attack -> +x
    away = np.vstack([[102.0, 34.0], 105.0 - (tactics.FORMATIONS["4-3-3"] * [50.0, 60.0] + [10.0, 4.0])])
    T = 50
    xy = static_players(T, np.vstack([home, away])) + rng.normal(0, 0.5, (T, 22, 2))
    tr = Tracks(xy=xy, team=[0] * 11 + [1] * 11, ball=np.full((T, 2), 52.5), fps=FPS)
    assert tactics.attack_sign(tr, 0) == 1 and tactics.attack_sign(tr, 1) == -1
    _, gk = tactics.outfield(tr, 1)
    assert gk == 11
    assert tactics.formation(tr, 0)[0] == "4-4-2"
    assert tactics.formation(tr, 1)[0] == "4-3-3"
    s = tactics.shape_stats(tr, 0)
    assert 18.0 < s["defensive_line_m"] < 22.0  # deepest outfield player at 20 m (template depth 0.2)


def test_noise_model_matches_measured_error():
    rng = np.random.default_rng(0)
    for tau in (0.0, 1.0):
        e = noise.position_noise(4000, 20, FPS, tau, rng)
        mag = np.linalg.norm(e, axis=-1)
        assert abs(np.median(mag) - noise.POS_MEDIAN) < 0.25
        assert abs(np.quantile(mag, 0.9) - noise.POS_P90) < 0.6
        assert abs(e.mean()) < 0.3  # unbiased


def test_inject_missing_frames_and_swaps():
    rng = np.random.default_rng(0)
    T = 3000
    xy = static_players(T, [(10.0, 10.0), (11.0, 10.0), (50.0, 30.0)])
    tr = make_tracks(xy, [0, 1, 1], np.full((T, 2), 20.0))
    t2 = noise.inject(tr, rng, missing_frac=0.2)
    miss = np.isnan(t2.xy[:, :, 0]).mean(0)
    assert np.all((miss > 0.15) & (miss < 0.3))
    t3 = noise.inject(tr, np.random.default_rng(1), swaps_per_min=10.0)
    assert np.isnan(t3.xy).sum() == 0 and t3.xy.shape == tr.xy.shape
    swapped = np.any(t3.xy[:, 0, 0] == 11.0)  # player 0's identity took the neighbour's track at some point
    assert swapped
