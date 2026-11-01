"""Project 15: fall-detection features, alarm logic and scoring protocol on synthetic skeletons.
fall.py imports Ultralytics and creates data folders at import time, so only the pure functions are loaded;
protocol.py (alarm rule, video metrics, threshold choice) is NumPy-only and imported whole."""

import pytest

np = pytest.importorskip("numpy")

from conftest import load_defs, load_module  # noqa: E402

fall = load_defs("15", "fall.py", ["FPS", "L_SH", "_ffill", "features"], {"np": np})
protocol = load_module("15", "protocol.py")
fall.alarm_frames = protocol.alarm_frames
alarm_len = load_defs("15", "detect.py", ["alarm_len"], {"FPS": fall.FPS}).alarm_len

N_FEATURES = 16


def person(t, fall_speed=0.0, fall_start=1.0):
    """Keypoints (17, 3) and box of an upright person whose whole body slides down at `fall_speed` px/s
    after `fall_start` seconds (t in seconds). Frame height 240."""
    dy = fall_speed * max(0.0, t - fall_start)
    kp = np.zeros((17, 3), np.float32)
    kp[:, 2] = 1.0  # confident keypoints
    kp[:, 0] = 120
    kp[:, 1] = 120 + dy
    kp[0] = [120, 50 + dy, 1]  # nose
    kp[5], kp[6] = [110, 70 + dy, 1], [130, 70 + dy, 1]  # shoulders
    kp[11], kp[12] = [112, 130 + dy, 1], [128, 130 + dy, 1]  # hips
    box = np.array([100, 40 + dy, 140, 220 + dy], np.float32)
    return kp, box


def clip(fps, seconds=3.0, **kw):
    ts = np.arange(int(seconds * fps)) / fps
    kps, boxes = zip(*(person(t, **kw) for t in ts))
    return np.stack(kps), np.stack(boxes)


def test_features_shape_and_names():
    kps, boxes = clip(30)
    feats, names = fall.features(kps, boxes, frame_h=240, fps=30)
    assert feats.shape == (len(kps), N_FEATURES)
    assert len(names) == N_FEATURES
    assert np.isfinite(feats).all()


def test_upright_person_has_small_torso_angle():
    kps, boxes = clip(30)
    feats, names = fall.features(kps, boxes, frame_h=240, fps=30)
    angle = feats[:, names.index("torso_angle")]
    assert np.allclose(angle, 0, atol=1)
    assert np.allclose(feats[:, names.index("v_head")], 0)


def test_lying_person_has_horizontal_torso():
    kps, boxes = clip(30, seconds=1)
    kps[:, 5:7, :2] = [[60, 120], [60, 130]]  # shoulders left of the hips, same height
    kps[:, 11:13, :2] = [[120, 120], [120, 130]]
    feats, names = fall.features(kps, boxes, frame_h=240, fps=30)
    assert np.allclose(feats[:, names.index("torso_angle")], 90, atol=1)


def test_velocity_features_do_not_depend_on_frame_rate():
    """Same physical motion (120 px/s downward) filmed at 30 and 60 fps -> same per-second features."""
    k30, b30 = clip(30, fall_speed=120.0)
    k60, b60 = clip(60, fall_speed=120.0)
    f30, names = fall.features(k30, b30, frame_h=240, fps=30)
    f60, _ = fall.features(k60, b60, frame_h=240, fps=60)
    for name in ("v_head", "v_top", "max_v_head_1.5s", "max_v_top_1.5s"):
        j = names.index(name)
        at_2s_30, at_2s_60 = f30[60, j], f60[120, j]  # t = 2.0 s in both clips
        assert at_2s_30 == pytest.approx(120 / 240, rel=1e-3), name  # 0.5 frame heights per second
        assert at_2s_60 == pytest.approx(at_2s_30, rel=1e-3), name


def test_missing_detections_are_filled_and_flagged():
    kps, boxes = clip(30, seconds=1)
    kps[10:15] = np.nan
    boxes[10:15] = np.nan
    feats, names = fall.features(kps, boxes, frame_h=240, fps=30)
    assert np.isfinite(feats).all()
    present = feats[:, names.index("present")]
    assert present[10:15].sum() == 0 and present[:10].all() and present[15:].all()


def test_ffill_forward_then_backward():
    a = np.array([[np.nan], [1.0], [np.nan], [3.0], [np.nan]])
    np.testing.assert_array_equal(fall._ffill(a)[:, 0], [1, 1, 1, 3, 3])


def test_alarm_after_sustained_high_probability():
    prob = [0.1] * 5 + [0.9] * 10
    assert fall.alarm_frames(prob, threshold=0.5, min_frames=10) == 14


def test_alarm_needs_consecutive_frames():
    prob = ([0.9] * 9 + [0.1]) * 3
    assert fall.alarm_frames(prob, threshold=0.5, min_frames=10) is None


def test_alarm_threshold_is_strict():
    assert fall.alarm_frames([0.5] * 20, threshold=0.5, min_frames=3) is None
    assert fall.alarm_frames([0.51] * 3, threshold=0.5, min_frames=3) == 2


def test_alarm_length_scales_with_fps():
    assert alarm_len(30) == 10
    assert alarm_len(60) == 20
    assert alarm_len(1) == 1


ONSETS = {"fall-a": 50, "fall-b": 50, "adl-a": None, "adl-b": None}


def test_video_metrics_counts_and_early_alarm():
    alarms = {"fall-a": 60, "fall-b": 20, "adl-a": None, "adl-b": 30}  # fall-b alarm is > 0.5 s early
    m = protocol.video_metrics(alarms, ONSETS)
    assert (m["TP"], m["FN"], m["FP"], m["TN"]) == (1, 1, 1, 1)
    assert m["sensitivity"] == 0.5 and m["specificity"] == 0.5
    assert m["f1"] == 0.5  # 2*1 / (2*1 + 1 + 1)
    assert m["median_latency_s"] == pytest.approx(10 / 30, abs=0.01)
    assert m["missed_falls"] == ["fall-b"] and m["false_alarm_videos"] == ["adl-b"]


def test_choose_threshold_separates_falls_from_adls():
    T = 100
    probs = {
        "fall-a": np.r_[np.zeros(50), np.full(50, 0.9)],
        "fall-b": np.r_[np.zeros(50), np.full(50, 0.8)],
        "adl-a": np.full(T, 0.6),  # a confident false alarm below 0.6 is filtered only by a higher threshold
        "adl-b": np.zeros(T),
    }
    thr, f1 = protocol.choose_threshold(probs, ONSETS, grid=(0.5, 0.6, 0.7, 0.8, 0.9))
    assert f1 == 1.0
    assert thr in (0.6, 0.7)  # plateau 0.6-0.7 (strict >): median of the tied values
