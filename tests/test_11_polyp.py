"""Project 11: flip TTA, screening metrics and validation threshold choice (no model weights, no data)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from conftest import load_defs  # noqa: E402

p = load_defs(
    "11",
    "polyp.py",
    ["TTA_FLIPS", "predict_probs", "dice_iou", "screening_metrics", "f_beta", "pick_threshold"],
    {"torch": torch, "np": np},
)


class Shift(torch.nn.Module):
    """A fake 'model' whose logit is the first channel: not flip-equivariant unless the input is."""

    def forward(self, x):
        return x[:, :1] * 4 - 2


def test_tta_equals_plain_on_flip_symmetric_input():
    x = torch.zeros(1, 3, 8, 8)
    x[:, :, 2:6, 2:6] = 1.0  # centered square: identical under every flip
    assert torch.allclose(p.predict_probs(Shift(), x, tta=True), p.predict_probs(Shift(), x, tta=False), atol=1e-6)


def test_tta_maps_predictions_back_to_the_original_frame():
    x = torch.zeros(1, 3, 8, 8)
    x[:, :, 0, 0] = 1.0  # one corner pixel; a flip bug would smear it to other corners
    probs = p.predict_probs(Shift(), x, tta=True)
    assert probs[0, 0, 0, 0] > 0.8
    assert probs[0, 0, 7, 7] < 0.2 and probs[0, 0, 0, 7] < 0.2 and probs[0, 0, 7, 0] < 0.2


def test_dice_iou_perfect_and_disjoint():
    target = torch.zeros(2, 1, 4, 4)
    target[0, 0, :2] = 1
    target[1, 0, :2] = 1
    probs = target.clone()
    probs[1] = 1 - target[1]  # second image predicts exactly the complement
    dice, iou = p.dice_iou(probs, target)
    assert dice[0] == pytest.approx(1.0) and iou[0] == pytest.approx(1.0)
    assert dice[1] == pytest.approx(0.0, abs=1e-5) and iou[1] == pytest.approx(0.0, abs=1e-5)


def test_screening_metrics_counts():
    target = torch.zeros(1, 1, 4, 4)
    target[0, 0, 0, :] = 1  # 4 polyp pixels
    probs = torch.zeros(1, 1, 4, 4)
    probs[0, 0, 0, :2] = 0.9  # 2 found
    probs[0, 0, 3, :2] = 0.9  # 2 false alarms
    m = p.screening_metrics(probs, target, 0.5)
    assert m["pixel_sensitivity"] == pytest.approx(0.5)
    assert m["pixel_precision"] == pytest.approx(0.5)
    assert m["false_alarm_area"] == pytest.approx(2 / 16)
    assert m["missed_polyps"] == 0.0  # exactly half the polyp found: not counted as missed


def test_lower_threshold_never_lowers_sensitivity():
    g = torch.Generator().manual_seed(0)
    probs = torch.rand(3, 1, 16, 16, generator=g)
    target = (torch.rand(3, 1, 16, 16, generator=g) > 0.7).float()
    sens = [p.screening_metrics(probs, target, t)["pixel_sensitivity"] for t in (0.8, 0.5, 0.2)]
    assert sens[0] <= sens[1] <= sens[2]


def test_f_beta_weights_recall():
    assert p.f_beta(0.5, 0.5) == pytest.approx(0.5)
    assert p.f_beta(0.4, 0.9) > p.f_beta(0.9, 0.4)
    assert p.f_beta(0.0, 0.0) == 0.0


def test_pick_threshold_prefers_low_threshold_when_polyps_are_underconfident():
    target = torch.zeros(1, 1, 10, 10)
    target[0, 0, :5] = 1
    probs = torch.full((1, 1, 10, 10), 0.1)
    probs[0, 0, :5] = 0.3  # the model is right but timid: 0.5 would miss everything
    thr, rows = p.pick_threshold(probs, target, [0.2, 0.5, 0.8])
    assert thr == 0.2
    assert len(rows) == 3 and all("f_beta" in r for r in rows)
