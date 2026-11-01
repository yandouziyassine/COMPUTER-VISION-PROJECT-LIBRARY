"""Temporal models on raw keypoint sequences: a causal 1D-CNN (TCN) and a GRU, trained on CPU.

Both output a fall probability per frame and only look at the past (causal), like the hand-crafted
window features, so they can run online and use the same alarm rule. NumPy + PyTorch only.
"""

import numpy as np
import torch
from torch import nn

# COCO left/right keypoint pairs, swapped when a sequence is mirrored horizontally
LR_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]


def ffill_time(a):
    """Forward-fill NaNs along axis 0 (missed detections keep the last known value), back-fill the start,
    and replace columns that are NaN everywhere with 0."""
    a = np.asarray(a, np.float32).copy()
    flat = a.reshape(len(a), -1)
    ok = ~np.isnan(flat)
    idx = np.where(ok, np.arange(len(a))[:, None], 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    filled = flat[idx, np.arange(flat.shape[1])]
    first = np.where(ok.any(0), ok.argmax(0), 0)  # back-fill the frames before the first detection
    before = np.arange(len(a))[:, None] < first[None, :]
    filled = np.where(before, flat[first, np.arange(flat.shape[1])], filled)
    return np.nan_to_num(filled).reshape(a.shape)


def mirror(kps, boxes, frame_w=320):
    """Horizontal flip of a keypoint sequence (data augmentation): x -> W - x, left/right swapped."""
    kps, boxes = kps.copy(), boxes.copy()
    kps[..., 0] = frame_w - kps[..., 0]
    for a, b in LR_PAIRS:
        kps[:, [a, b]] = kps[:, [b, a]]
    boxes[:, [0, 2]] = frame_w - boxes[:, [2, 0]]
    return kps, boxes


def sequence_inputs(kps, boxes, frame_h=240, fps=30):
    """Per-frame input vector (T, 94): keypoint x/y (34) and confidences (17), box (4), detection flag (1),
    all in frame heights, plus the per-second velocity of keypoints and box (38). Low-confidence keypoints
    and missed detections are forward-filled, as in the hand-crafted features."""
    T = len(kps)
    present = (~np.isnan(boxes[:, 0])).astype(np.float32)
    xy = kps[..., :2].astype(np.float32).copy()
    xy[~(np.nan_to_num(kps[..., 2]) >= 0.3)] = np.nan
    xy = ffill_time(xy.reshape(T, -1)) / frame_h
    conf = np.nan_to_num(kps[..., 2]).astype(np.float32)
    box = ffill_time(boxes) / frame_h
    pos = np.concatenate([xy, box], axis=1)
    vel = np.zeros_like(pos)
    vel[1:] = (pos[1:] - pos[:-1]) * fps
    return np.concatenate([xy, conf, box, present[:, None], vel], axis=1).astype(np.float32)


class CausalTCN(nn.Module):
    """Stack of dilated causal convolutions with residual connections. Kernel 3 and dilations 1..16
    give a receptive field of 63 frames (2.1 s at 30 fps)."""

    def __init__(self, n_in, ch=32, dilations=(1, 2, 4, 8, 16), k=3, dropout=0.2):
        super().__init__()
        self.inp = nn.Conv1d(n_in, ch, 1)
        self.pads = [(k - 1) * d for d in dilations]
        self.convs = nn.ModuleList(nn.Conv1d(ch, ch, k, dilation=d) for d in dilations)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Conv1d(ch, 1, 1)

    def forward(self, x):  # (B, T, C) -> logits (B, T)
        h = self.inp(x.transpose(1, 2))
        for pad, conv in zip(self.pads, self.convs):
            h = h + self.drop(torch.relu(conv(nn.functional.pad(h, (pad, 0)))))
        return self.head(h)[:, 0]


class GRUNet(nn.Module):
    """One-directional GRU (causal by construction) with a per-frame linear head."""

    def __init__(self, n_in, hidden=48, dropout=0.2):
        super().__init__()
        self.inp = nn.Linear(n_in, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):  # (B, T, C) -> logits (B, T)
        h, _ = self.gru(torch.relu(self.inp(x)))
        return self.head(self.drop(h))[..., 0]


def build(arch, n_in):
    return {"tcn": CausalTCN, "gru": GRUNet}[arch](n_in)


def _windows(seqs, length, rng):
    """One random window of `length` frames per sequence; shorter sequences are zero-padded and masked."""
    xs, ys, ms = [], [], []
    for x, y in seqs:
        T = len(x)
        s = rng.integers(0, T - length + 1) if T > length else 0
        xw, yw = x[s : s + length], y[s : s + length]
        pad = length - len(xw)
        xs.append(np.pad(xw, ((0, pad), (0, 0))))
        ys.append(np.pad(yw, (0, pad)))
        ms.append(np.r_[np.ones(len(xw)), np.zeros(pad)])
    return (torch.tensor(np.stack(a), dtype=torch.float32) for a in (xs, ys, ms))


def train(arch, train_seqs, steps=600, batch=32, length=150, lr=3e-3, noise=0.05, seed=0):
    """train_seqs: list of (inputs (T, C), per-frame target (T,)) already standardised, mirrored copies
    included. Samples random windows (positives of fall videos are always near the onset, so windows
    of 5 s keep most of them), weighted BCE for the rare positive frames. Returns the model in eval mode."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build(arch, train_seqs[0][0].shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    pos = sum(float(y.sum()) for _, y in train_seqs)
    neg = sum(len(y) for _, y in train_seqs) - pos
    pos_weight = torch.tensor(neg / max(pos, 1.0))
    model.train()
    for _ in range(steps):
        pick = rng.choice(len(train_seqs), size=min(batch, len(train_seqs)), replace=False)
        x, y, m = _windows([train_seqs[i] for i in pick], length, rng)
        x = x + noise * torch.randn_like(x)
        loss = nn.functional.binary_cross_entropy_with_logits(model(x), y, pos_weight=pos_weight, reduction="none")
        loss = (loss * m).sum() / m.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return model.eval()


@torch.no_grad()
def predict(model, x):
    """Per-frame fall probability for one full sequence (T, C)."""
    return torch.sigmoid(model(torch.tensor(x, dtype=torch.float32)[None]))[0].numpy()
