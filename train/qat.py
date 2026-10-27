"""Quantisation-aware training of the 784 -> 64 -> 10 MLP, and its export to the integers the
hardware runs.

Every precision uses the same recipe, epochs and seeds, and float is only a reference. int8
uses per-layer symmetric weights, q in [-127, 127] with scale max|W| / 127. Ternary follows
Ternary Weight Networks (Li et al., 2016, arXiv:1605.04711), per layer: weights larger in
magnitude than 0.7 * mean|W| become sign(W), the rest 0, scaled by the mean |W| of the
kept weights. Both use a straight-through estimator, so the
backward pass treats quantisation as the identity. Activations match the hardware: inputs are
pixel >> 1 in [0, 127], and the hidden layer is ReLU, then fake-quantised to [0, 127] with a
scale that tracks the batch maximum during training and is frozen for export.

Accuracy is measured on the exported integers with model/mlp.py, for the hardware's
positive-only output scale, the symmetric rule, and the full-width argmax as a reference.

    python train/qat.py --dataset fashion --precision ternary --seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "train"))
import accel_model as am  # noqa: E402
import data  # noqa: E402
import mlp  # noqa: E402
from mlp import to_input  # noqa: E402

HIDDEN = 64
EPOCHS = 20
BATCH = 128
LR = 1e-3
S_IN = 1.0 / 127.0
TERNARY_DELTA = 0.7    # the threshold factor from Ternary Weight Networks (Li et al., 2016)


def quant_weight(W: torch.Tensor, precision: str):
    """Returns (fake-quantised weight with STE, integer codes, scale)."""
    if precision == "float":
        return W, None, None
    Wd = W.detach()
    if precision == "int8":
        scale = Wd.abs().max().clamp_min(1e-8) / 127.0
        q = torch.clamp(torch.round(Wd / scale), -127, 127)
    elif precision == "ternary":
        delta = TERNARY_DELTA * Wd.abs().mean()
        mask = Wd.abs() > delta
        q = torch.sign(Wd) * mask
        scale = Wd.abs()[mask].mean() if mask.any() else torch.tensor(1e-8)
    else:
        raise ValueError(precision)
    return W + (q * scale - W).detach(), q, scale


class MLP(nn.Module):
    def __init__(self, precision: str):
        super().__init__()
        self.precision = precision
        self.fc1 = nn.Linear(784, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, 10)
        self.register_buffer("h_max", torch.tensor(0.0))

    def forward(self, x):
        W1, _, _ = quant_weight(self.fc1.weight, self.precision)
        h = F.relu(F.linear(x, W1, self.fc1.bias))
        if self.precision != "float":
            if self.training:
                m = h.detach().max()
                if self.h_max > 0:
                    self.h_max.mul_(0.9).add_(0.1 * m)
                else:
                    self.h_max.copy_(m)
            s_h = self.h_max.clamp_min(1e-8) / 127.0
            h = h + (torch.clamp(torch.round(h / s_h), 0, 127) * s_h - h).detach()
        W2, _, _ = quant_weight(self.fc2.weight, self.precision)
        return F.linear(h, W2, self.fc2.bias)


def train(dataset: str, precision: str, seed: int, epochs: int = EPOCHS, log=None):
    log = log or (lambda m: print(m, flush=True))
    torch.manual_seed(seed)
    np.random.seed(seed)
    xtr, ytr, xte, yte = data.load(dataset)
    Xtr = torch.tensor(to_input(xtr), dtype=torch.float32) * S_IN
    Ytr = torch.tensor(ytr, dtype=torch.long)
    model = MLP(precision)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr), generator=g)
        total = 0.0
        for i in range(0, len(Xtr), BATCH):
            idx = perm[i:i + BATCH]
            loss = F.cross_entropy(model(Xtr[idx]), Ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        sched.step()
        log(f"  [{dataset}/{precision}/seed{seed}] epoch {ep + 1}/{epochs} loss {total / len(Xtr):.4f}")
    return model


def to_integer(model: MLP, dataset: str, p: am.AccelParams, method: str = "positive"):
    """Exports the trained model to the integers the hardware runs. Layer 2's M0 and s come
    from the training set by ``method`` (see mlp.layer2_scale). Also returns the training-set
    layer-2 accumulators, so the other methods can be evaluated."""
    assert model.precision in ("int8", "ternary")
    model.eval()
    with torch.no_grad():
        _, q1, sc1 = quant_weight(model.fc1.weight, model.precision)
        _, q2, sc2 = quant_weight(model.fc2.weight, model.precision)
        s_h = float(model.h_max) / 127.0
        sc1, sc2 = float(sc1), float(sc2)
        W1 = q1.T.numpy().astype(np.int64)            # K x M = 784 x 64
        W2 = q2.T.numpy().astype(np.int64)            # 64 x 10
        b1 = np.round(model.fc1.bias.numpy() / (S_IN * sc1)).astype(np.int64)
        b2 = np.round(model.fc2.bias.numpy() / (s_h * sc2)).astype(np.int64)
    # Layer 1's accumulator is in units of S_IN*sc1, and the hidden code in units of s_h.
    m0_1, s_1 = am.choose_scale(S_IN * sc1 / s_h, p.s_max)
    xtr, _, _, _ = data.load(dataset)
    h_tr = am.layer_int(p, W1, to_input(xtr), b1, m0_1, s_1, True)
    acc2_tr = h_tr @ W2 + b2
    assert am.fits_signed(acc2_tr, p.acc_w)
    m0_2, s_2 = mlp.layer2_scale(acc2_tr, method, p.s_max)
    net = dict(W1=W1, b1=b1, m0_1=m0_1, s_1=s_1, W2=W2, b2=b2, m0_2=m0_2, s_2=s_2,
               s_h=s_h, scale1=sc1, scale2=sc2)
    return net, acc2_tr


def accuracy_float(model: MLP, pixels: np.ndarray, labels: np.ndarray) -> float:
    model.eval()
    with torch.no_grad():
        x = torch.tensor(to_input(pixels), dtype=torch.float32) * S_IN
        return float((model(x).argmax(1).numpy() == labels).mean())


def evaluate_methods(p: am.AccelParams, net: dict, acc2_tr, xte, yte) -> dict:
    """Integer test accuracy, ties and top-logit saturation for each layer-2 method."""
    _, h_te = mlp.int_forward(p, net, xte, return_hidden=True)
    acc2_te = h_te @ net["W2"] + net["b2"]
    out = {}
    for method in ("positive", "spec"):
        m0, s = mlp.layer2_scale(acc2_tr, method, p.s_max)
        logits = am.layer_int(p, net["W2"], h_te, net["b2"], m0, s, False)
        acc, ties = mlp.accuracy_int(logits, yte)
        out[method] = dict(acc=acc, ties=ties, m0_2=m0, s_2=s,
                           top_saturated_test=mlp.top_saturated(acc2_te, m0, s),
                           top_saturated_train=mlp.top_saturated(acc2_tr, m0, s))
    acc, ties = mlp.accuracy_int(acc2_te, yte)
    out["raw"] = dict(acc=acc, ties=ties)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mnist", "fashion"], required=True)
    ap.add_argument("--precision", choices=["float", "int8", "ternary"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--export", type=Path, help="write the integer network (.npz, positive-only scale)")
    ap.add_argument("--json", type=Path, help="append a result record (JSON lines)")
    args = ap.parse_args()

    torch.set_num_threads(4)
    t0 = time.time()
    model = train(args.dataset, args.precision, args.seed, args.epochs)
    _, _, xte, yte = data.load(args.dataset)
    rec = dict(dataset=args.dataset, precision=args.precision, seed=args.seed,
               epochs=args.epochs, acc_float_eval=accuracy_float(model, xte, yte))
    if args.precision != "float":
        p = am.AccelParams(ternary=args.precision == "ternary")
        net, acc2_tr = to_integer(model, args.dataset, p, "positive")
        rec.update(methods=evaluate_methods(p, net, acc2_tr, xte, yte),
                   m0_1=net["m0_1"], s_1=net["s_1"],
                   w1_zero_frac=float((net["W1"] == 0).mean()),
                   w2_zero_frac=float((net["W2"] == 0).mean()))
        if args.export:
            args.export.parent.mkdir(parents=True, exist_ok=True)
            np.savez(args.export, **net)
    rec["train_seconds"] = round(time.time() - t0, 1)
    print(json.dumps(rec))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
