"""Bit-exact integer model of the 784 -> 64 -> 10 MLP, computed the way the accelerator does it.

It has no PyTorch dependency, so the testbench and the FPGA host script can use it too.
A network is the dict that train/qat.py exports (results/accuracy/nets/*.npz).
"""

from __future__ import annotations

import numpy as np

import accel_model as am


def to_input(pixels: np.ndarray) -> np.ndarray:
    """8-bit pixels -> int8 activations in [0, 127], as the hardware receives them."""
    return (np.asarray(pixels) >> 1).astype(np.int64)


def load_net(path) -> dict:
    z = np.load(path)
    net = {k: z[k] for k in z.files}
    for k in ("m0_1", "s_1", "m0_2", "s_2"):
        net[k] = int(net[k])
    return net


def layers(net: dict):
    return (dict(W=net["W1"], bias=net["b1"], m0=net["m0_1"], s=net["s_1"], relu=True),
            dict(W=net["W2"], bias=net["b2"], m0=net["m0_2"], s=net["s_2"], relu=False))


def int_forward(p: am.AccelParams, net: dict, pixels: np.ndarray, return_hidden=False):
    """Returns the int8 logits, and the int8 hidden layer if asked."""
    L1, L2 = layers(net)
    h = am.layer_int(p, L1["W"], to_input(pixels), L1["bias"], L1["m0"], L1["s"], True)
    y = am.layer_int(p, L2["W"], h, L2["bias"], L2["m0"], L2["s"], False)
    return (y, h) if return_hidden else y


def accuracy_int(logits: np.ndarray, labels: np.ndarray):
    """Argmax accuracy, where a tie for the top logit counts as an error.
    Returns (accuracy, number of tied images)."""
    top = logits.max(axis=1, keepdims=True)
    n_top = (logits == top).sum(axis=1)
    correct = (logits.argmax(axis=1) == labels) & (n_top == 1)
    return float(correct.mean()), int((n_top > 1).sum())


# The layer-2 output scale (M0, s) is the host's choice, and only the argmax of the
# logits matters. Both rules are fitted on the training set. "positive", which the
# hardware uses, maps the largest positive logit to 127; very negative logits may then
# clamp at -128, which cannot change the argmax. "spec" is the symmetric rule, mapping
# the largest |logit| to 127.

def unclamped(acc, m0: int, s: int):
    """The requantiser's result before the clamp, (acc*M0 + 2^(s-1)) >> s."""
    acc = np.asarray(acc, dtype=object)
    return (acc * m0 + ((1 << (s - 1)) if s else 0)) >> s


def layer2_scale(acc2_train: np.ndarray, method: str, s_max: int):
    peak = int(acc2_train.max()) if method == "positive" else int(np.abs(acc2_train).max())
    assert peak > 0
    m0, s = am.choose_scale(127.0 / peak, s_max)
    # choose_scale rounds M0, which can push the peak just past the int8 range.
    while unclamped(peak, m0, s) > 127 or (method == "spec" and unclamped(-peak, m0, s) < -128):
        m0 -= 1
    return m0, s


def top_saturated(acc2: np.ndarray, m0: int, s: int) -> int:
    """Number of images whose top logit clips at 127."""
    return int((unclamped(acc2.max(axis=1), m0, s) > 127).sum())
