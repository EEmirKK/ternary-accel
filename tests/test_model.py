"""Pure-Python checks of the golden model (no simulator)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model"))
import accel_model as am  # noqa: E402


def numpy_reference(W, x, bias, m0, s, relu):
    """The layer written directly in NumPy: x @ W + b, then round, shift and clamp."""
    acc = np.asarray(x, dtype=np.int64) @ np.asarray(W, dtype=np.int64) + bias
    t = (acc * m0 + ((1 << (s - 1)) if s else 0)) >> s
    return np.clip(t, 0 if relu else -128, 127)


@pytest.mark.parametrize("ternary", [False, True], ids=["int8", "ternary"])
def test_layer_tiled_matches_numpy_reference(ternary):
    """The tiled model, which follows the hardware's tiles and widths, must equal a plain NumPy layer."""
    p = am.AccelParams(ternary=ternary)
    rng = np.random.default_rng(1)
    for _ in range(200):
        K = int(rng.integers(1, 200))
        M = int(rng.integers(1, 30))
        B = int(rng.integers(1, p.B_MAX + 1))
        W = rng.choice([-1, 0, 1], size=(K, M)) if ternary else rng.integers(-128, 128, size=(K, M))
        x = rng.integers(-128, 128, size=(B, K))
        bias = rng.integers(-5000, 5000, size=M)
        relu = bool(rng.integers(0, 2))
        m0, s = int(rng.integers(0, 1 << 16)), int(rng.integers(0, p.s_max + 1))
        y_t, _ = am.layer_tiled(p, W, x, bias, m0, s, relu)
        assert (y_t == numpy_reference(W, x, bias, m0, s, relu)).all()


@pytest.mark.parametrize("ternary", [False, True], ids=["int8", "ternary"])
def test_layer_int_matches_layer_tiled(ternary):
    """The fast path used for 10k-image accuracy must equal the tiled model."""
    p = am.AccelParams(ternary=ternary)
    rng = np.random.default_rng(0)
    for _ in range(200):
        K = int(rng.integers(1, 200))
        M = int(rng.integers(1, 30))
        B = int(rng.integers(1, p.B_MAX + 1))
        W = rng.choice([-1, 0, 1], size=(K, M)) if ternary else rng.integers(-128, 128, size=(K, M))
        x = rng.integers(-128, 128, size=(B, K))
        bias = rng.integers(-5000, 5000, size=M)
        relu = bool(rng.integers(0, 2))
        m0, s = int(rng.integers(0, 1 << 16)), int(rng.integers(0, p.s_max + 1))
        y_t, _ = am.layer_tiled(p, W, x, bias, m0, s, relu)
        y_f = am.layer_int(p, W, x, bias, m0, s, relu)
        assert (y_t == y_f).all()
    # worst case at K_MAX
    W = np.full((p.K_MAX, p.N), -1 if ternary else -128)
    x = np.full((3, p.K_MAX), -128)
    y_t, _ = am.layer_tiled(p, W, x, np.zeros(p.N, int), 1, 20, False)
    assert (am.layer_int(p, W, x, np.zeros(p.N, int), 1, 20, False) == y_t).all()


def test_layer_int_detects_overflow():
    p = am.AccelParams(ternary=True)
    W = np.ones((p.K_MAX, 4), dtype=int)
    x = np.full((1, p.K_MAX), 127)
    with pytest.raises(AssertionError, match="accumulator overflow"):
        am.layer_int(p, W, x, np.full(4, (1 << 18) - 1), 1, 0, False)
