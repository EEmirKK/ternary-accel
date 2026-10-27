"""Functional coverage model. Bins are sampled from what the tests drove and what the design
did, collected across every test of one build in a JSON file, and the suite's last test,
test_coverage_complete, fails if any bin was never hit.

The bins cover value extremes and zeros, bias sign, outputs clamped high and low, tile
counts, batch sizes and padding, requantiser corners, input gaps, output stalls and
stalled beats, busy transitions and a reset mid-stream, STATUS reads and bank switches in
the builds that have them, and two crosses: batch size x ReLU, input tiles x batch size.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

import accel_model as am

ROOT = Path(__file__).resolve().parents[1]


def _rng(n, edges):
    for lo, hi, name in edges:
        if lo <= n <= hi:
            return name
    return "other"


B_RANGES = [(1, 1, "B1"), (2, 4, "B2-4"), (5, 7, "B5-7"), (8, 8, "B8")]
K_RANGES = [(1, 1, "kt1"), (2, 2, "kt2"), (3, 5, "kt3-5"), (6, 10_000, "kt6+")]
M_RANGES = [(1, 1, "mt1"), (2, 2, "mt2"), (3, 10_000, "mt3+")]


def required_bins(p: am.AccelParams, perf: bool, double_buf: bool):
    bins = ["act=-128", "act=127", "act=0", "act<0", "act>0",
            "w=0", "w>0", "w<0", "bias<0", "bias>0", "y=127(clamp)", "y=low(clamp)",
            "K_padded", "M_padded", "K_exact", "M_exact",
            "relu=0", "relu=1", "s=0", "s=s_max", "m0=0",
            "in_gap", "out_stall", "stall_BIAS", "stall_CONFIG",
            "busy_0->1", "busy_1->0", "reset_mid_stream"]
    bins += ["w=-128", "w=127"] if not p.ternary else []
    # With int8 and double-buffering a WEIGHT beat can never stall, because the bank it
    # writes was last used at least N weight beats earlier, more than DRAIN = N-1.
    if not (double_buf and not p.ternary):
        bins.append("stall_WEIGHT")
    bins += [b for _, _, b in B_RANGES] + [b for _, _, b in K_RANGES] + [b for _, _, b in M_RANGES]
    bins += [f"{b}x relu={r}" for _, _, b in B_RANGES for r in (0, 1)]
    bins += [f"{k}x{b}" for _, _, k in K_RANGES for _, _, b in B_RANGES]
    if perf:
        bins.append("status_read")
    if double_buf:
        bins.append("bank_switch")
    return bins


def sample_layer(cov, p: am.AccelParams, L: dict, y: np.ndarray):
    W, x, bias = np.asarray(L["W"]), np.asarray(L["x"]), np.asarray(L["bias"])
    K, M = W.shape
    B = x.shape[0]
    hit = cov.setdefault("hits", {})

    def h(name):
        hit[name] = hit.get(name, 0) + 1

    for cond, name in ((x == -128, "act=-128"), (x == 127, "act=127"), (x == 0, "act=0"),
                       (x < 0, "act<0"), (x > 0, "act>0"), (W == 0, "w=0"), (W > 0, "w>0"),
                       (W < 0, "w<0"), (bias < 0, "bias<0"), (bias > 0, "bias>0")):
        if np.any(cond):
            h(name)
    if not p.ternary:
        if np.any(W == -128):
            h("w=-128")
        if np.any(W == 127):
            h("w=127")
    if np.any(y == 127):
        h("y=127(clamp)")
    if np.any(y == (0 if L["relu"] else -128)):
        h("y=low(clamp)")
    h("K_padded" if K % p.N else "K_exact")
    h("M_padded" if M % p.N else "M_exact")
    h(f"relu={int(L['relu'])}")
    if L["s"] == 0:
        h("s=0")
    if L["s"] == p.s_max:
        h("s=s_max")
    if L["m0"] == 0:
        h("m0=0")
    kt, mt = am.pad_to(K, p.N) // p.N, am.pad_to(M, p.N) // p.N
    b, k, m = _rng(B, B_RANGES), _rng(kt, K_RANGES), _rng(mt, M_RANGES)
    for name in (b, k, m, f"{b}x relu={int(L['relu'])}", f"{k}x{b}"):
        h(name)


def sample_counter(cov, name, count=1):
    if count:
        hit = cov.setdefault("hits", {})
        hit[name] = hit.get(name, 0) + count


def cov_file(build: str) -> Path:
    return ROOT / "sim_build" / f"coverage_{build}.json"


def build_id() -> str:
    return "n{}_{}_db{}_perf{}".format(
        os.environ.get("ACCEL_N", "4"), "ternary" if os.environ.get("ACCEL_TERNARY") == "1" else "int8",
        os.environ.get("ACCEL_DB", "0"), os.environ.get("ACCEL_PERF", "0"))


def load(build: str) -> dict:
    f = cov_file(build)
    return json.loads(f.read_text()) if f.exists() else {"hits": {}}


def save(build: str, cov: dict):
    cov_file(build).parent.mkdir(parents=True, exist_ok=True)
    cov_file(build).write_text(json.dumps(cov, indent=1, sort_keys=True))
