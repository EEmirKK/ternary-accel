"""cocotb tests for accel_top, with every output checked bit-exact against model/accel_model.py.
They run through tb/test_runner.py, which builds each variant.
"""

from __future__ import annotations

import json

import numpy as np

import cocotb
from cocotb.triggers import ClockCycles

from accel_tb import DOUBLE_BUF, PERF, AccelTB, am, auto_scale, fcov, rand_acts, rand_weights


def make_layer(tb, rng, K, M, B, wkind="uniform", xkind="uniform", relu=None, bias_scale=None):
    p = tb.p
    W = rand_weights(rng, p, K, M, wkind)
    x = rand_acts(rng, B, K, xkind)
    if bias_scale is None:
        bias_scale = (128 if p.ternary else 16384) * max(1, K // 8)
    bias = rng.integers(-bias_scale, bias_scale + 1, size=M)
    if relu is None:
        relu = bool(rng.integers(0, 2))
    m0, s = auto_scale(p, W, x, bias, relu)
    return dict(W=W, x=x, bias=bias, m0=m0, s=s, relu=relu)


@cocotb.test()
async def test_single_tile(dut):
    """One N x N tile, one image, no stalls."""
    tb = AccelTB(dut)
    await tb.start()
    rng = np.random.default_rng(1)
    N = tb.p.N
    await tb.run_layers([make_layer(tb, rng, N, N, 1, relu=False)])


@cocotb.test()
async def test_constrained_random(dut):
    """Random K, M, B, value distributions, ReLU, input gaps and output backpressure."""
    tb = AccelTB(dut)
    await tb.start()
    rng = np.random.default_rng(cocotb.RANDOM_SEED)
    p = tb.p
    kinds = ["uniform", "sparse", "extreme"]
    for it in range(40):
        tb.gap_prob = float(rng.choice([0.0, 0.2, 0.6]))
        tb.stall_prob = float(rng.choice([0.0, 0.3, 0.8]))
        K = int(rng.integers(1, 6 * p.N + 1))
        M = int(rng.integers(1, 5 * p.N + 1))
        B = int(rng.integers(1, p.B_MAX + 1))
        L = make_layer(tb, rng, K, M, B, wkind=str(rng.choice(kinds)),
                       xkind=str(rng.choice(kinds + ["relu"])))
        await tb.run_layers([L], idle_after=2)
    tb.log_coverage()


@cocotb.test()
async def test_extremes_full_k(dut):
    """K = K_MAX with worst-case operands: the largest positive and negative
    accumulators the widths were sized for (checks no overflow)."""
    tb = AccelTB(dut)
    await tb.start()
    p = tb.p
    K, M, B = p.K_MAX, p.N, 2
    wmin = -1 if p.ternary else -128
    wmax = 1 if p.ternary else 127
    # image 0: (-128)*(wmin) everywhere -> max positive; image 1: (-128)*wmax -> max negative
    W = np.full((K, M), wmin)
    W[:, 1] = wmax
    x = np.full((B, K), -128)
    x[1, :] = 127
    bias = np.resize(np.array([0, 0, (1 << (p.acc_w - 3)), -(1 << (p.acc_w - 3))]), M)
    acc, stats = am.layer_acc(p, W, x, bias)
    dut._log.info(f"max |acc| = {stats['max_acc']} vs ACC_W={p.acc_w} limit {(1 << (p.acc_w - 1)) - 1}")
    for relu in (False, True):
        m0, s = am.choose_scale(100.0 / stats["max_acc"], p.s_max)
        await tb.run_layers([dict(W=W, x=x, bias=bias, m0=m0, s=s, relu=relu)])


@cocotb.test()
async def test_all_min(dut):
    """Every operand at its minimum (-128 activations, minimum weights)."""
    tb = AccelTB(dut)
    await tb.start()
    p = tb.p
    K, M, B = 3 * p.N, 2 * p.N, 3
    W = np.full((K, M), -1 if p.ternary else -128)
    x = np.full((B, K), -128)
    bias = np.full(M, -5)
    m0, s = am.choose_scale(1 / 64, p.s_max)
    await tb.run_layers([dict(W=W, x=x, bias=bias, m0=m0, s=s, relu=False)])


@cocotb.test()
async def test_max_positive(dut):
    """127 activations with maximum weights; outputs saturate at +127."""
    tb = AccelTB(dut)
    await tb.start()
    p = tb.p
    K, M, B = 2 * p.N, p.N, 2
    W = np.full((K, M), 1 if p.ternary else 127)
    x = np.full((B, K), 127)
    bias = np.zeros(M, dtype=int)
    await tb.run_layers([dict(W=W, x=x, bias=bias, m0=1 << 15, s=15, relu=False)])


@cocotb.test()
async def test_zeros(dut):
    """Zero weights and zero activations: outputs are just requant(bias)."""
    tb = AccelTB(dut)
    await tb.start()
    p = tb.p
    rng = np.random.default_rng(3)
    K, M, B = 2 * p.N, 2 * p.N, 2
    for W_zero, x_zero in ((True, True), (True, False), (False, True)):
        L = make_layer(tb, rng, K, M, B, relu=False)
        if W_zero:
            L["W"] = np.zeros_like(L["W"])
        if x_zero:
            L["x"] = np.zeros_like(L["x"])
        L["bias"] = rng.integers(-1000, 1000, size=M)
        L["m0"], L["s"] = am.choose_scale(1 / 8, p.s_max)
        await tb.run_layers([L])


@cocotb.test()
async def test_identity(dut):
    """W = I, bias 0, scale 1: the output equals the input."""
    tb = AccelTB(dut)
    await tb.start()
    p = tb.p
    rng = np.random.default_rng(4)
    K = M = 2 * p.N
    B = p.B_MAX
    W = np.eye(K, dtype=int)
    x = rng.integers(-128, 128, size=(B, K))
    y = await tb.run_layers([dict(W=W, x=x, bias=np.zeros(M, dtype=int), m0=1 << 15, s=15, relu=False)])
    assert (y[0] == x).all()


@cocotb.test()
async def test_padding(dut):
    """K and M not multiples of N (layer 2 shape: M = 10)."""
    tb = AccelTB(dut)
    await tb.start()
    rng = np.random.default_rng(5)
    N = tb.p.N
    for K, M in ((N + 1, 10), (2 * N - 1, N + 1), (1, 1), (3, 2 * N + 3)):
        await tb.run_layers([make_layer(tb, rng, K, M, 3)])


@cocotb.test()
async def test_batch_extremes(dut):
    """B = 1 and B = B_MAX, with and without backpressure."""
    tb = AccelTB(dut)
    await tb.start()
    rng = np.random.default_rng(6)
    p = tb.p
    for stall in (0.0, 0.5):
        tb.stall_prob = stall
        tb.gap_prob = stall / 2
        for B in (1, p.B_MAX):
            await tb.run_layers([make_layer(tb, rng, 3 * p.N, 2 * p.N, B)])


@cocotb.test()
async def test_back_to_back_layers(dut):
    """Several layers streamed with no idle time between them, each with its own
    CONFIG (different M0, s, ReLU and B), plus an MLP-style layer 1 -> layer 2 chain."""
    tb = AccelTB(dut)
    await tb.start()
    rng = np.random.default_rng(7)
    p = tb.p
    layers = [make_layer(tb, rng, int(rng.integers(1, 4 * p.N)), int(rng.integers(1, 3 * p.N)),
                         int(rng.integers(1, p.B_MAX + 1))) for _ in range(5)]
    await tb.run_layers(layers)

    # chain: hidden layer with ReLU feeds the output layer (as the MLP will)
    B = 4
    L1 = make_layer(tb, rng, 5 * p.N, 16, B, xkind="relu", relu=True)
    y1 = await tb.run_layers([L1])
    W2 = rand_weights(rng, p, 16, 10)
    bias2 = rng.integers(-500, 500, size=10)
    m0, s = auto_scale(p, W2, y1[0], bias2, False)
    await tb.run_layers([dict(W=W2, x=y1[0], bias=bias2, m0=m0, s=s, relu=False)])


@cocotb.test()
async def test_requant_corners(dut):
    """Shift 0 and maximum shift, M0 = 0 and 65535, ReLU on and off."""
    tb = AccelTB(dut)
    await tb.start()
    rng = np.random.default_rng(8)
    p = tb.p
    for m0, s in ((0, 0), (1, 0), (65535, 0), (65535, p.s_max), (1, p.s_max),
                  (65535, p.acc_w), (12345, 1), (32768, 16)):
        for relu in (False, True):
            L = make_layer(tb, rng, 2 * p.N, p.N, 2, xkind="extreme", wkind="extreme", relu=relu)
            L["m0"], L["s"] = m0, s
            await tb.run_layers([L], idle_after=2)


@cocotb.test()
async def test_reset_mid_computation(dut):
    """Reset while beats are in flight, then a fresh layer must compute correctly."""
    tb = AccelTB(dut)
    await tb.start()
    rng = np.random.default_rng(9)
    p = tb.p
    for wait in (3, 2 * p.N, 25, 60):
        L = make_layer(tb, rng, 4 * p.N, 3 * p.N, p.B_MAX)
        beats = am.encode_layer(p, L["W"], L["x"], L["bias"], L["m0"], L["s"], L["relu"])
        task = cocotb.start_soon(tb.send(beats))
        await ClockCycles(dut.clk, wait)
        task.cancel()
        await tb.reset()
        await tb.run_layers([make_layer(tb, rng, 2 * p.N, 2 * p.N, 2)])


@cocotb.test(skip=not PERF)
async def test_perf_counters(dut):
    """STATUS returns exactly the cycles between two STATUS beats and the number of
    ACT beats accepted in between, then clears the counters."""
    tb = AccelTB(dut)
    await tb.start()
    rng = np.random.default_rng(10)
    p = tb.p
    await tb.read_status()                     # the first STATUS clears the counters
    for gap, stall in ((0.0, 0.0), (0.3, 0.5)):
        tb.gap_prob, tb.stall_prob = gap, stall
        _, _, c0 = await tb.read_status()
        layers = [make_layer(tb, rng, int(rng.integers(1, 4 * p.N)), int(rng.integers(1, 3 * p.N)),
                             int(rng.integers(1, p.B_MAX + 1))) for _ in range(3)]
        n_act = sum(sum(1 for t, _, _ in am.encode_layer(p, L["W"], L["x"], L["bias"], L["m0"], L["s"],
                                                             L["relu"]) if t == am.T_ACT) for L in layers)
        await tb.run_layers(layers, idle_after=5)
        total, acts, c1 = await tb.read_status()
        assert acts == n_act, f"ACT counter {acts}, sent {n_act}"
        assert total == c1 - c0, f"cycle counter {total}, testbench counted {c1 - c0}"
        dut._log.info(f"counters: {total} cycles, {acts} ACT beats (gap {gap}, stall {stall})")


@cocotb.test(skip=not DOUBLE_BUF)
async def test_double_buffer_no_weight_stall(dut):
    """With two weight banks and B > DRAIN, a WEIGHT beat never waits for the
    previous tile's activations to drain (single-buffered builds always do)."""
    tb = AccelTB(dut)
    await tb.start()
    rng = np.random.default_rng(11)
    p = tb.p
    L = make_layer(tb, rng, 6 * p.N, 2 * p.N, p.B_MAX)
    await tb.run_layers([L])
    assert tb.cov.get(f"in_stall_type_{am.T_WEIGHT}", 0) == 0, \
        f"{tb.cov[f'in_stall_type_{am.T_WEIGHT}']} WEIGHT beats stalled"


@cocotb.test()
async def test_cross_coverage(dut):
    """Directed closure of the k-tile x batch-size cross: every range of each, with
    light input gaps and output stalls."""
    tb = AccelTB(dut)
    await tb.start()
    rng = np.random.default_rng(12)
    p = tb.p
    tb.gap_prob, tb.stall_prob = 0.1, 0.3
    for kt in (1, 2, 4, 7):
        for B in (1, 3, 6, p.B_MAX):
            await tb.run_layers([make_layer(tb, rng, kt * p.N, p.N, B)], idle_after=2)


@cocotb.test()
async def test_coverage_complete(dut):
    """Runs last: every functional-coverage bin must have been hit by the tests above."""
    tb = AccelTB(dut)
    bid = fcov.build_id()
    cov = fcov.load(bid)
    req = fcov.required_bins(tb.p, PERF, DOUBLE_BUF)
    hits = cov.get("hits", {})
    missing = [b for b in req if hits.get(b, 0) == 0]
    dut._log.info(f"functional coverage {bid}: {len(req) - len(missing)}/{len(req)} bins hit")
    report = dict(build=bid, required=len(req), hit=len(req) - len(missing), missing=missing,
                  hits={b: hits.get(b, 0) for b in req})
    fcov.cov_file(bid).with_name(f"coverage_report_{bid}.json").write_text(json.dumps(report, indent=1))
    assert not missing, f"coverage bins never hit: {missing}"
