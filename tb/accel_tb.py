"""cocotb testbench parts for accel_top: the input driver, output sink, protocol checks and
coverage sampling.

Everything drives just after a rising edge and samples in ReadOnly, where this cycle's
values have settled, so a transfer happens at the next edge if valid and ready were both
high in ReadOnly.
"""

from __future__ import annotations

import os
import random
import sys
from collections import Counter
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, ReadOnly, RisingEdge

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model"))
import accel_model as am  # noqa: E402
import coverage as fcov  # noqa: E402  (tb/coverage.py, not the coverage package)

# Clock period in ps. Gate-level power runs set it to the LibreLane constraint,
# so the VCD's toggle rates are at the frequency the power is reported at.
CLK_PS = int(os.environ.get("ACCEL_CLK_PS", "10000"))
# With ACCEL_GL=1 the DUT is a post-route netlist, where internal RTL signals don't exist.
GATE_LEVEL = os.environ.get("ACCEL_GL", "0") == "1"


PERF = os.environ.get("ACCEL_PERF", "0") == "1"
DOUBLE_BUF = os.environ.get("ACCEL_DB", "0") == "1"


def params_from_env() -> am.AccelParams:
    return am.AccelParams(
        N=int(os.environ.get("ACCEL_N", "4")),
        ternary=os.environ.get("ACCEL_TERNARY", "0") == "1",
        B_MAX=int(os.environ.get("ACCEL_B_MAX", "8")),
        K_MAX=int(os.environ.get("ACCEL_K_MAX", "784")),
    )


class ProtocolError(AssertionError):
    pass


class InputGrammar:
    """Checks that the accepted input beats follow the host tile loop: an optional CONFIG,
    then for each output tile its BIAS beats, then for each input tile its WEIGHT beats
    and B ACT beats, with in_last only on the last ACT of the last input tile."""

    def __init__(self, p: am.AccelParams):
        self.p = p
        self.prev_type = None
        self.run = 0          # length of the current run of prev_type
        self.B = 1

    def accept(self, typ: int, data: int, last: bool):
        p = self.p
        if PERF and typ == am.T_CONFIG and (data >> 24) & 0xFF == 0:
            # STATUS request: legal between layers, doesn't change B or the run state
            if self.prev_type in (am.T_BIAS, am.T_WEIGHT):
                raise ProtocolError("STATUS in the middle of a tile")
            return
        if last and typ != am.T_ACT:
            raise ProtocolError("in_last on a non-ACT beat")
        run = self.run + 1 if typ == self.prev_type else 1
        if self.prev_type is not None and typ != self.prev_type:
            # the run that just ended must be complete
            need = {am.T_BIAS: p.bias_beats, am.T_WEIGHT: p.weight_beats,
                    am.T_ACT: self.B, am.T_CONFIG: 1}[self.prev_type]
            if self.run != need:
                raise ProtocolError(f"type {self.prev_type} run of {self.run}, expected {need}")
            legal_next = {
                am.T_CONFIG: {am.T_BIAS},
                am.T_BIAS: {am.T_WEIGHT},
                am.T_WEIGHT: {am.T_ACT},
                am.T_ACT: {am.T_WEIGHT, am.T_BIAS, am.T_CONFIG},
            }[self.prev_type]
            if typ not in legal_next:
                raise ProtocolError(f"type {typ} after {self.prev_type}")
        if typ == am.T_CONFIG:
            # checked after the previous ACT run, which used the old B
            self.B = (data >> 24) & 0xFF
            if not 1 <= self.B <= p.B_MAX:
                raise ProtocolError(f"CONFIG with B={self.B}")
            if not 0 <= (data >> 16) & 0x3F <= p.s_max:
                raise ProtocolError("CONFIG with illegal shift")
        if typ == am.T_ACT and last and run != self.B:
            raise ProtocolError("in_last not on the final image of the batch")
        self.prev_type, self.run = typ, run

    def reset(self):
        self.prev_type, self.run = None, 0


class AccelTB:
    def __init__(self, dut, seed: int | None = None):
        self.dut = dut
        self.p = params_from_env()
        self.rng = random.Random(seed if seed is not None else cocotb.RANDOM_SEED)
        self.gap_prob = 0.0       # probability of an in_valid bubble before a beat
        self.stall_prob = 0.0     # probability out_ready is low in a cycle
        self.received: list[int] = []
        self.grammar = InputGrammar(self.p)
        self.cov = Counter()
        self.cov_merged = Counter()   # part of self.cov already merged into the coverage file
        self.cycle = 0
        self.first_in_cycle = None    # cycle the first input beat was accepted
        self.last_out_cycle = None    # cycle the latest output beat was accepted
        self.last_accept_cycle = None  # cycle the latest input beat was accepted
        self._monitors_started = False

    async def start(self):
        d = self.dut
        d.in_valid.value = 0
        d.in_data.value = 0
        d.in_type.value = 0
        d.in_last.value = 0
        d.out_ready.value = 0
        d.rst_n.value = 0
        if not self._monitors_started:
            cocotb.start_soon(Clock(d.clk, CLK_PS, unit="ps").start())
            cocotb.start_soon(self._sink())
            cocotb.start_soon(self._checker())
            self._monitors_started = True
        await ClockCycles(d.clk, 3)
        d.rst_n.value = 1
        await RisingEdge(d.clk)

    async def reset(self, cycles: int = 3):
        """Assert reset (possibly mid-computation)."""
        d = self.dut
        d.rst_n.value = 0
        d.in_valid.value = 0
        await ClockCycles(d.clk, cycles)
        d.rst_n.value = 1
        self.grammar.reset()
        await RisingEdge(d.clk)
        self.received.clear()
        self.cov["reset_mid_stream"] += 1

    async def send(self, beats):
        d = self.dut
        for typ, data, last in beats:
            while self.gap_prob and self.rng.random() < self.gap_prob:
                d.in_valid.value = 0
                self.cov["in_gap"] += 1
                await RisingEdge(d.clk)
            d.in_valid.value = 1
            d.in_type.value = typ
            d.in_data.value = data
            d.in_last.value = int(last)
            stalled = 0
            while True:
                await ReadOnly()
                ok = int(d.in_ready.value) == 1
                await RisingEdge(d.clk)
                if ok:
                    break
                stalled += 1
            self.grammar.accept(typ, data, last)
            self.last_accept_cycle = self.cycle
            if self.first_in_cycle is None:
                self.first_in_cycle = self.cycle
            self.cov[f"in_type_{typ}"] += 1
            if stalled:
                self.cov[f"in_stall_type_{typ}"] += 1
        d.in_valid.value = 0
        d.in_last.value = 0

    async def _sink(self):
        d = self.dut
        held = None
        while True:
            await RisingEdge(d.clk)
            d.out_ready.value = 0 if (self.stall_prob and self.rng.random() < self.stall_prob) else 1
            await ReadOnly()
            if int(d.rst_n.value) == 0:
                held = None
                continue
            v = int(d.out_valid.value)
            if held is not None:
                # once valid, out_valid and out_data must hold until accepted
                if not v or int(d.out_data.value) != held:
                    raise ProtocolError("out_valid dropped or out_data changed while stalled")
            if v and int(d.out_ready.value):
                self.received.append(int(d.out_data.value))
                self.last_out_cycle = self.cycle
                held = None
            elif v:
                held = int(d.out_data.value)
                self.cov["out_stall"] += 1
            else:
                held = None

    async def _checker(self):
        d = self.dut
        p = self.p
        prev_busy = 0
        while True:
            await RisingEdge(d.clk)
            await ReadOnly()
            self.cycle += 1
            if int(d.rst_n.value) == 0:
                continue
            for name in ("in_ready", "out_valid"):
                if not getattr(d, name).value.is_resolvable:
                    raise ProtocolError(f"{name} is X/Z after reset")
            if GATE_LEVEL:
                continue
            cfg_b = int(d.cfg_b.value)
            if not 1 <= cfg_b <= p.B_MAX:
                raise ProtocolError(f"cfg_b={cfg_b} illegal")
            if int(d.act_b.value) >= cfg_b:
                raise ProtocolError("act_b out of range")
            if int(d.rq_b.value) >= cfg_b:
                raise ProtocolError("rq_b out of range")
            if int(d.w_beat.value) >= p.weight_beats:
                raise ProtocolError("w_beat out of range")
            if int(d.bias_beat.value) >= p.bias_beats:
                raise ProtocolError("bias_beat out of range")
            if DOUBLE_BUF:
                ab = int(d.act_bank.value)
                if ab != getattr(self, "_prev_bank", ab):
                    self.cov["bank_switch"] += 1
                self._prev_bank = ab
            busy = int(d.busy.value)
            if int(d.out_valid.value) and not busy:
                raise ProtocolError("out_valid while not busy")
            if busy != prev_busy:
                self.cov[f"busy_{prev_busy}->{busy}"] += 1
            prev_busy = busy

    async def wait_outputs(self, n: int, timeout_cycles: int):
        start = self.cycle
        while len(self.received) < n:
            await RisingEdge(self.dut.clk)
            if self.cycle - start > timeout_cycles:
                raise AssertionError(
                    f"timeout: got {len(self.received)} of {n} output beats")

    async def run_layers(self, layers, idle_after: int = 20):
        """Streams the layers (dicts of W, x, bias, m0, s, relu) back to back, checks every
        output beat bit-exact against the model, and returns the hardware's outputs, one
        B x M array per layer."""
        p = self.p
        beats, expected = [], []
        for L in layers:
            y, stats = am.layer_tiled(p, L["W"], L["x"], L["bias"], L["m0"], L["s"], L["relu"])
            assert stats["max_acc"] < (1 << (p.acc_w - 1))
            beats += am.encode_layer(p, L["W"], L["x"], L["bias"], L["m0"], L["s"], L["relu"])
            expected.append(y)
            K, M = L["W"].shape
            B = L["x"].shape[0]
            self.cov[f"B={B}"] += 1
            self.cov[f"k_tiles={am.pad_to(K, p.N) // p.N}"] += 1
            self.cov[f"m_tiles={am.pad_to(M, p.N) // p.N}"] += 1
            self.cov[f"relu={int(L['relu'])}"] += 1
        n_out = sum(y.shape[0] * (am.pad_to(y.shape[1], p.N) // p.N) for y in expected)
        base = len(self.received)
        await self.send(beats)
        await self.wait_outputs(base + n_out, timeout_cycles=50 * len(beats) + 2000)
        if idle_after:
            await ClockCycles(self.dut.clk, idle_after)
        got = self.received[base:]
        assert len(got) == n_out, f"expected {n_out} output beats, got {len(got)}"
        i = 0
        hw_out = []
        for li, (L, y) in enumerate(zip(layers, expected)):
            M = L["W"].shape[1]
            B = L["x"].shape[0]
            nb = B * (am.pad_to(M, p.N) // p.N)
            hw = am.decode_outputs(p, got[i:i + nb], M, B)
            i += nb
            if not (hw == y).all():
                bad = list(zip(*((hw != y).nonzero())))[:8]
                raise AssertionError(
                    f"layer {li}: mismatch at (image, output) {bad}; "
                    f"hw={[int(hw[b, m]) for b, m in bad]} model={[int(y[b, m]) for b, m in bad]}")
            hw_out.append(hw)
            fcov.sample_layer(self.fcov, p, L, y)
        self.merge_coverage()
        return hw_out   # what the hardware produced (already checked equal to the model)

    async def read_status(self):
        """Send a STATUS request; returns (total_cycles, act_beats, accept_cycle)."""
        base = len(self.received)
        await self.send([am.status_beat()])
        accepted = self.last_accept_cycle
        await self.wait_outputs(base + 2, timeout_cycles=200)
        total, acts = self.received[base], self.received[base + 1]
        del self.received[base:base + 2]
        self.cov["status_read"] += 1
        return total, acts, accepted

    @property
    def fcov(self):
        if not hasattr(self, "_fcov"):
            self._fcov = fcov.load(fcov.build_id())
        return self._fcov

    def merge_coverage(self):
        """Fold this test's stall / FSM counters into the functional coverage file."""
        names = {"in_gap": "in_gap", "out_stall": "out_stall", "busy_0->1": "busy_0->1",
                 "busy_1->0": "busy_1->0", "reset_mid_stream": "reset_mid_stream",
                 "status_read": "status_read", "bank_switch": "bank_switch",
                 f"in_stall_type_{am.T_WEIGHT}": "stall_WEIGHT", f"in_stall_type_{am.T_BIAS}": "stall_BIAS",
                 f"in_stall_type_{am.T_CONFIG}": "stall_CONFIG"}
        for k, bin_name in names.items():
            fcov.sample_counter(self.fcov, bin_name, self.cov[k] - self.cov_merged[k])
            self.cov_merged[k] = self.cov[k]
        fcov.save(fcov.build_id(), self.fcov)

    def log_coverage(self):
        self.dut._log.info("coverage: " + ", ".join(f"{k}:{v}" for k, v in sorted(self.cov.items())))



def rand_weights(rng, p, K, M, kind="uniform"):
    if p.ternary:
        if kind == "sparse":
            probs = (0.15, 0.7, 0.15)
        else:
            probs = (1 / 3, 1 / 3, 1 / 3)
        return rng.choice([-1, 0, 1], size=(K, M), p=probs)
    if kind == "sparse":
        W = rng.integers(-128, 128, size=(K, M))
        return W * (rng.random((K, M)) < 0.3)
    if kind == "extreme":
        return rng.choice([-128, -1, 0, 1, 127], size=(K, M))
    return rng.integers(-128, 128, size=(K, M))


def rand_acts(rng, B, K, kind="uniform"):
    if kind == "relu":       # layer-2 style inputs: post-ReLU in [0,127]
        return rng.integers(0, 128, size=(B, K))
    if kind == "sparse":
        return rng.integers(-128, 128, size=(B, K)) * (rng.random((B, K)) < 0.3)
    if kind == "extreme":
        return rng.choice([-128, -1, 0, 1, 127], size=(B, K))
    return rng.integers(-128, 128, size=(B, K))


def auto_scale(p, W, x, bias, relu):
    """Picks M0 and s so the largest |acc| lands just beyond the int8 range, so a few
    outputs saturate and the clamp gets exercised."""
    acc, _ = am.layer_acc(p, W, x, bias)
    peak = max(int(abs(acc).max()), 1)
    return am.choose_scale(160.0 / peak, p.s_max)
