"""cocotb tests for the FPGA wrapper (fpga/accel_fpga.v) and its UART.

test_uart_echo runs uart_rx and uart_tx at the real 115200-baud divisor, 434 clocks per bit.
test_fpga_mlp runs the whole wrapper with a fast divisor. It loads the real network over the
UART with fpga/protocol.py, runs batches of Fashion-MNIST test images, and checks every logit
against the model and the ACT-beat counter against the exact expected count.
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, First, ReadOnly

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "model"), str(ROOT / "train"), str(ROOT / "fpga")]
import accel_model as am  # noqa: E402
import data  # noqa: E402
import mlp  # noqa: E402
import protocol  # noqa: E402

CPB = int(os.environ.get("FPGA_CPB", "8"))
CLK_NS = 20                                  # 50 MHz, as on the DE0-Nano


async def uart_send(dut, sig, payload: bytes):
    for byte in payload:
        for bit in [0] + [(byte >> i) & 1 for i in range(8)] + [1]:
            sig.value = bit
            await ClockCycles(dut.clk, CPB)


async def uart_recv(dut, sig, n: int, timeout_cycles: int) -> bytes:
    out = bytearray()
    while len(out) < n:
        edge = FallingEdge(sig)
        if await First(edge, ClockCycles(dut.clk, timeout_cycles)) is not edge:
            raise AssertionError(f"UART timeout after {len(out)} of {n} bytes")
        await ClockCycles(dut.clk, CPB // 2)
        await ReadOnly()
        assert int(sig.value) == 0, "false start bit"
        b = 0
        for i in range(8):
            await ClockCycles(dut.clk, CPB)
            await ReadOnly()
            b |= int(sig.value) << i
        await ClockCycles(dut.clk, CPB)
        await ReadOnly()
        assert int(sig.value) == 1, "framing error (stop bit)"
        out.append(b)
        await ClockCycles(dut.clk, 1)        # leave the ReadOnly phase before anyone drives
    return bytes(out)


async def reset(dut, rx_name):
    getattr(dut, rx_name).value = 1
    dut.rst_n.value = 0
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 5)


@cocotb.test(skip=os.environ.get("FPGA_TOP") != "uart_loop")
async def test_uart_echo(dut):
    await reset(dut, "rx")
    rng = random.Random(1)
    payload = bytes([0x00, 0xFF, 0x55, 0xAA] + [rng.randrange(256) for _ in range(12)])
    rx_task = cocotb.start_soon(uart_recv(dut, dut.tx, len(payload), 10 ** 7))
    await uart_send(dut, dut.rx, payload)
    got = await rx_task
    assert got == payload, f"echo {got.hex()} != {payload.hex()}"
    dut._log.info(f"echoed {len(payload)} bytes at {CPB} clocks/bit")


@cocotb.test(skip=os.environ.get("FPGA_TOP") != "accel_fpga")
async def test_fpga_mlp(dut):
    ternary = os.environ.get("FPGA_TERNARY", "0") == "1"
    ds = os.environ.get("FPGA_DATASET", "fashion")
    batches = [int(b) for b in os.environ.get("FPGA_BATCHES", "8,3").split(",")]
    await reset(dut, "uart_rx")
    p = am.AccelParams(ternary=ternary)
    prec = "ternary" if ternary else "int8"
    net = mlp.load_net(ROOT / "results" / "accuracy" / "nets" / f"{ds}_{prec}_s0.npz")
    words, table = protocol.build_program(p, net)
    await uart_send(dut, dut.uart_rx, protocol.cmd_program(words) + protocol.cmd_table(table))
    dut._log.info(f"program loaded: {len(words)} words")
    _, _, xte, _ = data.load(ds)
    mt1 = table[1][2]
    start = 0
    results = []
    for B in batches:
        x = xte[start:start + B]
        start += B
        rx_task = cocotb.start_soon(uart_recv(dut, dut.uart_tx, protocol.reply_len(B, mt1), 10 ** 8))
        await uart_send(dut, dut.uart_rx, protocol.cmd_run(mlp.to_input(x)))
        buf = await rx_task
        logits, cycles, acts = protocol.parse_reply(buf, B, mt1)
        ref = mlp.int_forward(p, net, x)
        dut._log.info(f"reply for batch of {B}: {cycles} cycles, {acts} ACT beats")
        if not (logits == ref).all():
            bad = sorted({int(i) for i in np.nonzero((logits != ref).any(axis=1))[0]})
            raise AssertionError(f"batch of {B} (images {start - B}..{start - 1}): logits differ for "
                                 f"images {bad}; hw {logits[bad[0]].tolist()} model {ref[bad[0]].tolist()}")
        exp_acts = sum(kt * mt * B for _, kt, mt, _ in table)
        assert acts == exp_acts, f"ACT counter {acts}, expected {exp_acts}"
        rec = dict(precision=prec, dataset=ds, batch=B, cycles=cycles, act_beats=acts,
                   cycles_per_image=cycles / B, utilisation=acts / cycles, bit_exact=True)
        results.append(rec)
        dut._log.info("fpga: " + json.dumps(rec))
    if os.environ.get("FPGA_OUT"):
        out = Path(os.environ["FPGA_OUT"])
        out = out if out.is_absolute() else ROOT / out
        with open(out, "a") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
