"""Cycles per image and array utilisation from the on-chip
counters, running the real two-layer MLP (seed-0 networks) on Fashion-MNIST test
images. No input gaps or output stalls, so the counts are the hardware's own.

utilisation = useful MAC cycles / (N^2 x total cycles) = ACT beats / total cycles
(every accepted ACT beat makes all N^2 PEs do one MAC).

Env: PERF_DATASET (default fashion), PERF_BATCHES (comma list of batch sizes,
default "8,1"), PERF_OUT (JSON lines), ACCEL_* as for the other tests.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cocotb

from accel_tb import DOUBLE_BUF, PERF, AccelTB

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
import data  # noqa: E402
import mlp  # noqa: E402


@cocotb.test(skip=not PERF)
async def test_cycles_and_utilisation(dut):
    tb = AccelTB(dut)
    await tb.start()
    p = tb.p
    ds = os.environ.get("PERF_DATASET", "fashion")
    prec = "ternary" if p.ternary else "int8"
    net = mlp.load_net(ROOT / "results" / "accuracy" / "nets" / f"{ds}_{prec}_s0.npz")
    _, _, xte, _ = data.load(ds)
    L1, L2 = mlp.layers(net)
    for B in [int(b) for b in os.environ.get("PERF_BATCHES", "8,1").split(",")]:
        x = mlp.to_input(xte[:B])
        await tb.read_status()                                   # the first STATUS clears the counters
        (h,) = await tb.run_layers([dict(L1, x=x)], idle_after=0)
        c1, a1, _ = await tb.read_status()
        (y,) = await tb.run_layers([dict(L2, x=h)], idle_after=0)
        c2, a2, _ = await tb.read_status()
        assert (y == mlp.int_forward(p, net, xte[:B])).all()
        rec = dict(dataset=ds, precision=prec, double_buf=DOUBLE_BUF, batch=B, N=p.N,
                   layer1_cycles=c1, layer1_act_beats=a1, layer2_cycles=c2, layer2_act_beats=a2,
                   total_cycles=c1 + c2, cycles_per_image=(c1 + c2) / B,
                   utilisation=(a1 + a2) / (c1 + c2),
                   layer1_utilisation=a1 / c1, layer2_utilisation=a2 / c2, bit_exact=True)
        dut._log.info("perf: " + json.dumps(rec))
        if os.environ.get("PERF_OUT"):
            out = Path(os.environ["PERF_OUT"])
            out = out if out.is_absolute() else ROOT / out   # the simulator runs in its build dir
            with open(out, "a") as f:
                f.write(json.dumps(rec) + "\n")
