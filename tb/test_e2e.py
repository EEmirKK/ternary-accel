"""End-to-end: real test images through the RTL, layer 1 then layer 2.

Layer 2's input is the hardware's own layer-1 output. Every hidden value and
every logit is checked bit-exact against the integer model, and the final
logits are also compared with model/mlp.int_forward on the same images.

Env: E2E_DATASET (mnist|fashion), E2E_IMAGES (default 8), E2E_OUT (optional JSON path).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

import cocotb

from accel_tb import AccelTB

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
import data  # noqa: E402
import mlp  # noqa: E402  (model/ is already on sys.path via accel_tb)


@cocotb.test()
async def test_mlp_end_to_end(dut):
    tb = AccelTB(dut)
    await tb.start()
    p = tb.p
    ds = os.environ["E2E_DATASET"]
    n = int(os.environ.get("E2E_IMAGES", "8"))
    prec = "ternary" if p.ternary else "int8"
    net = mlp.load_net(ROOT / "results" / "accuracy" / "nets" / f"{ds}_{prec}_s0.npz")
    _, _, xte, yte = data.load(ds)
    L1, L2 = mlp.layers(net)

    logits = []
    for i in range(0, n, p.B_MAX):
        x = mlp.to_input(xte[i:min(i + p.B_MAX, n)])
        (h,) = await tb.run_layers([dict(L1, x=x)], idle_after=2)
        (y,) = await tb.run_layers([dict(L2, x=h)], idle_after=2)
        logits.append(y)
    logits = np.vstack(logits)

    ref = mlp.int_forward(p, net, xte[:n])
    assert (logits == ref).all(), "hardware logits differ from int_forward"
    acc, ties = mlp.accuracy_int(logits, yte[:n])
    pred = logits.argmax(axis=1)
    rec = dict(dataset=ds, precision=prec, images=n, bit_exact=True,
               hw_correct=int(round(acc * n)), ties=ties,
               first_predictions=pred[:16].tolist(), first_labels=yte[:16].tolist(),
               sim_time_ns=int(cocotb.utils.get_sim_time("ns")),
               # first input beat accepted -> last output beat accepted (no gaps/stalls
               # unless set); includes the host's layer-1 -> layer-2 turnaround
               stream_cycles=tb.last_out_cycle - tb.first_in_cycle,
               clk_ps=int(os.environ.get("ACCEL_CLK_PS", "10000")),
               gate_level=os.environ.get("ACCEL_GL", "0") == "1",
               variant=os.environ.get("E2E_VARIANT", "base"))
    dut._log.info("e2e: " + json.dumps(rec))
    if os.environ.get("E2E_OUT"):
        out = Path(os.environ["E2E_OUT"])
        out = out if out.is_absolute() else ROOT / out   # the simulator runs in its build dir
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
