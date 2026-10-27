"""pytest entry point for the cycle and utilisation measurement.

    PERF_OUT=results/sim/perf.jsonl pytest tb/test_perf_runner.py -s
Runs int8 and ternary, single-buffered (variant perf) and double-buffered (db).
"""

import os
from pathlib import Path

import pytest
from cocotb_tools.runner import get_runner

from test_runner import VARIANTS

ROOT = Path(__file__).resolve().parents[1]
RTL = sorted((ROOT / "rtl").glob("*.v"))


@pytest.mark.parametrize("variant", ["perf", "db"])
@pytest.mark.parametrize("ternary", [0, 1], ids=["int8", "ternary"])
def test_perf(ternary, variant):
    db, perf = VARIANTS[variant]
    n = int(os.environ.get("ACCEL_N", "4"))
    params = {"N": n, "TERNARY": ternary, "B_MAX": 8, "K_MAX": 784,
              "DOUBLE_BUF": db, "PERF_COUNTERS": perf}
    build_dir = ROOT / "sim_build" / f"perf_n{n}_{'ternary' if ternary else 'int8'}_{variant}"
    runner = get_runner("icarus")
    runner.build(sources=RTL, hdl_toplevel="accel_top", parameters=params, build_dir=build_dir,
                 timescale=("1ns", "1ps"), always=True)
    env = {"ACCEL_N": str(n), "ACCEL_TERNARY": str(ternary), "ACCEL_B_MAX": "8", "ACCEL_K_MAX": "784",
           "ACCEL_PERF": str(perf), "ACCEL_DB": str(db)}
    for k in ("PERF_DATASET", "PERF_BATCHES"):
        if os.environ.get(k):
            env[k] = os.environ[k]
    if os.environ.get("PERF_OUT"):
        env["PERF_OUT"] = str(Path(os.environ["PERF_OUT"]).resolve())
    runner.test(hdl_toplevel="accel_top", test_module="test_perf", test_dir=ROOT / "tb",
                build_dir=build_dir, extra_env=env, results_xml=str(build_dir / "perf.result.xml"))
