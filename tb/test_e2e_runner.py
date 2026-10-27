"""pytest entry point for the end-to-end MLP test on both builds and both datasets.

    pytest tb/test_e2e_runner.py                      # 8 images each (default)
    E2E_IMAGES=100 E2E_OUT=results/sim/e2e.jsonl pytest tb/test_e2e_runner.py
    E2E_VARIANT=perf|db ...                           # feature variants
"""

import os
from pathlib import Path

import pytest
from cocotb_tools.runner import get_runner

from test_runner import VARIANTS

ROOT = Path(__file__).resolve().parents[1]
RTL = sorted((ROOT / "rtl").glob("*.v"))


@pytest.mark.parametrize("dataset", ["mnist", "fashion"])
@pytest.mark.parametrize("ternary", [0, 1], ids=["int8", "ternary"])
def test_e2e(ternary, dataset):
    prec = "ternary" if ternary else "int8"
    if not (ROOT / "results" / "accuracy" / "nets" / f"{dataset}_{prec}_s0.npz").exists():
        pytest.skip("trained network not exported yet (run train/run_all.sh)")
    variant = os.environ.get("E2E_VARIANT", "base")
    db, perf = VARIANTS[variant]
    params = {"N": 4, "TERNARY": ternary, "B_MAX": 8, "K_MAX": 784,
              "DOUBLE_BUF": db, "PERF_COUNTERS": perf}
    build_dir = ROOT / "sim_build" / f"e2e_{prec}_{dataset}_{variant}"  # separate, so runs can go in parallel
    runner = get_runner("icarus")
    runner.build(sources=RTL, hdl_toplevel="accel_top", parameters=params,
                 build_dir=build_dir, timescale=("1ns", "1ps"), always=True)
    env = {"ACCEL_N": "4", "ACCEL_TERNARY": str(ternary), "ACCEL_B_MAX": "8",
           "ACCEL_K_MAX": "784", "E2E_DATASET": dataset,
           "E2E_IMAGES": os.environ.get("E2E_IMAGES", "8"), "E2E_VARIANT": variant,
           "ACCEL_PERF": str(perf), "ACCEL_DB": str(db)}
    if os.environ.get("E2E_OUT"):
        env["E2E_OUT"] = str(Path(os.environ["E2E_OUT"]).resolve())
    runner.test(hdl_toplevel="accel_top", test_module="test_e2e", test_dir=ROOT / "tb",
                build_dir=build_dir, extra_env=env,
                results_xml=str(build_dir / f"e2e_{dataset}.result.xml"))
