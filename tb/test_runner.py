"""pytest entry point: builds accel_top with Icarus for every variant and runs the cocotb tests.

    pytest tb/test_runner.py                          # all six builds
    pytest tb/test_runner.py -k ternary               # the ternary builds
    TESTCASE=test_padding pytest tb/test_runner.py    # one cocotb test
    ACCEL_N=8 pytest tb/test_runner.py -k base        # another array size
"""

import os
from pathlib import Path

import pytest
from cocotb_tools.runner import get_runner

ROOT = Path(__file__).resolve().parents[1]
RTL = sorted((ROOT / "rtl").glob("*.v"))


# variant -> (DOUBLE_BUF, PERF_COUNTERS); the other runners and tb/mutation.py import this
VARIANTS = {"base": (0, 0), "perf": (0, 1), "db": (1, 1)}


@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("ternary", [0, 1], ids=["int8", "ternary"])
def test_accel(ternary, variant):
    double_buf, perf = VARIANTS[variant]
    params = {"N": int(os.environ.get("ACCEL_N", "4")), "TERNARY": ternary, "B_MAX": 8, "K_MAX": 784,
              "DOUBLE_BUF": double_buf, "PERF_COUNTERS": perf}
    build_dir = ROOT / "sim_build" / f"accel_n{params['N']}_{'ternary' if ternary else 'int8'}_{variant}"
    runner = get_runner("icarus")
    runner.build(
        sources=RTL,
        hdl_toplevel="accel_top",
        parameters=params,
        build_dir=build_dir,
        timescale=("1ns", "1ps"),
        always=True,
    )
    # Coverage accumulates over this build's tests, so start from an empty file.
    cov_id = "n{}_{}_db{}_perf{}".format(params["N"], "ternary" if ternary else "int8", double_buf, perf)
    (ROOT / "sim_build" / f"coverage_{cov_id}.json").unlink(missing_ok=True)
    runner.test(
        hdl_toplevel="accel_top",
        test_module="test_accel",
        test_dir=ROOT / "tb",
        build_dir=build_dir,
        testcase=os.environ.get("TESTCASE") or None,
        extra_env={
            "ACCEL_N": str(params["N"]),
            "ACCEL_TERNARY": str(ternary),
            "ACCEL_B_MAX": str(params["B_MAX"]),
            "ACCEL_K_MAX": str(params["K_MAX"]),
            "ACCEL_PERF": str(perf),
            "ACCEL_DB": str(double_buf),
        },
    )
