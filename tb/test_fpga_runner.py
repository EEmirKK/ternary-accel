"""pytest entry point for the FPGA wrapper tests: the UART echo at 115200 baud, and the full MLP over the UART."""

import os
from pathlib import Path

import pytest
from cocotb_tools.runner import get_runner

ROOT = Path(__file__).resolve().parents[1]
RTL = sorted((ROOT / "rtl").glob("*.v"))
FPGA = [ROOT / "fpga" / f for f in ("uart_rx.v", "uart_tx.v", "accel_fpga.v")]


def _run(top, params, env, tag):
    build_dir = ROOT / "sim_build" / f"fpga_{tag}"
    runner = get_runner("icarus")
    srcs = RTL + FPGA + ([ROOT / "tb" / "uart_loop.v"] if top == "uart_loop" else [])
    runner.build(sources=srcs, hdl_toplevel=top, parameters=params, build_dir=build_dir,
                 timescale=("1ns", "1ps"), always=True)
    runner.test(hdl_toplevel=top, test_module="test_fpga", test_dir=ROOT / "tb", build_dir=build_dir,
                extra_env=dict(env, FPGA_TOP=top), results_xml=str(build_dir / "fpga.result.xml"))


def test_uart_echo_115200():
    _run("uart_loop", {"CLKS_PER_BIT": 434}, {"FPGA_CPB": "434"}, "uart")


@pytest.mark.parametrize("ternary", [0, 1], ids=["int8", "ternary"])
def test_fpga_mlp(ternary):
    env = {"FPGA_CPB": "8", "FPGA_TERNARY": str(ternary)}
    for k in ("FPGA_DATASET", "FPGA_BATCHES", "FPGA_OUT"):
        if os.environ.get(k):
            env[k] = str(Path(os.environ[k]).resolve()) if k == "FPGA_OUT" else os.environ[k]
    _run("accel_fpga", {"TERNARY": ternary, "CLKS_PER_BIT": 8, "PROG_DEPTH": 3328 if ternary else 13312},
         env, "int8" if not ternary else "ternary")
