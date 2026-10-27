"""Gate-level simulation of a LibreLane post-route netlist with zero-delay functional sky130
cell models. It runs the end-to-end MLP test and dumps a VCD for the power analysis.

    GL_RUN=flow/runs/int8_17.5ns GL_TERNARY=0 GL_CLK_PS=17500 \
    GL_VCD=/abs/path/out.vcd E2E_IMAGES=8 pytest tb/test_gl_runner.py -s

GL_SIM is icarus (the default) or verilator, which must be on PATH and at least 5.036 for cocotb 2.x.
"""

import glob
import os
from pathlib import Path

import pytest
from cocotb_tools.runner import Icarus, get_runner

ROOT = Path(__file__).resolve().parents[1]


class IcarusVCD(Icarus):
    """cocotb's Icarus runner passes vvp '-none' (no dumping) unless waves are on,
    and '-fst' when they are; OpenSTA's read_vcd needs VCD. Overrides a private
    cocotb method (checked against cocotb 2.1.0)."""

    def _test_command(self):
        cmds = super()._test_command()
        return [["-vcd" if a == "-none" else a for a in cmd] for cmd in cmds]


def cell_models():
    pdk = os.path.expanduser(os.environ.get("PDK_ROOT", "~/.ciel"))
    base = glob.glob(f"{pdk}/ciel/sky130/versions/*/sky130A/libs.ref/sky130_fd_sc_hd/verilog")
    assert len(base) == 1, f"expected one sky130 PDK version, found {base}"
    return [Path(base[0]) / "primitives.v", Path(base[0]) / "sky130_fd_sc_hd.v"]


@pytest.mark.skipif("GL_RUN" not in os.environ, reason="set GL_RUN to a LibreLane run dir")
def test_gate_level():
    run = (ROOT / os.environ["GL_RUN"]).resolve()
    netlist = run / "final" / "nl" / "accel_top.nl.v"
    assert netlist.exists(), netlist
    ternary = os.environ["GL_TERNARY"]
    clk_ps = os.environ["GL_CLK_PS"]
    sim = os.environ.get("GL_SIM", "icarus")
    build_dir = ROOT / "sim_build" / f"gl_{run.name}_{sim}"
    sources = [netlist, *cell_models()]
    defines = {"FUNCTIONAL": 1, "UNIT_DELAY": "#0"}
    build_args, test_args, plusargs = [], [], []
    if sim == "icarus":
        sources.append(ROOT / "tb" / "gl_dump.v")
        build_args = ["-s", "gl_dump"]
        if os.environ.get("GL_VCD"):
            plusargs = [f"+vcd={os.environ['GL_VCD']}"]
    else:
        build_args = ["--trace", "-Wno-fatal", "-Wno-lint", "-Wno-style"]
    runner = IcarusVCD() if sim == "icarus" else get_runner(sim)
    runner.build(sources=sources, hdl_toplevel="accel_top", defines=defines,
                 build_args=build_args, build_dir=build_dir,
                 timescale=("1ns", "1ps"), always=True,
                 waves=(sim == "verilator" and bool(os.environ.get("GL_VCD"))))
    env = {"ACCEL_N": os.environ.get("GL_N", "4"), "ACCEL_TERNARY": ternary, "ACCEL_B_MAX": "8", "ACCEL_K_MAX": "784",
           "ACCEL_GL": "1", "ACCEL_CLK_PS": clk_ps,
           "E2E_DATASET": os.environ.get("E2E_DATASET", "fashion"),
           "E2E_IMAGES": os.environ.get("E2E_IMAGES", "8")}
    if os.environ.get("E2E_OUT"):
        env["E2E_OUT"] = str(Path(os.environ["E2E_OUT"]).resolve())
    runner.test(hdl_toplevel="accel_top", test_module="test_e2e", test_dir=ROOT / "tb",
                build_dir=build_dir, extra_env=env, plusargs=plusargs, test_args=test_args,
                waves=(sim == "verilator" and bool(os.environ.get("GL_VCD"))),
                results_xml=str(build_dir / "gl.result.xml"))
