"""Mutation testing: inject one RTL bug at a time and check the cocotb suite catches it.

    python tb/mutation.py [--seed 12345] [--out results/verification/mutants.md]

Every mutant runs against every build (int8/ternary x base/perf/db) with the same seed.
A mutant applies to a build if the mutated code is part of that build. The RTL is
restored after each mutant and checked against the starting snapshot at the end.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path

from test_runner import VARIANTS

ROOT = Path(__file__).resolve().parents[1]
BUILDS = [f"{t}-{v}" for t in ("int8", "ternary") for v in VARIANTS]


def sel(prec=None, db=None, perf=None):
    """Builds whose (precision, DOUBLE_BUF, PERF_COUNTERS) match."""
    out = set()
    for b in BUILDS:
        t, v = b.split("-")
        d, pc = VARIANTS[v]
        if (prec in (None, t)) and (db in (None, d)) and (perf in (None, pc)):
            out.add(b)
    return out


ALL = set(BUILDS)
INT8, TERN = sel("int8"), sel("ternary")


# id, file, old, new, description, builds it applies to
MUTANTS = [
    ("M1t", "accel_top.v", "TERNARY ? 2 * N - 2 : N - 1;", "TERNARY ? 2 * N - 3 : N - 1;",
     "ternary WEIGHT drain 2N-3 instead of 2N-2", TERN),
    ("M1i", "accel_top.v", "TERNARY ? 2 * N - 2 : N - 1;", "TERNARY ? 2 * N - 2 : N - 2;",
     "int8 WEIGHT drain N-2 instead of N-1", INT8),
    ("M2", "accel_top.v", "localparam LAT      = 2 * N;", "localparam LAT      = 2 * N + 1;",
     "tag pipeline latency 2N+1", ALL),
    ("M3", "requant.v", "sum <= p_lo + (p_hi <<< 8) + rnd;", "sum <= p_lo + (p_hi <<< 8);",
     "no rounding constant", ALL),
    ("M4", "pe_ternary.v", "psum_out <= psum_in + operand + {{(PSUM_W-1){1'b0}}, w_neg};",
     "psum_out <= psum_in + operand;", "ternary -1 without carry-in", TERN),
    ("M5", "pe_int8.v", "wire signed [15:0] prod = w * a_in;",
     "wire signed [15:0] prod = $signed({1'b0,w[6:0]}) * a_in;", "int8 weight sign bit dropped", INT8),
    ("M6", "requant.v", "wire signed [PW-1:0] lo = relu ? 0 : -128;", "wire signed [PW-1:0] lo = relu ? 0 : -127;",
     "clamp -127 instead of -128", ALL),
    ("M7", "accel_top.v", "else if (out_xfer)                   out_valid <= 1'b0;",
     "else                                 out_valid <= 1'b0;", "out_valid drops without out_ready", ALL),
    ("M8", "acc_bank.v", "row[c*ACC_W +: ACC_W] + {{(ACC_W-PSUM_W){col[PSUM_W-1]}}, col};",
     "row[c*ACC_W +: ACC_W] + {{(ACC_W-PSUM_W){1'b0}}, col};", "accumulator zero-extends the column sum", ALL),
    ("M9", "acc_bank.v", "wire [BW-1:0]            row_b = acc_en ? acc_b : rd_b;",
     "wire [BW-1:0]            row_b = rd_b;", "shared row mux ignores acc_b", ALL),
    ("M10", "requant.v", "sum <= p_lo + (p_hi <<< 8) + rnd;", "sum <= p_lo + (p_hi <<< 7) + rnd;",
     "high partial product shifted by 7", ALL),
    # performance counters and STATUS read-out
    ("P1", "accel_top.v", "if (is_act) cnt_acts <= cnt_acts + 1'b1;", "if (acc_xfer) cnt_acts <= cnt_acts + 1'b1;",
     "ACT counter counts every accepted beat", sel(perf=1)),
    ("P2", "accel_top.v", "cnt_cycles <= 32'd1;        // this cycle starts the next interval",
     "cnt_cycles <= 32'd0;", "cycle counter restarts at 0 (off by one)", sel(perf=1)),
    ("P3", "accel_top.v", "wire [31:0] st_beat = (st_left == 2'd2) ? st_first : st_word;",
     "wire [31:0] st_beat = (st_left == 2'd2) ? st_word : st_first;", "STATUS beats in the wrong order",
     sel(perf=1)),
    ("P4", "accel_top.v", "assign rq_xfer = out_xfer & (st_left == 2'd0);", "assign rq_xfer = out_xfer;",
     "requant sequencer also reacts to STATUS beats", sel(perf=1)),
    # double-buffering
    ("D1", "accel_top.v", "((w_beat == {WBW{1'b0}}) ? ~act_bank : load_bank)", "1'b0",
     "every tile loads bank 0 (no double-buffering, no stall)", sel(db=1)),
    ("D2", "accel_top.v", "if (is_act && act_bank) drain_cnt1 <= DRAIN;", "if (1'b0) drain_cnt1 <= DRAIN;",
     "bank-1 drain counter never set", sel(db=1)),
    ("D3", "pe_int8.v", "assign w     = b_in ? w1 : w0;", "assign w     = w1;",
     "int8 PE ignores the bank tag", sel("int8", db=1)),
    ("D4", "pe_ternary.v", "assign w     = b_in ? w1 : w0;", "assign w     = w0;",
     "ternary PE ignores the bank tag", sel("ternary", db=1)),
]


def run_build(build: str, seed: int):
    t, v = build.split("-")
    env = dict(os.environ, COCOTB_RANDOM_SEED=str(seed))
    out = subprocess.run(["pytest", "tb/test_runner.py", "-s", "-q", "-k", f"{t} and {v}"], cwd=ROOT,
                         env=env, capture_output=True, text=True, timeout=3600).stdout
    m = re.search(r"TESTS=(\d+) PASS=(\d+) FAIL=(\d+) SKIP=(\d+)", out)
    fails = re.findall(r"test_accel\.(test_\w+) failed", out)
    if not m:
        return None, ["(build or simulator error)"]
    return (int(m.group(2)), int(m.group(1)) - int(m.group(4))), fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--only", nargs="*", help="mutant ids to run (default: all)")
    args = ap.parse_args()
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip() or "uncommitted"
    rtl = ROOT / "rtl"
    lines = [f"Mutation results at commit `{commit}`, seed {args.seed} (`python tb/mutation.py`).", "",
             "Baseline (no mutant): " + ", ".join(
                 "{} {}/{}".format(b, *run_build(b, args.seed)[0]) for b in BUILDS), "",
             "| # | Injected bug (file) | " + " | ".join(BUILDS) + " |",
             "|---|---|" + "---|" * len(BUILDS)]
    caught = {b: [0, 0] for b in BUILDS}
    snapshot = {f.name: f.read_text() for f in rtl.glob("*.v")}
    for mid, f, old, new, desc, applies in MUTANTS:
        if args.only and mid not in args.only:
            continue
        p = rtl / f
        orig = p.read_text()
        assert orig.count(old) == 1, f"{mid}: pattern not found exactly once in {f}"
        p.write_text(orig.replace(old, new))
        cells = []
        try:
            for b in BUILDS:
                (passed, total), fails = run_build(b, args.seed)
                if b in applies:
                    caught[b][1] += 1
                    caught[b][0] += passed < total
                    cells.append(f"**{total - passed}/{total} fail**: " + ", ".join(
                        x.replace("test_", "") for x in fails) if passed < total else "**NOT CAUGHT**")
                else:
                    cells.append(f"n/a ({passed}/{total} pass)")
        finally:
            p.write_text(orig)
        lines.append(f"| {mid} | {desc} (`{f}`) | " + " | ".join(cells) + " |")
        print(lines[-1], flush=True)
    changed = [n for n, t in snapshot.items() if (rtl / n).read_text() != t]
    assert not changed, f"RTL not restored: {changed}"
    lines += ["", "Caught in every build it applies to: " + ", ".join(
        f"{b} {c}/{n}" for b, (c, n) in caught.items()) + "."]
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)


if __name__ == "__main__":
    main()
