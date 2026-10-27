"""Checks a finished LibreLane run against the closure criterion and appends the result to
results/asic/search.jsonl.

    python flow/check_closure.py <build> <period_ns>

A run closes if setup slack is non-negative at the three typical corners and hold slack at
all nine, and DRC (KLayout and Magic), LVS and antenna checks are clean, which is LibreLane's
default sky130 signoff. Slow-corner setup, the period it implies, and the max slew,
capacitance and fanout counts are recorded but don't gate.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
build, period = sys.argv[1], float(sys.argv[2])
tag = f"{build}_{sys.argv[2]}ns"
run = ROOT / "flow" / "runs" / tag
m = json.loads((run / "final" / "metrics.json").read_text())


def corners(prefix, sel):
    return {k.split("corner:")[1]: v for k, v in m.items()
            if k.startswith(prefix + "__corner:") and sel(k.split("corner:")[1])}


tt_setup = corners("timing__setup__ws", lambda c: "_tt_" in c)
ss_setup = corners("timing__setup__ws", lambda c: "_ss_" in c)
hold = corners("timing__hold__ws", lambda c: True)
assert len(tt_setup) == 3 and len(ss_setup) == 3 and len(hold) == 9, (tt_setup, ss_setup, hold)

checks = {
    "tt_setup_wns_ge_0": min(tt_setup.values()) >= 0,
    "hold_wns_ge_0_all_corners": min(hold.values()) >= 0,
    "klayout_drc_clean": m.get("klayout__drc_error__count") == 0,
    "magic_drc_clean": m.get("magic__drc_error__count") == 0,
    "lvs_clean": m.get("design__lvs_error__count") == 0,
    "antenna_clean": m.get("antenna__violating__nets") == 0,
    "unmapped_is_only_u_array": m.get("design__instance_unmapped__count") == 1,
}
ss_wns = min(ss_setup.values())
rec = {
    "tag": tag, "build": build, "period_ns": period,
    # the commit the run was hardened at, from the first line harden.sh writes to its log
    "run_commit": (ROOT / "flow" / "runs" / f"{tag}.log").read_text().split()[1].rstrip(";"),
    "closes": all(checks.values()), "checks": checks,
    "tt_setup_wns_ns": min(tt_setup.values()),
    "tt_setup_tns_ns": min(corners("timing__setup__tns", lambda c: "_tt_" in c).values()),
    "all_corner_setup_tns_ns": m.get("timing__setup__tns"),   # includes ss (not gating)
    "nom_tt_setup_wns_ns": tt_setup["nom_tt_025C_1v80"],
    "hold_wns_ns": min(hold.values()), "hold_worst_corner": min(hold, key=hold.get),
    "ss_setup_wns_ns": ss_wns,
    "ss_period_needed_ns": round(period - ss_wns, 3),
    "max_slew_violations": m.get("design__max_slew_violation__count"),
    "max_cap_violations": m.get("design__max_cap_violation__count"),
    "max_fanout_violations": m.get("design__max_fanout_violation__count"),
    "tt_setup_by_corner": tt_setup, "ss_setup_by_corner": ss_setup, "hold_by_corner": hold,
}
out = ROOT / "results" / "asic"
out.mkdir(parents=True, exist_ok=True)
with open(out / "search.jsonl", "a") as f:
    f.write(json.dumps(rec) + "\n")
print(json.dumps({k: rec[k] for k in ("tag", "closes", "tt_setup_wns_ns", "hold_wns_ns",
                                      "hold_worst_corner", "ss_setup_wns_ns", "ss_period_needed_ns",
                                      "max_slew_violations")}))
if not rec["closes"]:
    print("failed:", [k for k, v in checks.items() if not v])
