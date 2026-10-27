"""Prints the int8 vs ternary ASIC result tables as Markdown, from results/asic/*.json. Each
reduction is (int8 - ternary) / int8 of the measured values.

    python flow/asic_summary.py [build_a build_b [period ...]]    # default: int8 ternary 17.5 11.5
"""

import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results" / "asic"
search = {json.loads(l)["tag"]: json.loads(l) for l in open(R / "search.jsonl")}


def load(tag):
    return (json.loads((R / f"{tag}_area.json").read_text()),
            json.loads((R / f"{tag}_power.json").read_text()), search[tag])


def tns(tag, sel):
    m = json.loads((R / f"{tag}_metrics.json").read_text())
    return min(v for k, v in m.items() if k.startswith("timing__setup__tns__corner:") and sel in k)


def red(a, b):
    return 100.0 * (a - b) / a


def f(x, d=1):
    return f"{x:,.{d}f}"


def table(period, a_build="int8", b_build="ternary"):
    ta, tb = f"{a_build}_{period}ns", f"{b_build}_{period}ns"
    (aa, pa, sa), (ab, pb, sb) = load(ta), load(tb)
    rows = [
        ("Closes (tt setup, all-corner hold, LibreLane default sky130 signoff)", str(sa["closes"]), str(sb["closes"]), ""),
        ("tt setup WNS (ns), worst of nom/min/max tt", f(sa["tt_setup_wns_ns"], 3), f(sb["tt_setup_wns_ns"], 3), ""),
        ("Setup TNS at tt (ns), worst of nom/min/max tt", f(tns(ta, "_tt_"), 3), f(tns(tb, "_tt_"), 3), ""),
        ("Setup TNS at ss (ns), worst of nom/min/max ss (not gating)", f(tns(ta, "_ss_"), 3), f(tns(tb, "_ss_"), 3), ""),
        ("Hold WNS (ns), all 9 corners", f(sa["hold_wns_ns"], 3), f(sb["hold_wns_ns"], 3), ""),
        ("ss setup WNS (ns), worst of nom/min/max ss (not gating)", f(sa["ss_setup_wns_ns"], 3), f(sb["ss_setup_wns_ns"], 3), ""),
        ("Period needed at ss = period − ss WNS (ns), derived", f(sa["ss_period_needed_ns"], 2), f(sb["ss_period_needed_ns"], 2), ""),
        ("Cell area, whole core (µm², excl. fill/decap/tap)", f(aa["logic_area_um2"]), f(ab["logic_area_um2"]),
         f(red(aa["logic_area_um2"], ab["logic_area_um2"])) + " %"),
        ("Cell area, PE array `u_array` (µm²)", f(aa["array_logic_area_um2"]), f(ab["array_logic_area_um2"]),
         f(red(aa["array_logic_area_um2"], ab["array_logic_area_um2"])) + " %"),
        ("Standard cells (logic) in core / in array", f"{aa['logic_count']:,} / {aa['array_logic_count']:,}",
         f"{ab['logic_count']:,} / {ab['array_logic_count']:,}", ""),
        ("Core area / die area (µm²)", f"{f(aa['core_area_um2'], 0)} / {f(aa['die_area_um2'], 0)}",
         f"{f(ab['core_area_um2'], 0)} / {f(ab['die_area_um2'], 0)}", f(red(aa["core_area_um2"], ab["core_area_um2"])) + " % (core)"),
        ("Power: internal (mW)", f(pa["groups"]["total"]["internal_w"] * 1e3, 3), f(pb["groups"]["total"]["internal_w"] * 1e3, 3),
         f(red(pa["groups"]["total"]["internal_w"], pb["groups"]["total"]["internal_w"])) + " %"),
        ("Power: switching (mW)", f(pa["groups"]["total"]["switching_w"] * 1e3, 3), f(pb["groups"]["total"]["switching_w"] * 1e3, 3),
         f(red(pa["groups"]["total"]["switching_w"], pb["groups"]["total"]["switching_w"])) + " %"),
        ("Power: leakage (µW)", f(pa["leakage_w"] * 1e6, 3), f(pb["leakage_w"] * 1e6, 3), f(red(pa["leakage_w"], pb["leakage_w"])) + " %"),
        ("**Dynamic power (internal + switching), mW**", f(pa["dynamic_w"] * 1e3, 3), f(pb["dynamic_w"] * 1e3, 3),
         "**" + f(red(pa["dynamic_w"], pb["dynamic_w"])) + " %**"),
        ("  of which clock tree (mW)", f(pa["groups"]["clock"]["total_w"] * 1e3, 3), f(pb["groups"]["clock"]["total_w"] * 1e3, 3),
         f(red(pa["groups"]["clock"]["total_w"], pb["groups"]["clock"]["total_w"])) + " %"),
        ("  of which sequential / combinational (mW)",
         f"{f(pa['groups']['sequential']['total_w'] * 1e3, 3)} / {f(pa['groups']['combinational']['total_w'] * 1e3, 3)}",
         f"{f(pb['groups']['sequential']['total_w'] * 1e3, 3)} / {f(pb['groups']['combinational']['total_w'] * 1e3, 3)}", ""),
        ("Dynamic power, PE array cells only (mW, excl. clock tree)", f(pa["array"]["dynamic_w"] * 1e3, 3),
         f(pb["array"]["dynamic_w"] * 1e3, 3), f(red(pa["array"]["dynamic_w"], pb["array"]["dynamic_w"])) + " %"),
        ("**Energy per image, dynamic (nJ)**", f(pa["energy_per_image_dynamic_j"] * 1e9, 2),
         f(pb["energy_per_image_dynamic_j"] * 1e9, 2),
         "**" + f(red(pa["energy_per_image_dynamic_j"], pb["energy_per_image_dynamic_j"])) + " %**"),
        ("Energy per image, total (nJ)", f(pa["energy_per_image_total_j"] * 1e9, 2), f(pb["energy_per_image_total_j"] * 1e9, 2),
         f(red(pa["energy_per_image_total_j"], pb["energy_per_image_total_j"])) + " %"),
        ("Stream cycles for the 8 images (first input → last output)", f"{pa['e2e']['stream_cycles']:,}",
         f"{pb['e2e']['stream_cycles']:,}", ""),
        ("Gate-level sim bit-exact vs golden model (8 Fashion images)", str(pa["e2e"]["bit_exact"]), str(pb["e2e"]["bit_exact"]), ""),
        ("Pins annotated from VCD / unannotated", f"{pa['annotation']['pins_annotated']:,} / {pa['annotation']['pins_unannotated']}",
         f"{pb['annotation']['pins_annotated']:,} / {pb['annotation']['pins_unannotated']}", ""),
    ]
    out = [f"**{period} ns ({1000 / float(period):.1f} MHz), criterion: tt setup, all-corner hold, LibreLane default "
           f"sky130 signoff.** Runs `{ta}`, `{tb}`.", "",
           f"| | {a_build} | {b_build} | reduction |", "|---|---|---|---|"]
    out += [f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |" for r in rows]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    a, b = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("int8", "ternary")
    periods = sys.argv[3:] or ["17.5", "11.5"]
    for p in periods:
        print(table(p, a, b))
