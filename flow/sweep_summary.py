"""Builds the size-sweep table for int8 and ternary at N = 4, 8 and 16, all at 17.5 ns.

    python flow/sweep_summary.py      # writes results/sweep/sweep.json and sweep.md

Each size gets the synthesis area from its Yosys log, and post-route numbers when a hardened
and measured run exists (results/asic/<build>_17.5ns_*.json). The source of every number is
recorded.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S3, OUT = ROOT / "results" / "asic", ROOT / "results" / "sweep"
RUNS = ROOT / "flow" / "runs"


def build(prec, n):
    return prec if n == 4 else f"{prec}_n{n}"


def synth_area(tag):
    log = RUNS / tag / "06-yosys-synthesis" / "yosys-synthesis.log"
    if not log.exists():
        return None, None
    txt = log.read_text(errors="replace")
    top = re.findall(r"Chip area for top module '\\accel_top': ([\d.]+)", txt)
    arr = re.findall(r"Chip area for module '[^']*systolic_array': ([\d.]+)", txt)
    return (float(top[-1]) if top else None), (float(arr[-1]) if arr else None)


def limiter(tag):
    """Worst-slack register group at nominal tt, if the endpoint analysis was run."""
    f = ROOT / "results" / "sweep" / f"limiter_{tag}.txt"
    return f.read_text().strip().splitlines()[0] if f.exists() else None


rows = []
for n in (4, 8, 16):
    for prec in ("int8", "ternary"):
        b = build(prec, n)
        tag = f"{b}_17.5ns"
        r = dict(N=n, precision=prec, build=b)
        s_top, s_arr = synth_area(tag)
        if s_top is None:
            s_top, s_arr = synth_area(f"{b}_synth")
            r["synth_source"] = f"{b}_synth (synthesis only)"
        else:
            r["synth_source"] = tag
        r["synth_area_um2"], r["synth_array_area_um2"] = s_top, s_arr
        if (S3 / f"{tag}_power.json").exists():
            a = json.loads((S3 / f"{tag}_area.json").read_text())
            p = json.loads((S3 / f"{tag}_power.json").read_text())
            srch = {json.loads(l)["tag"]: json.loads(l) for l in open(S3 / "search.jsonl")}.get(tag, {})
            r.update(source="post-route + gate-level power", core_area_um2=a["logic_area_um2"],
                     array_area_um2=a["array_logic_area_um2"], die_area_um2=a["die_area_um2"],
                     cells=a["logic_count"], dyn_power_mw=p["dynamic_w"] * 1e3,
                     energy_nj=p["energy_per_image_dynamic_j"] * 1e9, cycles_8img=p["e2e"]["stream_cycles"],
                     bit_exact=p["e2e"]["bit_exact"], unannotated=p["annotation"]["pins_unannotated"],
                     closes=srch.get("closes"), tt_wns_ns=srch.get("tt_setup_wns_ns"),
                     fmax_tt_mhz=1000.0 / (17.5 - srch["tt_setup_wns_ns"]) if srch else None,
                     limiter=limiter(tag))
        else:
            r.update(source="synthesis only (Yosys, no placement/routing/timing)")
        rows.append(r)
rows[0]["caption"] = "Yosys synthesis cell area, sky130 HD standard-cell library, 17.5 ns flow, all sizes and both builds."
rows[0]["caption_post"] = ("Post-route cell area (excl. fill/decap/tap), sky130 HD, LibreLane, same flow and 17.5 ns constraint for all. "
                           "16×16 exceptions to full signoff: int8 misses ss hold on input-port paths only; ternary has 1 antenna net.")
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "sweep.json").write_text(json.dumps(rows, indent=1))


def fmt(v, d=0):
    return "—" if v is None else f"{v:,.{d}f}"


lines = ["| N | Build | Source | Cell area, whole core (µm²) | Cell area, PE array (µm²) | Synthesis area (µm²) | "
         "1 / tt critical path (MHz), derived from the 17.5 ns WNS | Worst register group at tt | Dynamic power (mW) | Energy / image (nJ) | Cycles, 8 images |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    lines.append(f"| {r['N']} | {r['precision']} | {r['source']} | {fmt(r.get('core_area_um2'))} | "
                 f"{fmt(r.get('array_area_um2'))} | {fmt(r['synth_area_um2'])} | {fmt(r.get('fmax_tt_mhz'), 1)} | {r.get('limiter') or '—'} | "
                 f"{fmt(r.get('dyn_power_mw'), 3)} | {fmt(r.get('energy_nj'), 1)} | {fmt(r.get('cycles_8img'))} |")
lines += ["", "| N | Whole-core cell area reduction | PE-array reduction | Dynamic energy / image reduction | Source |",
          "|---|---|---|---|---|"]
for n in (4, 8, 16):
    i, t = [r for r in rows if r["N"] == n]
    if i.get("core_area_um2") and t.get("core_area_um2"):
        lines.append(f"| {n} | {100 * (1 - t['core_area_um2'] / i['core_area_um2']):.1f} % | "
                     f"{100 * (1 - t['array_area_um2'] / i['array_area_um2']):.1f} % | "
                     f"{100 * (1 - t['energy_nj'] / i['energy_nj']):.1f} % | post-route |")
    elif i["synth_area_um2"] and t["synth_area_um2"]:
        lines.append(f"| {n} | {100 * (1 - t['synth_area_um2'] / i['synth_area_um2']):.1f} % (synthesis) | "
                     f"{100 * (1 - t['synth_array_area_um2'] / i['synth_array_area_um2']):.1f} % (synthesis) | — | "
                     f"synthesis only |")
(OUT / "sweep.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
