"""Turns the flow/power.tcl reports into JSON.

    python flow/parse_power.py <outdir> <e2e record file> <out.json>

It records the design totals by group (sequential, combinational, clock), each split into
internal, switching and leakage power, and the PE array's total over every cell under
u_array/, which leaves out the clock tree because its buffers sit at the top level. OpenSTA
averages activity over the whole VCD, so energy per image is the average power times the
simulated time, divided by the number of images.
"""

import json
import re
import sys
from pathlib import Path

outdir, e2e_file, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3]

groups = {}
for line in (outdir / "power.rpt").read_text().splitlines():
    m = re.match(r"^(Sequential|Combinational|Clock|Macro|Pad|Total)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)", line)
    if m:
        groups[m.group(1).lower()] = dict(zip(("internal_w", "switching_w", "leakage_w", "total_w"),
                                              map(float, m.groups()[1:])))

arr = dict(internal_w=0.0, switching_w=0.0, leakage_w=0.0, total_w=0.0, cells=0)
for line in (outdir / "array_power.rpt").read_text().splitlines():
    f = line.split()
    if len(f) == 5 and f[4].startswith("u_array/"):
        for k, v in zip(("internal_w", "switching_w", "leakage_w", "total_w"), f[:4]):
            arr[k] += float(v)
        arr["cells"] += 1

ann = dict(re.findall(r"^(vcd|unannotated)\s+(\d+)", (outdir / "annotation.rpt").read_text(), re.M))
rec = json.loads(Path(e2e_file).read_text().strip().splitlines()[-1])

t = groups["total"]
dyn = t["internal_w"] + t["switching_w"]
arr_dyn = arr["internal_w"] + arr["switching_w"]
sim_s = rec["sim_time_ns"] * 1e-9
n = rec["images"]
res = dict(
    groups=groups,
    total_w=t["total_w"], dynamic_w=dyn, leakage_w=t["leakage_w"],
    array=dict(arr, dynamic_w=arr_dyn),
    annotation=dict(pins_annotated=int(ann["vcd"]), pins_unannotated=int(ann["unannotated"])),
    e2e=rec,
    vcd_time_s=sim_s,
    energy_per_image_total_j=t["total_w"] * sim_s / n,
    energy_per_image_dynamic_j=dyn * sim_s / n,
    group_sum_check=abs(sum(g["total_w"] for k, g in groups.items() if k != "total") - t["total_w"]) / t["total_w"],
)
Path(out).write_text(json.dumps(res, indent=2))
print(json.dumps({k: res[k] for k in ("total_w", "dynamic_w", "leakage_w", "energy_per_image_total_j",
                                      "energy_per_image_dynamic_j", "annotation", "group_sum_check")}))
print("array:", json.dumps(res["array"]))
