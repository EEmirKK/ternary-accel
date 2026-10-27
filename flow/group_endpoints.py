"""Groups the endpoint slacks from flow/endpoints.tcl by the register they belong to.

    python flow/group_endpoints.py <final netlist> <endpoints file>

A register is named after the net its flop's Q pin drives, because synthesis keeps RTL
names on nets rather than on flop instances.
"""
import re
import sys
from collections import defaultdict

netlist, ends = sys.argv[1], sys.argv[2]
q = {}
for m in re.finditer(r"sky130_fd_sc_hd__\w+\s+(\S+)\s*\((.*?)\);", open(netlist).read(), re.S):
    mq = re.search(r"\.Q\(\s*([^)]*?)\s*\)", m.group(2))
    if mq:
        q[m.group(1).lstrip("\\")] = mq.group(1).lstrip("\\")


def group(net: str) -> str:
    if "u_pe" in net or net.startswith("u_array"):
        return "PE array"
    for pre in ("u_acc.", "u_rq.", "u_skew", "u_deskew"):
        if net.startswith(pre):
            base = net.split("[")[0]
            return base if pre == "u_rq." else pre.rstrip(".")
    return re.sub(r"\[\d+\]", "", net)


worst = defaultdict(lambda: (1e9, 0))
for line in open(ends):
    pin, slack = line.split()
    inst = pin.rsplit("/", 1)[0].replace("\\", "")
    g = group(q.get(inst, q.get(inst.split("/")[-1], inst)))
    s, n = worst[g]
    worst[g] = (min(s, float(slack)), n + 1)
for g, (s, n) in sorted(worst.items(), key=lambda kv: kv[1][0])[:12]:
    print(f"  worst slack {s:8.3f} ns  ({n:4d} endpoints)  {g}")
