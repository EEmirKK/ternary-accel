"""Summarise results/accuracy/accuracy_runs.jsonl: mean +- sample std (n-1) over seeds.

    python train/summarize.py            # prints Markdown tables
"""

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHODS = [("positive", "Positive-only scale (hardware)"),
           ("spec", "Symmetric rule (max \\|logit\\|)"),
           ("raw", "Full-width argmax (reference)")]


def ms(xs):
    xs = [100 * x for x in xs]
    return f"{st.mean(xs):.2f} ± {st.stdev(xs):.2f}"


def main():
    runs = [json.loads(line) for line in open(ROOT / "results" / "accuracy" / "accuracy_runs.jsonl")]
    by = defaultdict(list)
    for r in runs:
        by[(r["dataset"], r["precision"])].append(r)
    for k in by:
        by[k].sort(key=lambda r: r["seed"])
    seeds = sorted({r["seed"] for r in runs})
    print(f"Seeds {seeds}; mean ± sample std (n−1) over seeds; accuracy in %.\n")

    print("| Dataset | float (eval) | int8 QAT (float eval) | ternary QAT (float eval) |")
    print("|---|---|---|---|")
    for ds in ("mnist", "fashion"):
        print(f"| {ds} | " + " | ".join(ms([r["acc_float_eval"] for r in by[(ds, p)]])
                                     for p in ("float", "int8", "ternary")) + " |")
    print()

    print("| Dataset | Method | int8 | ternary | int8 − ternary (pt) | ties int8 (per seed) | ties ternary (per seed) |")
    print("|---|---|---|---|---|---|---|")
    for ds in ("mnist", "fashion"):
        for m, name in METHODS:
            a8 = [r["methods"][m]["acc"] for r in by[(ds, "int8")]]
            at = [r["methods"][m]["acc"] for r in by[(ds, "ternary")]]
            gap = [100 * (x - y) for x, y in zip(a8, at)]
            t8 = "/".join(str(r["methods"][m]["ties"]) for r in by[(ds, "int8")])
            tt = "/".join(str(r["methods"][m]["ties"]) for r in by[(ds, "ternary")])
            print(f"| {ds} | {name} | {ms(a8)} | {ms(at)} | {st.mean(gap):+.2f} ± {st.stdev(gap):.2f} | {t8} | {tt} |")
    print()

    print("| Dataset | Precision | top logit clipped, test (per seed): positive / spec | top logit clipped, train: positive / spec | W1 zeros % | W2 zeros % |")
    print("|---|---|---|---|---|---|")
    for ds in ("mnist", "fashion"):
        for p in ("int8", "ternary"):
            rs = by[(ds, p)]
            te = "/".join(str(r["methods"]["positive"]["top_saturated_test"]) for r in rs) + " ; " + \
                 "/".join(str(r["methods"]["spec"]["top_saturated_test"]) for r in rs)
            tr = "/".join(str(r["methods"]["positive"]["top_saturated_train"]) for r in rs) + " ; " + \
                 "/".join(str(r["methods"]["spec"]["top_saturated_train"]) for r in rs)
            z1 = ms([r["w1_zero_frac"] for r in rs])
            z2 = ms([r["w2_zero_frac"] for r in rs])
            print(f"| {ds} | {p} | {te} | {tr} | {z1} | {z2} |")


if __name__ == "__main__":
    main()
