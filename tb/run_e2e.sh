#!/usr/bin/env bash
# Runs the end-to-end test on the RTL for every build and dataset, four simulations in parallel.
#   tb/run_e2e.sh [images=1000] [variant=base|perf|db]
# Writes results/sim/e2e.jsonl for the base variant and results/sim/e2e_<variant>.jsonl otherwise.
set -euo pipefail
cd "$(dirname "$0")/.."
n=${1:-1000}
variant=${2:-base}
if [ "$variant" = base ]; then out=results/sim/e2e.jsonl; else out=results/sim/e2e_$variant.jsonl; fi
mkdir -p results/logs
: > "$out"
for prec in int8 ternary; do
  for ds in mnist fashion; do
    ( start=$(date +%s)
      E2E_VARIANT=$variant E2E_IMAGES=$n E2E_OUT=$out pytest tb/test_e2e_runner.py -q -s -k "$prec and $ds" \
        > results/logs/e2e_${prec}_${ds}_${variant}.log 2>&1 && st=pass || st=FAIL
      echo "$prec $ds $n images: $st, $(( $(date +%s) - start )) s wall" ) &
  done
done
wait
