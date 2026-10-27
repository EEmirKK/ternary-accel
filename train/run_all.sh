#!/usr/bin/env bash
# Trains every dataset, precision and seed with the same recipe, three jobs at a time
# because each peaks at about 1.6 GB of RAM.
#   train/run_all.sh
# Writes results/accuracy/accuracy_runs.jsonl, and the seed-0 integer networks, which the
# RTL end-to-end tests run, to results/accuracy/nets/.
set -euo pipefail
cd "$(dirname "$0")/.."
out=results/accuracy/accuracy_runs.jsonl
mkdir -p results/accuracy/nets results/logs
: > "$out"
jobs=()
for ds in mnist fashion; do
  for prec in float int8 ternary; do
    for seed in 0 1 2; do
      exp=""
      if [[ $prec != float && $seed == 0 ]]; then exp="--export results/accuracy/nets/${ds}_${prec}_s${seed}.npz"; fi
      jobs+=("python train/qat.py --dataset $ds --precision $prec --seed $seed --json $out $exp > results/logs/${ds}_${prec}_s${seed}.log 2>&1")
    done
  done
done
printf '%s\n' "${jobs[@]}" | xargs -P 3 -I{} bash -c '{}'
echo "done: $(wc -l < "$out") runs"
