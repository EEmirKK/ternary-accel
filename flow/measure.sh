#!/usr/bin/env bash
# Measures one hardened run (after flow/harden.sh): area from the final layout, a
# gate-level simulation of the routed netlist on Fashion-MNIST test images (checked
# bit-exact), and power from that simulation's switching activity.
#   flow/measure.sh <build> <period_ns> [images=8]
# Needs LIBRELANE_DIR (default ~/librelane) and PDK_ROOT (default ~/.ciel).
# Writes results/asic/<build>_<period>ns_{area,power,metrics}.json.
set -euo pipefail
cd "$(dirname "$0")/.."
build=$1; period=$2; nimg=${3:-8}
tag="${build}_${period}ns"
run=flow/runs/$tag; fin=$(pwd)/$run/final
out=results/asic; gl=$(pwd)/$run/gl
mkdir -p $out $gl
t=$(case "$build" in ternary*) echo 1;; *) echo 0;; esac)
clk_ps=$(python3 -c "print(round($period*1000))")
n=$(echo "$build" | grep -oE '_n[0-9]+' | tr -dc '0-9' || true); n=${n:-4}             # array size from the build name, int8_n8 -> 8
lib=$(ls "${PDK_ROOT:-$HOME/.ciel}"/ciel/sky130/versions/*/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib)
here=$(pwd)
LIBRELANE_DIR=${LIBRELANE_DIR:-$HOME/librelane}
echo "commit $(git rev-parse --short HEAD); measure $tag, $nimg images, N=$n, clk ${clk_ps} ps" > $gl/measure.log

( cd "$LIBRELANE_DIR" && ODB=$fin/odb/accel_top.odb OUT=$here/$out/${tag}_area.json \
  nix-shell --run "openroad -no_init -no_splash -exit $here/flow/area.tcl" ) >> $gl/measure.log 2>&1

rm -f $gl/e2e.jsonl
GL_SIM=icarus GL_RUN=$run GL_TERNARY=$t GL_N=$n GL_CLK_PS=$clk_ps GL_VCD=$gl/activity.vcd \
  E2E_IMAGES=$nimg E2E_OUT=$gl/e2e.jsonl pytest tb/test_gl_runner.py -q -s >> $gl/measure.log 2>&1

( cd "$LIBRELANE_DIR" && LIB=$lib ODB=$fin/odb/accel_top.odb SDC=$fin/sdc/accel_top.sdc \
  SPEF=$fin/spef/nom/accel_top.nom.spef VCD=$gl/activity.vcd SCOPE=accel_top OUTDIR=$gl \
  nix-shell --run "openroad -no_init -no_splash -exit $here/flow/power.tcl" ) >> $gl/measure.log 2>&1
python3 flow/parse_power.py $gl $gl/e2e.jsonl $out/${tag}_power.json

# Keep only the metric groups the summaries and the README use.
python3 - $fin/metrics.json $out/${tag}_metrics.json <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
keep = {k: v for k, v in m.items() if k.startswith(("timing__", "design__", "route__drc", "magic__",
        "klayout__", "antenna__", "power__", "clock__"))}
json.dump(keep, open(sys.argv[2], "w"), indent=1, sort_keys=True)
PY
echo "done $tag"
