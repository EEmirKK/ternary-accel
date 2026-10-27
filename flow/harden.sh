#!/usr/bin/env bash
# Hardens one build at one clock period with LibreLane's Classic flow.
#   flow/harden.sh <build> <period_ns> [tag_suffix]    # build names a flow/<build>.yaml
# Needs a LibreLane checkout in LIBRELANE_DIR (default ~/librelane), run through nix-shell.
# The run goes to flow/runs/<build>_<period>ns, with its log beside it.
set -euo pipefail
cd "$(dirname "$0")"
build=$1; period=$2
tag="${build}_${period}ns${3:+_$3}"
mkdir -p runs
cmd="librelane ${build}.yaml -c CLOCK_PERIOD=${period} --run-tag ${tag} --overwrite --hide-progress-bar"
echo "commit $(git rev-parse --short HEAD); cmd: $cmd" > "runs/${tag}.log"
here=$(pwd)
( cd "${LIBRELANE_DIR:-$HOME/librelane}" && nix-shell --run "cd $here && $cmd" ) >> "runs/${tag}.log" 2>&1
