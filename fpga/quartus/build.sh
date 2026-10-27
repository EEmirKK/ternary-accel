#!/usr/bin/env bash
# Compiles the DE0-Nano project with a Windows install of Quartus, called from WSL.
#   fpga/quartus/build.sh <int8|ternary>
# Needs QUARTUS_BIN (the Quartus bin64 directory) and QUARTUS_WORK (a work directory on the
# Windows side, because Quartus can't build reliably on \\wsl$ paths). The reports come back
# to results/fpga/<build>/, and the .sof stays in the work directory.
set -euo pipefail
cd "$(dirname "$0")/../.."
build=$1
t=$([ "$build" = ternary ] && echo 1 || echo 0)
Q=${QUARTUS_BIN:?set QUARTUS_BIN to the Quartus bin64 dir}
W=${QUARTUS_WORK:?set QUARTUS_WORK to a Windows-side work dir}/$build
rm -rf "$W"; mkdir -p "$W/src"
cp rtl/*.v fpga/uart_rx.v fpga/uart_tx.v fpga/accel_fpga.v fpga/de0nano_top.v "$W/src/"
sed "s/^set_parameter -name TERNARY 0/set_parameter -name TERNARY $t/" fpga/quartus/de0nano.qsf > "$W/de0nano.qsf"
cp fpga/quartus/de0nano.sdc "$W/"
echo "PROJECT_REVISION = \"de0nano\"" > "$W/de0nano.qpf"
( cd "$W" && "$Q/quartus_sh.exe" --flow compile de0nano -c de0nano ) > "$W/build.log" 2>&1 || { tail -30 "$W/build.log"; exit 1; }
out=results/fpga/$build
mkdir -p $out
cp "$W"/output_files/*.rpt "$W"/output_files/*.summary $out/ 2>/dev/null || true
cp "$W/build.log" $out/
# Keep the Windows work path and the machine name out of the published reports.
wq=$(wslpath -m "$QUARTUS_WORK"); host=$(cmd.exe /c hostname 2>/dev/null | tr -d '\r\n')
sed -i -e "s#$wq/#<quartus_work>/#g" -e "s/$host/<host>/g" $out/*
chmod 644 $out/*
echo "built $build: $(grep -m1 -i 'Total logic elements' $out/de0nano.fit.summary)"
