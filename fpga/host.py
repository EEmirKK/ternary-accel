"""Runs the MLP on the DE0-Nano over the UART and checks every logit bit-for-bit against the model.

    python fpga/host.py --port COM5 --precision int8 --dataset fashion --images 1000

It sends the beat program once, then batches of test images, and records the on-chip cycle and
ACT-beat counters of each batch. The summary goes to results/fpga/board_<precision>_<dataset>.json
unless --out is given. WSL2 can't open COM ports, so run it with Windows Python.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "model"), str(ROOT / "train"), str(ROOT / "fpga")]
import accel_model as am  # noqa: E402
import data  # noqa: E402
import mlp  # noqa: E402
import protocol  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--precision", choices=["int8", "ternary"], default="int8")
    ap.add_argument("--dataset", choices=["mnist", "fashion"], default="fashion")
    ap.add_argument("--images", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", type=Path, help="default: results/fpga/board_<precision>_<dataset>.json")
    args = ap.parse_args()
    args.out = args.out or ROOT / "results" / "fpga" / f"board_{args.precision}_{args.dataset}.json"

    import serial  # from the pyserial package

    p = am.AccelParams(ternary=args.precision == "ternary")
    net = mlp.load_net(ROOT / "results" / "accuracy" / "nets" / f"{args.dataset}_{args.precision}_s0.npz")
    words, table = protocol.build_program(p, net)
    _, _, xte, yte = data.load(args.dataset)
    mt1 = table[1][2]

    with serial.Serial(args.port, args.baud, timeout=10) as ser:
        t0 = time.time()
        ser.write(protocol.cmd_program(words))
        ser.write(protocol.cmd_table(table))
        ser.flush()
        t_load = time.time() - t0
        print(f"program: {len(words)} words sent in {t_load:.1f} s")
        mismatches, n, correct, runs = 0, 0, 0, []
        for i in range(0, args.images, args.batch):
            x = xte[i:i + args.batch]
            B = len(x)
            t1 = time.time()
            ser.write(protocol.cmd_run(mlp.to_input(x)))
            buf = ser.read(protocol.reply_len(B, mt1))
            if len(buf) != protocol.reply_len(B, mt1):
                raise SystemExit(f"timeout: got {len(buf)} bytes")
            logits, cycles, acts = protocol.parse_reply(buf, B, mt1)
            ref = mlp.int_forward(p, net, x)
            bad = int((logits != ref).any(axis=1).sum())
            mismatches += bad
            n += B
            acc, _ = mlp.accuracy_int(logits, yte[i:i + B])
            correct += round(acc * B)
            runs.append(dict(batch=B, cycles=cycles, act_beats=acts, seconds=time.time() - t1,
                             images_not_bit_exact=bad))
            print(f"images {i}-{i + B - 1}: {'bit-exact' if not bad else f'{bad} MISMATCH'}; "
                  f"{cycles} cycles ({cycles / B:.0f}/image), utilisation {acts / cycles:.3f}")
    summary = dict(precision=args.precision, dataset=args.dataset, images=n, images_bit_exact=n - mismatches,
                   correct=correct, program_words=len(words), program_load_s=t_load, baud=args.baud, runs=runs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"{n - mismatches}/{n} images bit-exact vs model; written {args.out}")


if __name__ == "__main__":
    main()
