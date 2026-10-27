"""The host-to-FPGA byte protocol (see fpga/accel_fpga.v). The cocotb test and the host script
share it, so the board gets exactly what the simulation was given. The beat program is the
BIAS and WEIGHT beats that model.accel_model.encode_layer produces for each layer, in order.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
import accel_model as am  # noqa: E402
import mlp  # noqa: E402


def build_program(p: am.AccelParams, net: dict):
    """Returns (program words, layer table entries [(cfg, k_tiles, m_tiles, base)])."""
    words, table = [], []
    for L in mlp.layers(net):
        K, M = L["W"].shape
        dummy = np.zeros((1, K), dtype=np.int64)
        beats = am.encode_layer(p, L["W"], dummy, L["bias"], L["m0"], L["s"], L["relu"])
        base = len(words)
        words += [w for t, w, _ in beats if t in (am.T_BIAS, am.T_WEIGHT)]
        cfg = am.config_word(L["m0"], L["s"], L["relu"], 1)        # B is set per run
        table.append((cfg, am.pad_to(K, p.N) // p.N, am.pad_to(M, p.N) // p.N, base))
    return words, table


def cmd_program(words) -> bytes:
    return b"P" + struct.pack("<I", len(words)) + b"".join(struct.pack("<I", w & 0xFFFFFFFF) for w in words)


def cmd_table(table) -> bytes:
    return b"L" + b"".join(struct.pack("<IHHI", cfg, kt, mt, base) for cfg, kt, mt, base in table)


def cmd_run(x_q: np.ndarray) -> bytes:
    """x_q: B x K int8 activations (pixel >> 1), K a multiple of 4."""
    B, K = x_q.shape
    assert K % 4 == 0 and 1 <= B <= 255
    return b"R" + bytes([B]) + (np.asarray(x_q, dtype=np.int64) & 0xFF).astype(np.uint8).tobytes()


def reply_len(B: int, m_tiles1: int) -> int:
    return 1 + B * m_tiles1 * 4 + 8


def parse_reply(buf: bytes, B: int, m_tiles1: int, M: int = 10):
    """Returns (logits B x M int8, total_cycles, act_beats)."""
    assert buf[0:1] == b"D", f"bad reply header {buf[:1]!r}"
    n = B * m_tiles1 * 4
    raw = np.frombuffer(buf[1:1 + n], dtype=np.int8).reshape(B, m_tiles1 * 4)
    cycles, acts = struct.unpack("<II", buf[1 + n:1 + n + 8])
    return raw[:, :M].astype(np.int64), cycles, acts
