"""Bit-exact integer model of the accelerator, and the encoder that turns a layer into
the beat stream the host sends. It mirrors the RTL, including the widths inside the
array and in the accumulators.

W is K x M (int8, or -1/0/+1 for ternary), x is B x K int8 activations with one row
per image, the bias has M entries in the accumulator range, and outputs are B x M int8.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# in_type codes of the input beats
T_CONFIG, T_BIAS, T_WEIGHT, T_ACT = 0, 1, 2, 3


def clog2(x: int) -> int:
    return (x - 1).bit_length()


@dataclass(frozen=True)
class AccelParams:
    """Build parameters, with every width derived exactly as in the RTL."""

    N: int = 4
    ternary: bool = False
    B_MAX: int = 8
    K_MAX: int = 784

    @property
    def weight_w(self) -> int:
        return 2 if self.ternary else 8

    @property
    def prod_w(self) -> int:
        # The ternary PE computes +-a, and -(-128) = +128 needs 9 bits.
        return 9 if self.ternary else 16

    @property
    def psum_w(self) -> int:
        return self.prod_w + clog2(self.N)

    @property
    def acc_w(self) -> int:
        # K_MAX products of at most 2^14 (int8) or 2^7 (ternary), a sign bit, one bit for the bias.
        prod_log = 7 if self.ternary else 14
        return clog2(self.K_MAX) + prod_log + 2

    @property
    def data_w(self) -> int:
        return 8 * self.N

    @property
    def weight_beats(self) -> int:
        # A ternary beat carries four rows of 2-bit weights.
        return (self.N + 3) // 4 if self.ternary else self.N

    @property
    def bias_beats(self) -> int:
        return (self.acc_w + 7) // 8

    @property
    def s_max(self) -> int:
        # The largest shift whose rounding constant still fits the requantiser.
        return self.acc_w + 16


def fits_signed(v, bits: int) -> bool:
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    v = np.asarray(v)
    return bool(np.all((v >= lo) & (v <= hi)))


def requant(acc, m0: int, s: int, relu: bool):
    """(acc*M0 + 2^(s-1)) >> s, clamped to [0, 127] with ReLU or [-128, 127] without.
    Python ints are used, so nothing can overflow."""
    assert 0 <= m0 < (1 << 16), "M0 is 16-bit unsigned"
    acc = np.asarray(acc, dtype=object)
    rnd = (1 << (s - 1)) if s > 0 else 0
    t = (acc * m0 + rnd) >> s  # >> on Python ints is an arithmetic (floor) shift
    lo = 0 if relu else -128
    t = np.minimum(np.maximum(t, lo), 127)
    return t.astype(np.int64)


def pad_to(n: int, mult: int) -> int:
    return -(-n // mult) * mult


def check_layer_args(p: AccelParams, W, x, bias, m0, s):
    K, M = W.shape
    B = x.shape[0]
    assert x.shape[1] == K
    assert bias.shape == (M,)
    assert 1 <= K <= p.K_MAX, f"K={K} outside 1..{p.K_MAX}"
    assert 1 <= B <= p.B_MAX, f"B={B} outside 1..{p.B_MAX}"
    assert 0 <= s <= p.s_max, f"s={s} outside 0..{p.s_max}"
    assert 0 <= m0 < (1 << 16)
    assert fits_signed(x, 8), "activations must be int8"
    if p.ternary:
        assert np.all(np.isin(W, (-1, 0, 1))), "ternary weights must be in {-1,0,+1}"
    else:
        assert fits_signed(W, 8), "int8 weights out of range"
    assert fits_signed(bias, p.acc_w), "bias does not fit ACC_W"


def _pad_layer(p: AccelParams, W, x, bias):
    K, M = W.shape
    B = x.shape[0]
    Kp, Mp = pad_to(K, p.N), pad_to(M, p.N)
    Wp = np.zeros((Kp, Mp), dtype=np.int64)
    Wp[:K, :M] = W
    xp = np.zeros((B, Kp), dtype=np.int64)
    xp[:, :K] = x
    biasp = np.zeros(Mp, dtype=np.int64)
    biasp[:M] = bias
    return Wp, xp, biasp


def layer_acc(p: AccelParams, W, x, bias, m0: int = 1, s: int = 0):
    """Accumulator values (B x M), computed tile by tile with every width asserted. Also
    returns the largest magnitudes seen, for the testbench's overflow checks."""
    W = np.asarray(W, dtype=np.int64)
    x = np.asarray(x, dtype=np.int64)
    bias = np.asarray(bias, dtype=np.int64)
    check_layer_args(p, W, x, bias, m0, s)
    K, M = W.shape
    N = p.N
    Wp, xp, biasp = _pad_layer(p, W, x, bias)
    Kp, Mp = Wp.shape
    acc_all = np.zeros((x.shape[0], Mp), dtype=np.int64)
    max_psum = 0
    max_acc = 0
    for mt in range(Mp // N):
        cols = slice(mt * N, mt * N + N)
        acc = np.tile(biasp[cols], (x.shape[0], 1))
        for kt in range(Kp // N):
            rows = slice(kt * N, kt * N + N)
            # the column sums leaving the bottom of the array
            psum = xp[:, rows] @ Wp[rows, cols]
            assert fits_signed(psum, p.psum_w), "in-array partial sum overflow"
            max_psum = max(max_psum, int(np.abs(psum).max()))
            acc = acc + psum
            assert fits_signed(acc, p.acc_w), "accumulator overflow"
            max_acc = max(max_acc, int(np.abs(acc).max()))
        acc_all[:, cols] = acc
    return acc_all[:, :M], {"max_psum": max_psum, "max_acc": max_acc}


def layer_tiled(p: AccelParams, W, x, bias, m0: int, s: int, relu: bool):
    """One layer exactly as the hardware computes it. Returns (y, stats), y being B x M int8."""
    acc, stats = layer_acc(p, W, x, bias, m0, s)
    return requant(acc, m0, s, relu), stats


def layer_int(p: AccelParams, W, x, bias, m0: int, s: int, relu: bool, chunk: int = 512):
    """Fast equivalent of layer_tiled for any number of images, since batching doesn't
    change the result. Integer addition is exact, so this gives the same accumulators,
    and it checks the same bounds: every column sum fits PSUM_W, and the running total
    stays within ACC_W after every input tile."""
    W = np.asarray(W, dtype=np.int64)
    x = np.asarray(x, dtype=np.int64)
    bias = np.asarray(bias, dtype=np.int64)
    check_layer_args(p, W, x[:1], bias, m0, s)
    assert fits_signed(x, 8), "activations must be int8"
    Wp, xp, biasp = _pad_layer(p, W, x, bias)
    N = p.N
    Kp, Mp = Wp.shape
    Wt = Wp.reshape(Kp // N, N, Mp)
    acc = np.empty((xp.shape[0], Mp), dtype=np.int64)
    # Chunked over images, because the per-tile array is images x k_tiles x Mp int64.
    for i in range(0, xp.shape[0], chunk):
        xc = xp[i:i + chunk].reshape(-1, Kp // N, N)
        psum = np.einsum("bkn,knm->bkm", xc, Wt)
        assert fits_signed(psum, p.psum_w), "in-array partial sum overflow"
        acc_run = biasp + np.cumsum(psum, axis=1)
        assert fits_signed(acc_run, p.acc_w), "accumulator overflow"
        acc[i:i + chunk] = acc_run[:, -1]
    return requant(acc[:, :W.shape[1]], m0, s, relu)


def choose_scale(scale: float, s_max: int):
    """Writes a positive scale as M0 / 2^s with a 16-bit M0, taking the largest s that
    fits, for the most precision."""
    assert scale > 0
    for s in range(s_max, -1, -1):
        m0 = int(round(scale * (1 << s)))
        if m0 < (1 << 16):
            return max(m0, 1), s
    raise ValueError(f"scale {scale} too large to represent")


# Host-side stream encoding. A beat is (in_type, in_data, in_last), with lane i of
# in_data at bits [8i+7:8i].

def pack_lanes(vals, lane_bits: int = 8) -> int:
    word = 0
    mask = (1 << lane_bits) - 1
    for i, v in enumerate(vals):
        word |= (int(v) & mask) << (lane_bits * i)
    return word


def unpack_lanes(word: int, n: int, lane_bits: int = 8, signed: bool = True):
    mask = (1 << lane_bits) - 1
    out = []
    for i in range(n):
        v = (word >> (lane_bits * i)) & mask
        if signed and v >= 1 << (lane_bits - 1):
            v -= 1 << lane_bits
        out.append(v)
    return out


def config_word(m0: int, s: int, relu: bool, B: int) -> int:
    """CONFIG beat: [15:0] M0, [21:16] s, [22] ReLU enable, [31:24] B."""
    assert 0 <= m0 < (1 << 16) and 0 <= s < 64 and 1 <= B < 256
    return m0 | (s << 16) | (int(relu) << 22) | (B << 24)


def status_beat():
    """A STATUS request, which is a CONFIG beat with B = 0 (PERF_COUNTERS builds only)."""
    return (T_CONFIG, 0, False)


def bias_beats(p: AccelParams, bias_tile):
    """Beat j carries byte j (little-endian, two's complement) of each output's bias."""
    beats = []
    for j in range(p.bias_beats):
        beats.append(pack_lanes([(int(b) >> (8 * j)) & 0xFF for b in bias_tile]))
    return beats


def weight_beats(p: AccelParams, wtile):
    """wtile is N x N, rows indexed by K and columns by M. An int8 beat r holds row r.
    A ternary beat j holds rows 4j to 4j+3, with row q, column c at bit 2*(q*N + c),
    coded 00 = 0, 01 = +1, 11 = -1."""
    N = p.N
    if not p.ternary:
        return [pack_lanes(wtile[r]) for r in range(N)]
    beats = []
    for j in range(p.weight_beats):
        fields = []
        for q in range(4):
            r = 4 * j + q
            fields.extend(wtile[r] if r < N else [0] * N)
        beats.append(pack_lanes(fields, lane_bits=2))
    return beats


def encode_layer(p: AccelParams, W, x, bias, m0: int, s: int, relu: bool,
                 send_config: bool = True):
    """Beat list for one layer, in host tile-loop order (CONFIG, BIAS, WEIGHT, ACT)."""
    W = np.asarray(W, dtype=np.int64)
    x = np.asarray(x, dtype=np.int64)
    bias = np.asarray(bias, dtype=np.int64)
    check_layer_args(p, W, x, bias, m0, s)
    B = x.shape[0]
    N = p.N
    Wp, xp, biasp = _pad_layer(p, W, x, bias)
    Kp, Mp = Wp.shape

    beats = []
    if send_config:
        beats.append((T_CONFIG, config_word(m0, s, relu, B), False))
    n_kt = Kp // N
    for mt in range(Mp // N):
        cols = slice(mt * N, mt * N + N)
        for w in bias_beats(p, biasp[cols]):
            beats.append((T_BIAS, w, False))
        for kt in range(n_kt):
            rows = slice(kt * N, kt * N + N)
            for w in weight_beats(p, Wp[rows, cols]):
                beats.append((T_WEIGHT, w, False))
            for b in range(B):
                last = kt == n_kt - 1 and b == B - 1
                beats.append((T_ACT, pack_lanes(xp[b, rows]), last))
    return beats


def decode_outputs(p: AccelParams, words, M: int, B: int):
    """Rebuilds the B x M outputs from the output beats, which come per output tile, one per image."""
    N = p.N
    Mp = pad_to(M, N)
    assert len(words) == B * (Mp // N), f"expected {B * (Mp // N)} beats, got {len(words)}"
    y = np.zeros((B, Mp), dtype=np.int64)
    i = 0
    for mt in range(Mp // N):
        for b in range(B):
            y[b, mt * N:(mt + 1) * N] = unpack_lanes(words[i], N)
            i += 1
    return y[:, :M]
