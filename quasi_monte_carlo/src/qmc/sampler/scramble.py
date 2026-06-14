"""Scramble strategies — the randomization axis, orthogonal to the sampler.

A Scrambler is a *stateless* pure transform ``(value, seed) -> value`` on the
raw 32-bit Sobol' output. All context (pixel, dimension, sample) lives in the
sampler, which derives `seed` and passes it in. Keeping these stateless makes
them trivial to unit-test and to port to elementwise Dr.Jit array code.

Swap the strategy on a sampler to run the core QMC experiment: raw vs
random-digit vs Owen scrambling.
"""

from qmc.sampler.sobol_core import MASK32


def reverse_bits_32(n: int) -> int:
    n = int(n) & MASK32
    n = ((n & 0x55555555) << 1) | ((n >> 1) & 0x55555555)
    n = ((n & 0x33333333) << 2) | ((n >> 2) & 0x33333333)
    n = ((n & 0x0F0F0F0F) << 4) | ((n >> 4) & 0x0F0F0F0F)
    n = ((n & 0x00FF00FF) << 8) | ((n >> 8) & 0x00FF00FF)
    n = ((n << 16) | (n >> 16)) & MASK32
    return n


class Scrambler:
    """Base strategy. Must be a pure function of (v, seed)."""

    def scramble(self, v: int, seed: int) -> int:
        raise NotImplementedError


class NoScramble(Scrambler):
    """Identity — raw, deterministic Sobol' (baseline). Same points every
    pixel, so on its own it produces correlated, structured error; useful as
    a control and for inspecting the bare sequence."""

    def scramble(self, v: int, seed: int) -> int:
        return int(v) & MASK32


class RandomDigitScramble(Scrambler):
    """Random-digit (XOR) scramble. Flips each binary digit independently by
    XORing a per-dimension random constant. Cheap, preserves the net
    property, decorrelates pixels — but leaves the sequence's structured
    intra-sample correlations partly intact (lower quality than Owen)."""

    def scramble(self, v: int, seed: int) -> int:
        return (int(v) ^ int(seed)) & MASK32


class OwenScramble(Scrambler):
    """Owen (nested uniform) scramble via the Laine-Kárras / Burley hash
    (PBRT's FastOwenScramble). Each digit is permuted conditioned on the
    higher-order digits, which removes the structured aliasing that plain
    XOR leaves behind — best convergence, the usual default."""

    def scramble(self, v: int, seed: int) -> int:
        v = reverse_bits_32(v)           # operate MSB-first
        seed = int(seed) & MASK32
        v = (v ^ ((v * 0x3D20ADEA) & MASK32)) & MASK32
        v = (v + seed) & MASK32
        v = (v * (((seed >> 16) | 1) & MASK32)) & MASK32
        v = (v ^ ((v * 0x05526C56) & MASK32)) & MASK32
        v = (v ^ ((v * 0x53A22864) & MASK32)) & MASK32
        return reverse_bits_32(v)
