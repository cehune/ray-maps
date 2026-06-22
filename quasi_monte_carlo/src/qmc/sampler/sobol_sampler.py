"""
Owen-scrambled Sobol' sampler, addressed by path decision.

Each decision maps to a fixed logical dimension (the layout lives in sampler.py);
a 2D decision uses two consecutive Sobol' dimensions d, d+1. Dimensions at or past
`max_dim` fall back to an independent uniform — "pad the tail" — so a `max_dim`
cut on a block boundary cleanly Sobol'-izes whole bounces and the dimension sweep
measures how many leading dimensions deserve real Sobol' before padding wins.

Randomization is per pixel: the scramble seed is keyed by (pixel, dimension), so
neighbouring pixels get decorrelated, error-measurable realizations. The sampler
holds no per-draw state — between setup_path() calls every value is a pure
function of (pixel, sample, dimension, seed), so calls are order-independent.
"""

from qmc.sampler.sobol_core import sobol_uint, to_float, hash_ints
from qmc.sampler.scramble import OwenScramble
from qmc.sampler.sampler import Sampler


class SobolSampler(Sampler):
    def __init__(self, dim_lookup, max_dim=64, scrambler=None, seed=0):
        """
        dim_lookup : direction-number table, shape (>=max_dim, 32), from
                     direction_numbers.directions().
        max_dim    : dimensions [0, max_dim) use Sobol'; the rest are padded
                     with an independent uniform.
        scrambler  : a Scrambler instance (default OwenScramble, PBRT-style).
        seed       : selects an independent realization.
        """
        self.dim_lookup = dim_lookup
        if max_dim > dim_lookup.shape[0]:
            raise ValueError("max_dim must be <= the direction-number table size")
        self.max_dim = max_dim
        self.scrambler = scrambler if scrambler is not None else OwenScramble()
        self.seed = seed
        self.px = self.py = self.sample_index = 0

    def setup_path(self, px, py, sample_index):
        self.px = int(px)
        self.py = int(py)
        self.sample_index = int(sample_index)

    def _value(self, dim: int) -> float:
        """Scrambled Sobol' value for this (pixel, sample) at logical `dim`."""
        if dim >= self.max_dim:
            # Past the good Sobol' dimensions: pad with an independent uniform.
            # sample_index is in the hash so the spp samples of a pixel decorrelate
            # (otherwise the padded tail would be identical across them).
            return to_float(hash_ints(self.px, self.py, dim,
                                      self.sample_index, seed=self.seed))
        seed = hash_ints(self.px, self.py, dim, seed=self.seed)
        u = sobol_uint(self.sample_index, dim, self.dim_lookup)
        return to_float(self.scrambler.scramble(u, seed))

    # ── Sampler primitives ────────────────────────────────────────────────────
    def _one(self, dim: int) -> float:
        return self._value(dim)

    def _pair(self, dim: int) -> tuple[float, float]:
        # a 2D decision consumes two consecutive Sobol' dimensions
        return self._value(dim), self._value(dim + 1)
