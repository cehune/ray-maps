"""
Padded Sobol' sampler, addressed by path decision.

Every logical dimension is served by its OWN scrambled, index-shuffled copy of a
low-dimensional (here 2D) Sobol' set. Two distinct decisions therefore draw from
independent streams — there is no joint low-discrepancy across dimensions, which
is exactly what makes the high-dimensional projection problem (and dimension
misalignment) disappear by construction. Each individual 1D/2D decision is still
well stratified. See Burley 2020, "Practical Hash-based Owen Scrambling".

Two independent randomizations per decision, kept distinct on purpose:
  index shuffle : permutes WHICH sample index this dimension consumes.
  value scramble: randomizes the Sobol' OUTPUT bits (the Scrambler member).

Stateless between setup_path(): a value is a pure function of (pixel, sample,
dimension, seed), so calls are order-independent.
"""

from qmc.sampler.sobol_core import sobol_uint, to_float, hash_ints, permutation_element
from qmc.sampler.scramble import OwenScramble
from qmc.sampler.sampler import Sampler


class PaddedSobolSampler(Sampler):
    def __init__(self, dim_lookup, spp, scrambler=None, seed=0):
        """
        dim_lookup : direction-number table (only dims 0,1 are used).
        spp        : must match the renderer's spp (the index-shuffle period).
        scrambler  : a Scrambler instance (default OwenScramble).
        seed       : selects an independent realization.
        """
        self.dim_lookup = dim_lookup
        self.spp = int(spp)
        self.scrambler = scrambler if scrambler is not None else OwenScramble()
        self.seed = seed
        self.px = self.py = self.sample_index = 0

    def setup_path(self, px, py, sample_index):
        self.px = int(px)
        self.py = int(py)
        self.sample_index = int(sample_index)

    def _draw(self, dim: int, n_sub: int):
        """`n_sub` (1 or 2) scrambled values for logical dimension `dim`. The
        dimension only seeds the shuffle/scramble, so distinct dims are
        independent streams of the same 2D base Sobol' set."""
        shuffle_seed  = hash_ints(self.px, self.py, dim, 0, seed=self.seed)
        scramble_seed = hash_ints(self.px, self.py, dim, 1, seed=self.seed)
        # shuffle which Sobol' sample index this dimension consumes
        idx = permutation_element(self.sample_index, self.spp, shuffle_seed)
        out = []
        for k in range(n_sub):
            u = sobol_uint(idx, k, self.dim_lookup)        # base Sobol' dim 0 or 1
            s = hash_ints(scramble_seed, k)                # per-component scramble
            out.append(to_float(self.scrambler.scramble(u, s)))
        return out

    # ── Sampler primitives ────────────────────────────────────────────────────
    def _one(self, dim: int) -> float:
        return self._draw(dim, 1)[0]

    def _pair(self, dim: int) -> tuple[float, float]:
        a, b = self._draw(dim, 2)
        return a, b
