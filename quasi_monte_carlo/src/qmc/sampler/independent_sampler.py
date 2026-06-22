"""
Plain Monte-Carlo control, addressed by path decision.

Stateless hash-based uniforms keyed by (pixel, sample, dimension, seed): a strong
integer mixer (hash_ints) makes each decision an independent draw. Being a pure
function of its inputs, it is reproducible and order-independent, so it plugs into
the exact same addressable interface as the QMC samplers and gives the slope-(-1/2)
baseline to measure them against. Dimension identity is irrelevant to i.i.d.
sampling; it just guarantees distinct decisions get distinct streams.
"""

from qmc.sampler.sobol_core import hash_ints, to_float
from qmc.sampler.sampler import Sampler


class IndependentSampler(Sampler):
    def __init__(self, seed=0):
        self.seed = seed
        self.px = self.py = self.sample_index = 0

    def setup_path(self, px, py, sample_index):
        self.px = int(px)
        self.py = int(py)
        self.sample_index = int(sample_index)

    def _u(self, dim: int) -> float:
        return to_float(hash_ints(self.px, self.py, self.sample_index, dim,
                                  seed=self.seed))

    # ── Sampler primitives ────────────────────────────────────────────────────
    def _one(self, dim: int) -> float:
        return self._u(dim)

    def _pair(self, dim: int) -> tuple[float, float]:
        return self._u(dim), self._u(dim + 1)
