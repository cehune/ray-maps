"""
This is just so I don't have to chagne the non-vectorized sampler too much lol
want a unified itnerface rather than using a billion conditionals, so this just returns random values
"""
from sobol_core import hash_ints
from qmc.sampler.sampler import Sampler
import numpy as np


class IndependantSampler(Sampler):
    def __init__(self, seed=0): 
        self.seed = seed
    def setup_path(self, px, py, sample_index):
        self._rng = np.random.default_rng(hash_ints(px, py, sample_index, seed=self.seed))
    def next_1d(self): return float(self._rng.random())
    def next_2d(self): return tuple(self._rng.random(2))
