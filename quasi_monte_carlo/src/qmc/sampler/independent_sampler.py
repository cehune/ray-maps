from qmc.sampler.sobol_core import hash_ints
import numpy as np
from qmc.sampler.sampler import Sampler

class IndependentSampler(Sampler):
    def __init__(self, seed=0):
        self.seed = seed
        self.rng = None
        self.dim = 0
        
    def setup_path(self, px, py, sample_index):
        s = hash_ints(int(px), int(py), int(sample_index), seed=self.seed)
        self.rng = np.random.default_rng(s)
        self.dim = 0

    def next_1d(self) -> float:
        self.dim += 1
        return float(self.rng.random())

    def next_2d(self) -> tuple[float, float]:
        self.dim += 2
        u = self.rng.random(2)
        return float(u[0]), float(u[1])