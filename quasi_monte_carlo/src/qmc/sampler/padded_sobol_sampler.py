"""

Padded

Only draw from first 2 dimensions and assume every dimension request is mostly well behaved 
By using the mix, we shuffle the index by using permutation_element from the sobol_core. 

The idea is that we are getting a unique mapping from an index and dimension such that when
we feed it in, we get some index. Then we can pretend like its dimen 0, so dimen 12 with index 5
might map to an index 182, which we use from either the first or second run. This gives the same
tms net gaurantee over enough samples, and a gaurantee if we assume that every random number is 
isolated. 

We avoid the degraded high dimension sobol projections, though across dimensions there is
no actual relationship. Usually this is a bit better anyways. 


Two independent randomizations per request, kept distinct on purpose:
  index shuffle  
  
  This permutes WHICH sample index this dimension uses
  
  value scramble 
  
  This randomizes the Sobol' OUTPUT bits (the Scrambler member).
"""

from qmc.sampler.sobol_core import sobol_uint, to_float, hash_ints, permutation_element
from qmc.sampler.scramble import OwenScramble
from qmc.sampler.sampler import Sampler


class PaddedSobolSampler(Sampler):
    def __init__(self, dim_lookup, spp, scrambler=None, seed=0):
        """
        dim_lookup        : direction-number table, shape (max_dim, 32), from
                    load_direction_numbers(). Load with direction_numbers.py!!!!!
                    from joe and kuo, effectively the same as your matrices but jsut the firs
                    t row for each (assume that it's row not column here)
        spp.      : must match the renderers spp, basically how many per pixel
        scrambler : a Scrambler instance (default OwenScramble). Kinda follows PBRT
        seed      : global seed selecting an independent realization.
        """
        self.dim_lookup = dim_lookup
        self.spp = int(spp)
        self.scrambler = scrambler if scrambler is not None else OwenScramble()
        self.seed = seed

        self.px = self.py = 0
        self.sample_index = 0
        self.dim = 0

    def setup_path(self, px, py, sample_index):
        """ call once per path, before tracing begins."""
        self.px = int(px)
        self.py = int(py)
        self.sample_index = int(sample_index)
        self.dim = 0

    def _draw(self, n_sub: int):
        """
        n_sub can be 1 or 2, and of course we return the same amount of sobol
        scrambled values!

        store the dimension still because that will be used for the seed
        """
        shuffle_seed  = hash_ints(self.px, self.py, self.dim, 0, seed=self.seed)
        scramble_seed = hash_ints(self.px, self.py, self.dim, 1, seed=self.seed)
        self.dim += n_sub

        # shuffle which Sobol' sample index this padded dimension consumes,
        # so different dimensions aren't locked to the same point.
        # therefore, pixel, dimension, and sample index all contribute so that we
        # deterministicalyl have unique scramble
        idx = permutation_element(self.sample_index, self.spp, shuffle_seed)

        out = []
        for k in range(n_sub):
            u = sobol_uint(idx, k, self.dim_lookup)               # Sobol' dim 0 or 1
            s = hash_ints(scramble_seed, k)              # per-component scramble
            out.append(to_float(self.scrambler.scramble(u, s)))
        return out

    def next_1d(self) -> float:
        """ consume one dimen, returned value is obv already mapped [0, 1) """
        return self._draw(1)[0]

    def next_2d(self) -> tuple[float, float]:
        """consume two dimensions, return (u,v) in [0,1)^2 — a coherent 2D
        Sobol' sample from dims 0 and 1."""
        a, b = self._draw(2)
        return a, b
