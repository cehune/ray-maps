"""
Randomization should happen per pixel. Seeds are keyed by the pixel and dimension
but within it we use the same style of permutation. Higher dimensions have worse
2D projections because we can't gaurantee the same good low discrepency without
also having a crazy amount of samples that scale with it. 
"""

from .sobol_core import sobol_uint, to_float, hash_ints
from .scramble import OwenScramble


class SobolSampler:
    def __init__(self, dim_lookup, scrambler=None, seed=0):
        """
        dim_lookup        : direction-number table, shape (max_dim, 32), from
                    load_direction_numbers(). Load with direction_numbers.py!!!!!
                    from joe and kuo, effectively the same as your matrices but jsut the firs
                    t row for each (assume that it's row not column here)
        scrambler : a Scrambler instance (default OwenScramble). Kinda follows PBRT
        seed      : global seed selecting an independent realization.
        """ 
        self.dim_lookup = dim_lookup
        self.max_dim = dim_lookup.shape[0]
        # idc, we always need one to gaurantee that we can measure varaicne
        self.scrambler = scrambler if scrambler is not None else OwenScramble()
        self.seed = seed # just a number man, 0 is honestly probbaly always fine since
        # permutations by the scrambling methods should be enough

        self.px = self.py = 0
        self.sample_index = 0
        self.dim = 0

    def init(self, px, py, sample_index):
        """Call once per path, before tracing begins. (aka each sample per pixel will have xyz samples)"""
        self.px = int(px)
        self.py = int(py)
        self.sample_index = int(sample_index)
        self.dim = 0

    def _value(self, dim: int) -> float:
        """
        hash to get a seed deterministically for all paths for a given pixel
        the val is the deterministic sobol value
        the scramble gaurantees that all indexes for a dimension in your path
        because its based on your seed, will still have the discrepency. 

        This of course relies on your scrambling being "good" and following the low
        discrepency rules
        
        """
        if dim >= self.max_dim:
            raise IndexError(
                f"Sobol dimension {dim} exceeds table max_dim {self.max_dim}; "
                f"load more dimensions or use the padded sampler.")
        seed = hash_ints(self.px, self.py, dim, seed=self.seed)
        val_u =  sobol_uint(self.sample_index, dim, self.dim_lookup)
        return to_float(self.scrambler.scramble(val_u, seed))

    def next_1d(self) -> float:
        """ consume one dimen, returned value is obv already mapped [0, 1) """
        d = self.dim
        self.dim += 1
        return self._value(d)

    def next_2d(self) -> tuple[float, float]:
        """
        consumes and returnes two dimensions
         sample index has to be the same for both, which makes sense because ie for a pixel
        jitter, the sample for the x and y  adjustment is linked!
        """
        d = self.dim
        self.dim += 2
        return self._value(d), self._value(d + 1)
