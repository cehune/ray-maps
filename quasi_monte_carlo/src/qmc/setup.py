# src/qmc/setup.py
from qmc.sampler.constants import SAMPLERS, SCRAMBLERS
from qmc.sampler.direction_numbers import directions
from qmc.sampler.independent_sampler import IndependentSampler

def build_sampler(name, spp, max_dim=32, scramble_method="owen", seed=0):
    """ 
    construct a QMC sampler.
    `name` in {'sobol','padded_sobol','independent'}
    """
    if name == "independent":
        return IndependentSampler(seed=seed)

    if name not in SAMPLERS:
        raise ValueError(f"unsupported sampler {name!r}; choose from "
                         f"{list(SAMPLERS) + ['independent']}")
    if scramble_method not in SCRAMBLERS:
        raise ValueError(f"unknown scramble {scramble_method!r}; choose from {list(SCRAMBLERS)}")

    dim_lookup = directions(max_dim)
    scrambler  = SCRAMBLERS[scramble_method]()
    sampler    = SAMPLERS[name]
    if name == "padded_sobol":
        return sampler(dim_lookup, spp, scrambler=scrambler, seed=seed)    # padded needs spp, ignores max_dim
    # keep the sampler's dimension cap in sync with the table we built
    return sampler(dim_lookup, max_dim=max_dim, scrambler=scrambler, seed=seed)