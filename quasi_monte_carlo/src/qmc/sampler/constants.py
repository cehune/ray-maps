from qmc.sampler.sobol_sampler import SobolSampler
from qmc.sampler.padded_sobol_sampler import PaddedSobolSampler
from qmc.sampler.scramble import NoScramble, RandomDigitScramble, OwenScramble

SCRAMBLERS = {
    'none': NoScramble,
    'xor':  RandomDigitScramble,
    'owen': OwenScramble,
}
SAMPLERS = {
    'sobol': SobolSampler,
    'padded_sobol': PaddedSobolSampler
}