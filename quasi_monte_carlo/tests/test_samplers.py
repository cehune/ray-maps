"""SobolSampler / PaddedSobolSampler correctness.

Strongest check is the (0, m, 2)-net property: the first 2^m two-dimensional
points must land exactly one per cell of every 2^a x 2^(m-a) dyadic grid.
Owen and random-digit scrambling both preserve this; a broken generator or
scramble shifts points off the dyadic lattice and the test fails.
"""

import numpy as np
import pytest

from qmc.sampler.sobol_sampler import SobolSampler
from qmc.sampler.padded_sobol_sampler import PaddedSobolSampler
from qmc.sampler.scramble import NoScramble, RandomDigitScramble, OwenScramble

SCRAMBLERS = [NoScramble, RandomDigitScramble, OwenScramble]
M = 8                      # 2^8 = 256 points
N = 1 << M


def _is_net(points, m):
    """True iff `points` form a (0, m, 2)-net (one point per dyadic box)."""
    for a in range(m + 1):
        nx, ny = 1 << a, 1 << (m - a)
        grid = np.zeros((nx, ny), dtype=int)
        for x, y in points:
            grid[min(int(x * nx), nx - 1), min(int(y * ny), ny - 1)] += 1
        if not np.all(grid == 1):
            return False
    return True


def _collect(sampler):
    pts = np.empty((N, 2))
    for i in range(N):
        sampler.setup_path(3, 4, i)
        pts[i] = sampler.next_2d()
    return pts


@pytest.mark.parametrize("scrambler", SCRAMBLERS)
def test_sobol_net_property(v64, scrambler):
    pts = _collect(SobolSampler(v64, scrambler=scrambler(), seed=7))
    assert _is_net(pts, M)


@pytest.mark.parametrize("scrambler", SCRAMBLERS)
def test_padded_net_property(v64, scrambler):
    pts = _collect(PaddedSobolSampler(v64, spp=N, scrambler=scrambler(), seed=7))
    assert _is_net(pts, M)


@pytest.mark.parametrize("scrambler", SCRAMBLERS)
def test_values_in_unit_interval(v64, scrambler):
    pts = _collect(SobolSampler(v64, scrambler=scrambler(), seed=1))
    assert pts.min() >= 0.0 and pts.max() < 1.0


def test_no_scramble_is_raw_sobol(v64):
    """NoScramble must return the bare Sobol' value, identical at every pixel
    and independent of the seed."""
    a = SobolSampler(v64, scrambler=NoScramble(), seed=0)
    b = SobolSampler(v64, scrambler=NoScramble(), seed=999)
    a.setup_path(0, 0, 5)
    b.setup_path(10, 20, 5)        # different pixel + seed
    assert a.next_2d() == b.next_2d()


def test_pixels_are_decorrelated(v64):
    """Owen scramble must give neighboring pixels independent sequences."""
    a = SobolSampler(v64, scrambler=OwenScramble(), seed=0)
    b = SobolSampler(v64, scrambler=OwenScramble(), seed=0)
    a.setup_path(0, 0, 3)
    b.setup_path(1, 0, 3)
    assert a.next_2d() != b.next_2d()


def test_seed_changes_realization(v64):
    a = SobolSampler(v64, scrambler=OwenScramble(), seed=0)
    b = SobolSampler(v64, scrambler=OwenScramble(), seed=1)
    a.setup_path(0, 0, 3)
    b.setup_path(0, 0, 3)
    assert a.next_2d() != b.next_2d()


def test_reproducible(v64):
    """Same (pixel, sample, seed) must reproduce exactly — needed for debugging."""
    def draw():
        s = SobolSampler(v64, scrambler=OwenScramble(), seed=42)
        s.setup_path(7, 9, 11)
        return (s.next_1d(), s.next_2d())
    assert draw() == draw()


def test_dimension_advances(v64):
    """next_1d / next_2d must consume distinct dimensions (no accidental reuse)."""
    s = SobolSampler(v64, scrambler=OwenScramble(), seed=0)
    s.setup_path(0, 0, 12345)
    a = s.next_1d()
    b = s.next_1d()
    assert a != b


def test_scramble_mean_near_half(v64):
    """A measure-preserving scramble keeps the sample mean near 0.5."""
    pts = _collect(PaddedSobolSampler(v64, spp=N, scrambler=OwenScramble(), seed=3))
    assert abs(pts.mean() - 0.5) < 0.02
