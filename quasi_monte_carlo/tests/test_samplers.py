"""SobolSampler / PaddedSobolSampler / IndependentSampler correctness.

Strongest check is the (0, m, 2)-net property: the first 2^m two-dimensional
points must land exactly one per cell of every 2^a x 2^(m-a) dyadic grid.
Owen and random-digit scrambling both preserve this; a broken generator or
scramble shifts points off the dyadic lattice and the test fails.

Samplers are addressed by decision (cam_2d / bounce_2d / bounce_1d) and hold no
per-draw state, so the remaining tests pin down that mapping and its statelessness.
"""

import numpy as np
import pytest

from qmc.sampler.sobol_sampler import SobolSampler
from qmc.sampler.padded_sobol_sampler import PaddedSobolSampler
from qmc.sampler.independent_sampler import IndependentSampler
from qmc.sampler.scramble import NoScramble, RandomDigitScramble, OwenScramble
from qmc.sampler.sampler import CAMERA_DIMS, BOUNCE_STRIDE, CAM, SLOT

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
    """One 2D camera sample (dims 0,1) per sample index."""
    pts = np.empty((N, 2))
    for i in range(N):
        sampler.setup_path(3, 4, i)
        pts[i] = sampler.cam_2d("jitter")
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
    assert a.cam_2d("jitter") == b.cam_2d("jitter")


def test_pixels_are_decorrelated(v64):
    """Owen scramble must give neighboring pixels independent sequences."""
    a = SobolSampler(v64, scrambler=OwenScramble(), seed=0)
    b = SobolSampler(v64, scrambler=OwenScramble(), seed=0)
    a.setup_path(0, 0, 3)
    b.setup_path(1, 0, 3)
    assert a.cam_2d("jitter") != b.cam_2d("jitter")


def test_seed_changes_realization(v64):
    a = SobolSampler(v64, scrambler=OwenScramble(), seed=0)
    b = SobolSampler(v64, scrambler=OwenScramble(), seed=1)
    a.setup_path(0, 0, 3)
    b.setup_path(0, 0, 3)
    assert a.cam_2d("jitter") != b.cam_2d("jitter")


def test_reproducible(v64):
    """Same (pixel, sample, seed, decision) must reproduce exactly."""
    def draw():
        s = SobolSampler(v64, scrambler=OwenScramble(), seed=42)
        s.setup_path(7, 9, 11)
        return (s.bounce_1d(0, "lobe"), s.bounce_2d(0, "dir"))
    assert draw() == draw()


# ── addressable mapping & statelessness ────────────────────────────────────────

def test_dimension_mapping(v64):
    """Every decision resolves to its fixed logical dimension; a 2D decision
    uses two consecutive Sobol' dims."""
    s = SobolSampler(v64, scrambler=OwenScramble(), seed=3)
    s.setup_path(2, 5, 9)
    d = CAMERA_DIMS + 2 * BOUNCE_STRIDE + SLOT["dir"]
    assert s.bounce_2d(2, "dir") == (s._value(d), s._value(d + 1))
    assert s.bounce_1d(2, "rr") == s._value(CAMERA_DIMS + 2 * BOUNCE_STRIDE + SLOT["rr"])
    assert s.cam_2d("lens") == (s._value(CAM["lens"]), s._value(CAM["lens"] + 1))


def test_distinct_slots_distinct_values(v64):
    """Different decisions map to different dimensions (no accidental reuse)."""
    s = SobolSampler(v64, scrambler=OwenScramble(), seed=0)
    s.setup_path(0, 0, 7)
    vals = [s.bounce_1d(0, "lobe"), s.bounce_1d(0, "rr"),
            s.bounce_2d(0, "nee")[0], s.bounce_2d(0, "dir")[0]]
    assert len(set(vals)) == len(vals)


def test_alignment_is_structural(v64):
    """A decision's value is independent of which other decisions were drawn,
    or in what order — there is no cursor to misalign. 'a' walks a full diffuse
    path; 'b' jumps straight to bounce 3's direction; they must agree."""
    a = SobolSampler(v64, scrambler=OwenScramble(), seed=1); a.setup_path(4, 4, 2)
    b = SobolSampler(v64, scrambler=OwenScramble(), seed=1); b.setup_path(4, 4, 2)
    a.cam_2d("jitter"); a.cam_2d("lens")
    a.bounce_2d(0, "nee"); a.bounce_1d(0, "lobe"); a.bounce_2d(0, "dir")
    a.bounce_2d(3, "nee"); a.bounce_1d(3, "lobe")
    assert a.bounce_2d(3, "dir") == b.bounce_2d(3, "dir")


def test_independent_is_addressable_and_reproducible():
    """The MC control honours the same interface: in-range, reproducible, and
    distinct decisions give distinct draws."""
    s = IndependentSampler(seed=0)
    s.setup_path(3, 1, 4)
    a = s.bounce_2d(0, "dir")
    assert 0.0 <= a[0] < 1.0 and 0.0 <= a[1] < 1.0
    assert s.bounce_2d(0, "dir") == a                       # reproducible
    assert s.bounce_1d(1, "lobe") != s.bounce_1d(2, "lobe")  # distinct decisions


def test_scramble_mean_near_half(v64):
    """A measure-preserving scramble keeps the sample mean near 0.5."""
    pts = _collect(PaddedSobolSampler(v64, spp=N, scrambler=OwenScramble(), seed=3))
    assert abs(pts.mean() - 0.5) < 0.02
