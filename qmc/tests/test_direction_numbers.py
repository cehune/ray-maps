"""Direction-number loading + Sobol' generation correctness."""

import numpy as np
import pytest

from sampler.direction_numbers import load_direction_numbers
from sampler.sobol_core import sobol_uint


def test_shape_and_dtype(v64):
    assert v64.shape == (64, 32)
    assert v64.dtype == np.uint32


def test_dim0_is_van_der_corput(v64):
    # dim 0: direction number k is bit (31-k) set -> radical inverse base 2
    expected = np.array([1 << (31 - k) for k in range(4)], dtype=np.uint32)
    assert np.array_equal(v64[0][:4], expected)


def test_dim1_known_values(v64):
    # First data row (s=1, a=0, m=[1]) -> 2^31, 3*2^30, 5*2^29, ...
    expected = np.array([2147483648, 3221225472, 2684354560, 4026531840],
                        dtype=np.uint32)
    assert np.array_equal(v64[1][:4], expected)


def test_requesting_too_many_dims_raises(dn_path):
    # The table has ~100 dims; ask for more than the file holds.
    with pytest.raises(ValueError):
        load_direction_numbers(dn_path, max_dim=100_000)


def _sobol_points(v, n, dims, gray=False):
    out = np.zeros((n, len(dims)))
    for i in range(n):
        idx = i ^ (i >> 1) if gray else i
        for di, d in enumerate(dims):
            out[i, di] = sobol_uint(idx, d, v) / 2.0 ** 32
    return out


def test_matches_scipy_reference(v64):
    """Our generator (in Gray-code order) must reproduce SciPy's unscrambled
    Sobol' sequence exactly — SciPy uses the same Joe-Kuo direction numbers."""
    qmc = pytest.importorskip("scipy.stats").qmc
    mine = _sobol_points(v64, 64, [0, 1, 2, 3], gray=True)
    ref = qmc.Sobol(d=4, scramble=False).random(64)
    assert np.abs(mine - ref).max() == 0.0
