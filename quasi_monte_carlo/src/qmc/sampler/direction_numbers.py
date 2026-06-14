import numpy as np
import os

_TABLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "joe_kuo_new_100.txt")

def directions(max_dim: int, filepath: str = _TABLE_PATH) -> np.ndarray:
    """(max_dim, 32) Sobol' direction-number table, built for exactly the
    requested number of dimensions. This is the per-dimension computation;
    callers only ever specify max_dim."""
    return load_direction_numbers(filepath, max_dim=max_dim)

def load_direction_numbers(filepath: str, max_dim: int = 64) -> np.ndarray:
    """
    Load Joe-Kuo direction numbers from file.
    expect to use ./joe_kuo_new_100.txt or anything from 
    https://web.maths.unsw.edu.au/~fkuo/sobol/

    It just needs to be in their format.

    Returns:
        v: np.ndarray of shape (max_dim, 32), dtype=uint32
           v[d][k] is the k-th direction number for dimension d,
           stored as a 32-bit integer (fixed point: value = v / 2^32)

    The table file has a header line, then one row per dimension. Joe-Kuo's
    numbering is 1-based and dimension 1 (van der Corput) is implicit, so the
    first tabulated row is d=2. Columns are:
        d  s  a  m_1  m_2 ... m_s
    where d is the dimension index, s is the degree of the primitive
    polynomial, a encodes its coefficients, and m_1..m_s are the initial
    direction numbers. File row d is stored at array index v[d-1]
    (so file d=2 -> v[1]).
    """
    M = 32  # bit precision

    # v[d][k] = direction number k for dimension d
    # d=0 is Van der Corput, handled by the generator directly
    v = np.zeros((max_dim, M), dtype=np.uint32)

    # dim 0: Van der Corput — direction number k is 1 in bit position k
    # i.e. v[0][k] = 2^(31-k) so that bit k of the output is bit k of index
    for k in range(M):
        v[0][k] = 1 << (M - 1 - k)

    # dims 1..max_dim-1: load from table
    with open(filepath, 'r') as f:
        # skip header line
        next(f)

        for d in range(1, max_dim):
            line = f.readline()
            if not line:
                raise ValueError(
                    f"File only has {d} dimensions, requested {max_dim}"
                )

            tokens = line.split()
            # columns: d  s  a  m_1 .. m_s   (d is the dimension index)
            s = int(tokens[1])   # degree of primitive polynomial
            a = int(tokens[2])   # polynomial coefficients (packed)
            m = [int(x) for x in tokens[3:]]  # initial direction numbers

            # initial values: m[k] scaled to 32-bit fixed point
            # m[k] must be odd and < 2^(k+1)
            init = np.zeros(M, dtype=np.uint32)
            for k in range(s):
                init[k] = np.uint32(m[k]) << np.uint32(M - 1 - k)

            # fill remaining direction numbers via recurrence:
            # v[k] = v[k-s] XOR (v[k-s] >> s) XOR
            #        XOR_{j=1}^{s-1} a_j * v[k-j]
            # where a_j is bit j of a (the polynomial coefficients)
            for k in range(s, M):
                vk = init[k - s] ^ (init[k - s] >> np.uint32(s))
                for j in range(1, s):
                    if (a >> (s - 1 - j)) & 1:
                        vk ^= init[k - j]
                init[k] = vk

            v[d] = init

    return v