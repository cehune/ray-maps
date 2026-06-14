"""Shared Sobol' machinery: raw point generation + integer hashing helpers.

These are the pieces both SobolSampler and PaddedSobolSampler build on. They
are deliberately scalar/pure-Python for prototyping and unit-testing; the
hot-path versions for the wavefront integrator can mirror this logic on
Dr.Jit arrays later (every function here is a pure bit/int op, so it ports
directly to elementwise array code).

All math is 32-bit unless noted; Python ints are masked explicitly to avoid
relying on numpy overflow semantics.
"""

MASK32 = 0xFFFFFFFF # max 32 bit hexadecimal
MASK64 = 0xFFFFFFFFFFFFFFFF # max 64 bit one


def sobol_uint(index: int, dim: int, v) -> int:
    """Raw 32-bit Sobol' value for sample `index`, dimension `dim`.

    The sample value in [0, 1) is ``sobol_uint(...) / 2**32``.

    Th s is the true "sobol" result, aka you convert your index for a dimension
    to an output. Not gray code, we're directly getting the result. This is fine
    for now.

    """
    x = 0
    a = int(index)
    b = 0
    row = v[dim]
    """
    basically saying while a is greater than 0
    we check the right most bit, then if its 1 just xor it with the 
    value from the direction number, shift a down. Return the mask result
    which turns it back to hexadecimal
    """
    while a:
        if a & 1:
            x ^= int(row[b])
        a >>= 1
        b += 1
    return x & MASK32


def to_float(u: int) -> float:
    """Map a 32-bit fixed-point value to [0, 1)."""
    return (u & MASK32) * (1.0 / 2.0 ** 32)


# ---------------------------------------------------------------------------
# Hashing / seeding
# ---------------------------------------------------------------------------

def mix_bits(v: int) -> int:
    """
    
    64-bit finalizer, known as avalanching?
    it uses a series of bit shifts (>>), XORs (^=), and multiplications by large prime constants. 
    The goal is to ensure that changing even a single bit of the input v radically changes 
    roughly 50% of the output bits. it destroys any patterns in the input data.

    taken from other code like pbrts mixbit
    """
    v = int(v) & MASK64
    v ^= v >> 31
    v = (v * 0x7FB5D329728EA185) & MASK64
    v ^= v >> 27
    v = (v * 0x81DADEF4BC2DD44D) & MASK64
    v ^= v >> 33
    return v


def hash_ints(*args: int, seed: int = 0) -> int:
    """
    deterministically hash a tuple of integers to a 32-bit value.

    used to derive per-(pixel, dimension, ...) scramble/shuffle seeds. The
    global `seed` lets you generate independent realizations of the whole
    estimator (e.g. for convergence/variance studies across runs).
    """
    h = mix_bits((int(seed) & MASK64) ^ 0x9E3779B97F4A7C15)
    for a in args:
        h = mix_bits(h ^ (int(a) & MASK64))
    return h & MASK32


def permutation_element(i: int, n: int, p: int) -> int:
    """
    return the i-th element of a pseudo-random permutation of [0, n),
    keyed by `p` (Andrew Kensler's hashed permutation).

    used by the padded sampler to shuffle which sample index each padded
    dimension consumes, decorrelating the per-dimension Sobol' sets without
    storing an explicit permutation table.

    just some extra scrambleing 
    """
    if n <= 1:
        return 0
    w = n - 1
    w |= w >> 1
    w |= w >> 2
    w |= w >> 4
    w |= w >> 8
    w |= w >> 16
    i = int(i) & MASK32
    p = int(p) & MASK32
    while True:
        i = (i ^ p) & MASK32
        i = (i * 0xE170893D) & MASK32
        i ^= p >> 16
        i ^= (i & w) >> 4
        i ^= p >> 8
        i = (i * 0x0929EB3F) & MASK32
        i ^= p >> 23
        i ^= (i & w) >> 1
        i = (i * (1 | (p >> 27))) & MASK32
        i = (i * 0x6935FA69) & MASK32
        i ^= (i & w) >> 11
        i = (i * 0x74DCB303) & MASK32
        i ^= (i & w) >> 2
        i = (i * 0x9E501CC3) & MASK32
        i ^= (i & w) >> 2
        i = (i * 0xC860A3DF) & MASK32
        i &= w
        i ^= i >> 5
        if i < n:
            break
    return (i + p) % n
