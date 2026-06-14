import matplotlib.pyplot as plt
import numpy as np

def build_C1(B):
    """Dim 1: C_1 = I_B.  Gives van der Corput (bit-reversal in base 2)."""
    return np.eye(B, dtype=np.uint8)


def build_C2(B):
    """Dim 2: Sobol's degree-1 primitive polynomial p(x) = x + 1, m_1 = 1.
       Antonov-Saleev recurrence reduces to  m_i = (2 * m_{i-1}) XOR m_{i-1}.
       Column i of C_2 is the bit pattern of m_i, MSB at row 0."""
    m = [1]
    for _ in range(B - 1):
        # ^ is XOR: add the two integers mod 2 bit-by-bit, no carries.
        m.append((2 * m[-1]) ^ m[-1])

    C = np.zeros((B, B), dtype=np.uint8)
    for col in range(B):
        width = col + 1            # m_{col+1} fits in col+1 bits
        val   = m[col]
        for row in range(width):
            # (val >> k) brings bit k of val down to position 0;
            # & 1 then masks everything except that lowest bit.
            C[row, col] = (val >> (width - 1 - row)) & 1
    return C


def sobol_2d(m_pow, B=32):
    """Generate 2^m_pow Sobol points in 2D via direct matmul mod 2."""
    N  = 1 << m_pow                # 1 << k is just 2**k, written bitwise
    C1 = build_C1(B)
    C2 = build_C2(B)

    n_all = np.arange()

    # Build a (N, B) matrix where row n is the LSB-first bit vector of n.
    n_all  = np.arange(N, dtype=np.uint64)
    shifts = np.arange(B, dtype=np.uint64)
    # n_all[:, None] >> shifts  shifts each n right by 0, 1, ..., B-1.
    # & 1 then picks the lowest bit of each shifted value, i.e. bit j of n.
    bits = ((n_all[:, None] >> shifts) & 1).astype(np.uint8)

    # Matrix-vector product over F_2, done in two steps:
    #   (1) integer matmul produces sums of 0s and 1s,
    #   (2) % 2 reduces those sums mod 2 (equivalent to XOR-ing the ANDed bits).
    Y1 = (bits @ C1.T) % 2
    Y2 = (bits @ C2.T) % 2

    # Read each output bit vector as a binary fraction in [0, 1):
    # row i (0-indexed) carries place value 2^{-(i+1)}.
    place = 2.0 ** -np.arange(1, B + 1)
    X = Y1 @ place
    Y = Y2 @ place
    return X, Y


def plot_sobol(m_pow, B=32, save_path=None):
    X, Y = sobol_2d(m_pow, B)
    N = len(X)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(X, Y, s=10, c=np.arange(N), cmap='viridis')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_title(f"Sobol (dim 1, dim 2),  N = 2^{m_pow} = {N}")
    ax.set_xlabel("x  (from $C_1$)")
    ax.set_ylabel("y  (from $C_2$)")
    ax.grid(True, alpha=0.3)
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches='tight')
    return fig


if __name__ == "__main__":
    # Quick sanity check: first 8 points should be
    #   x: 0, 1/2, 1/4, 3/4, 1/8, 5/8, 3/8, 7/8
    #   y: 0, 1/2, 3/4, 1/4, 5/8, 1/8, 3/8, 7/8
    X, Y = sobol_2d(3, B=8)
    print("n  x        y")
    for n, (x, y) in enumerate(zip(X, Y)):
        print(f"{n}  {x:.4f}   {y:.4f}")

    # Build a panel of plots for m = 4, 6, 8, 10
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for ax, m_pow in zip(axes.flat, [4, 6, 8, 10]):
        X, Y = sobol_2d(m_pow)
        N = len(X)
        ax.scatter(X, Y, s=6, c=np.arange(N), cmap='viridis')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_title(f"N = 2^{m_pow} = {N}")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Sobol 2D point sets at increasing N", y=1.00)
    fig.tight_layout()
    fig.savefig("/Users/celine/Documents/projects/ray-maps/qmc/sampling/photos/2d.png", dpi=120, bbox_inches='tight')
    print("Saved photo for sobol")