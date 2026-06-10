B = 8

C = [[1 if i == j else 0 for j in range(B)] for i in range(B)]

def int_to_bits(n, B):
    return [(n >> j) & 1 for j in range(B)]

def matvec_f2(C, n_bits, B):
    y = [0] * B
    for i in range(B):
        for j in range(B):
            y[i] ^= C[i][j] & n_bits[j]
    return y

def bits_to_fraction(y):
    x = 0.0
    for i, b in enumerate(y):
        if b:
            x += 2.0 ** -(i + 1)   # row i (0-indexed) is the 2^{-(i+1)} place
    return x

N = 8
for n in range(N):
    n_bits = int_to_bits(n, B)
    y      = matvec_f2(C, n_bits, B)
    x      = bits_to_fraction(y)
    print(f"n={n}  n_bits={n_bits}  y={y}  x={x}")
    