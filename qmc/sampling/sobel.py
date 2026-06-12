import numpy as np
def sobol_2d(n, w=8):
    v0, v1 = build_direction_numbers(w)
    scale = 1.0 / (1 << w) # 1 / (2^8) or 1 / 256 for w = 8
    points = np.zeroes((n, 2))

    for i in range(n):
        x, y = 0, 0
        # really is just matrix multiplication
        for k in range(w):
            if (i >> k) & i # shit











def sobol_2d_gray(n, w=8):
    V0, V1 = build_direction_numbers(w)
    scale = 1.0 / (1 << w)

    points = np.zeros((n, 2))
    for i in range(n):
        x, y = 0, 0
        for k in range(w):
            if (i >> k) & 1:   # bits of i directly, no gray code
                x ^= V0[k]
                y ^= V1[k]
        points[i] = [x * scale, y * scale]

    return points

def sobol_2d(n, w=8):
    V0, V1 = build_direction_numbers(w)
    scale = 1.0 / (1 << w)

    points = np.zeros((n, 2))
    for i in range(n):
        g = i ^ (i >> 1)  # Gray code of i
        x, y = 0, 0
        for k in range(w):
            if (g >> k) & 1:
                x ^= V0[k]
                y ^= V1[k]
        points[i] = [x * scale, y * scale]

    return points