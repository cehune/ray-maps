"""
Pick a point on a mesh surface the renderer's way: choose a shape proportional
to its area (renderer.py:103-114, cdf=cumsum(areas) + searchsorted), then a
uniform point inside it (segment_gen_sampler.py:308). Big shapes get more dots,
but the dot DENSITY is flat everywhere -- that flatness is p(x) = 1/|M|.

Run:  python validation/visualize_surface_sampling.py
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

N = 400
TRIS = np.array([
    [[0.2, 0.2], [1.2, 0.2], [0.7, 1.0]],   # small
    [[2.0, 0.2], [4.0, 0.2], [3.0, 1.8]],   # medium
    [[0.2, 2.2], [3.8, 2.2], [2.0, 5.2]],   # large
])
COLORS = ["#d1495b", "#edae49", "#00798c"]
rng = np.random.default_rng(0)

# Level 1: pick a triangle proportional to area (the renderer's cdf + searchsorted).
e1, e2 = TRIS[:, 1] - TRIS[:, 0], TRIS[:, 2] - TRIS[:, 0]
areas = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
cdf = np.cumsum(areas)
idx = np.minimum(np.searchsorted(cdf, rng.random(N) * areas.sum()), len(TRIS) - 1)

# Level 2: a uniform point inside the chosen triangle (square-to-triangle warp).
a, b, c = TRIS[idx, 0], TRIS[idx, 1], TRIS[idx, 2]
su1, u2 = np.sqrt(rng.random(N)), rng.random(N)
pts = (1 - su1)[:, None] * a + (su1 * (1 - u2))[:, None] * b + (su1 * u2)[:, None] * c

# Plot.
fig, ax = plt.subplots(figsize=(6.5, 6.5))
for i in range(len(TRIS)):
    ax.add_patch(Polygon(TRIS[i], closed=True, fill=False, edgecolor=COLORS[i], lw=2))
    m = idx == i
    ax.scatter(pts[m, 0], pts[m, 1], s=4, c=COLORS[i], alpha=0.5, linewidths=0)
    ax.text(*TRIS[i].mean(0), str(int(m.sum())), ha="center", va="center", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
ax.set_aspect("equal")
ax.set_title(" N={N}")
fig.tight_layout()
fig.savefig("surface_sampling.png", dpi=130, bbox_inches="tight")
plt.show()
