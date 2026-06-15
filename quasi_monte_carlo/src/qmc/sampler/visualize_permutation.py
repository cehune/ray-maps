"""Visualize Sobol' scrambling: same low discrepancy, independent randomness.

Generates the first 2^m two-dimensional Sobol' points and renders several
panels:

  * the raw (unscrambled) sequence — the deterministic base point set,
  * a few independently *scrambled* runs (each a different random "permutation"
    of the sequence, selected by the global seed),
  * an i.i.d. uniform-random panel for contrast.

The scrambled panels look random relative to one another (and to the base),
yet every one keeps exactly one point per dyadic cell — that's the
low-discrepancy (t,m,s)-net property surviving the randomization. The uniform
panel, by contrast, clumps and leaves gaps.

Run:
    python sampler/visualize_permutation.py            # defaults: m=6, owen, 4 runs
    python sampler/visualize_permutation.py --m 8 --runs 6 --scramble owen --show
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from qmc.sampler.direction_numbers import load_direction_numbers
from qmc.sampler.sobol_sampler import SobolSampler
from qmc.sampler.constants import SCRAMBLERS, SAMPLERS
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TABLE = os.path.join(HERE, 'joe_kuo_new_100.txt')


def collect_2d(sampler, n):
    """First n two-dimensional samples at a fixed pixel."""
    pts = np.empty((n, 2))
    for i in range(n):
        sampler.setup_path(0, 0, i)
        pts[i] = sampler.next_2d()
    return pts


def cells_off_net(pts, g):
    """How many cells of a g x g grid do NOT hold exactly one point.
    Zero for a (0,m,2)-net; positive when points clump (e.g. i.i.d. random)."""
    grid = np.zeros((g, g), dtype=int)
    xs = np.minimum((pts[:, 0] * g).astype(int), g - 1)
    ys = np.minimum((pts[:, 1] * g).astype(int), g - 1)
    np.add.at(grid, (xs, ys), 1)
    return int(np.count_nonzero(grid != 1))


def _draw_panel(ax, pts, g, title, color):
    # dyadic grid: for a Sobol' net every cell holds exactly one point
    for k in range(1, g):
        ax.axhline(k / g, color='0.85', lw=0.6, zorder=0)
        ax.axvline(k / g, color='0.85', lw=0.6, zorder=0)
    ax.scatter(pts[:, 0], pts[:, 1], s=14, color=color, zorder=2)
    off = cells_off_net(pts, g)
    ax.set_title(f"{title}\n{off} cells off-net", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')


def make_figure(v, m=6, scramble='owen', runs=4, base_seed=0):
    n = 1 << m
    g = 1 << (m // 2) if m % 2 == 0 else 1 << ((m + 1) // 2)  # sqrt-ish grid
    scr_cls = SCRAMBLERS[scramble]

    panels = []  # (title, points, color)

    # base, unscrambled sequence
    base = collect_2d(SobolSampler(v, scrambler=NoScramble()), n)
    panels.append((f"raw Sobol' (base)", base, '0.25'))

    # independent scrambled runs — each a different random permutation
    cmap = plt.get_cmap('viridis')
    for r in range(runs):
        seed = base_seed + r
        s = SobolSampler(v, scrambler=scr_cls(), seed=seed)
        panels.append((f"{scramble} scramble · seed {seed}",
                       collect_2d(s, n), cmap(0.15 + 0.7 * r / max(runs - 1, 1))))

    # i.i.d. uniform random, for contrast
    rng = np.random.default_rng(base_seed)
    panels.append(("just some random nums", rng.random((n, 2)), '#c0392b'))

    ncols = min(4, len(panels))
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 3.5 * nrows),
                             layout='constrained')
    axes = np.atleast_1d(axes).ravel()

    for ax, (title, pts, color) in zip(axes, panels):
        _draw_panel(ax, pts, g, title, color)
    for ax in axes[len(panels):]:
        ax.axis('off')

    fig.suptitle(
        f"Sobol' scrambling — {n} points (m={m}) {g} sized grid\n",
        fontsize=11)
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--m', type=int, default=6, help="number of points = 2^m")
    p.add_argument('--sampler', choices=list(SAMPLERS), default='sobol')
    p.add_argument('--scramble', choices=list(SCRAMBLERS), default='owen')
    p.add_argument('--runs', type=int, default=4, help="number of scrambled realizations")
    p.add_argument('--seed', type=int, default=0, help="base global seed")
    p.add_argument('--table', default=DEFAULT_TABLE)
    p.add_argument('--save', default=os.path.join(HERE, 'photos', 'permutations'))
    p.add_argument('--show', action='store_true')
    args = p.parse_args()

    v = load_direction_numbers(args.table, max_dim=2)
    fig = make_figure(v, m=args.m, scramble=args.scramble,
                      runs=args.runs, base_seed=args.seed)
    base = args.save + f"_m{args.m}_runs{args.runs}_seed{args.seed}_sampler{args.sampler}_scramble{args.scramble}.png"
    if args.save:
        os.makedirs(os.path.dirname(args.save), exist_ok=True)
        fig.savefig(base, dpi=140)
        print(f"Saved {base}")
    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
