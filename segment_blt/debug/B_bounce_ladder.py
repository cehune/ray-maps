"""
B. Bounce ladder (+ optional cluster_c sweep).

Sweep num_prop_iterations ∈ {0, 1, 2, 4, 8, 16}. Tells you whether
propagation saturates to a sensible plateau:

  • Monotonic rise → plateau at ~gain/(1-gain) × the 1-iter value:
        propagation works, you were just truncated.
  • Plateaus far too low → transport kernel sub-stochastic (over-divided
        MMIS, or too-aggressive geom clamp, or too few pairs per receiver).
  • Doesn't move past N=1 → propagation isn't wiring rad_in[j] → rad_in[i].
  • N=0 is direct-only — see script C for the focused version of that.

CLI:
  python B_bounce_ladder.py                  # default c=10
  python B_bounce_ladder.py --c 10 30 60     # sweep cluster_c on top of N

Look for "zero-incoming" and "camera-first 0" in the diag output — that's
the new instrumentation. If camera-first 0% is high, lots of pixels get
zero indirect and the image will have black holes.
"""
from _setup import cornell_scene

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import mitsuba as mi
from segments.renderer import Renderer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c", nargs="+", type=int, default=[10],
                    help="cluster_c values to sweep (default: 10)")
    ap.add_argument("--width", type=int, default=128)
    args = ap.parse_args()

    W = H = args.width
    scene = cornell_scene(W, H)
    renderer = Renderer()

    ladder = [0, 1, 2, 4, 8, 16]
    results = {}  # (c, N) → (img, mean, p99)

    for c in args.c:
        for n in ladder:
            print(f"\n── cluster_c={c}  num_prop_iterations={n} ──")
            img = renderer.render_vec(
                scene, height=H, width=W, n_iterations=1,
                num_prop_iterations=n,
                cluster_c=c,
                verbose=False, beta=0.1, add_light_samples=False,
            )
            lum = img.sum(-1)
            m = float(lum.mean())
            p = float(np.percentile(lum, 99))
            results[(c, n)] = (img, m, p)
            print(f"   mean lum = {m:.4e}   p99 lum = {p:.4e}")

    # for single-c case keep the rest of the script identical
    c0 = args.c[0]
    images = [results[(c0, n)][0] for n in ladder]
    means  = [results[(c0, n)][1] for n in ladder]
    p99s   = [results[(c0, n)][2] for n in ladder]

    print("\n── summary ─────────────────────────────")
    print(f"{'N':>4} {'mean_lum':>12} {'p99_lum':>12} {'mean/N=1':>10}")
    base = means[1] if means[1] > 0 else 1e-12
    for n, m, p in zip(ladder, means, p99s):
        print(f"{n:>4} {m:>12.4e} {p:>12.4e} {m/base:>10.3f}")

    geo_predicted = []
    if means[1] > 0 and means[2] > 0:
        # estimate gain from N=1 → N=2 ratio: total_N = sum_{k=0..N-1} gain^k * direct
        # so means[2]/means[1] ≈ (1+g)/1 → g ≈ ratio-1
        g = max(0.0, means[2] / means[1] - 1.0)
        print(f"  estimated effective gain g ≈ {g:.3f}")
        if g < 1.0:
            plateau = means[1] * (1 / (1 - g)) if g < 0.999 else float("inf")
            print(f"  predicted plateau (mean lum)   ≈ {plateau:.4e}")

    # tile the images
    n_cols = len(ladder)
    fig, axes = plt.subplots(2, n_cols, figsize=(3 * n_cols, 6),
                             gridspec_kw={'height_ratios': [3, 2]})
    for i, (n, img) in enumerate(zip(ladder, images)):
        axes[0, i].imshow(np.clip(img ** (1/2.2), 0, 1))
        axes[0, i].set_title(f"N={n}\nmean={means[i]:.2e}")
        axes[0, i].axis('off')

    ax = axes[1, 0]
    ax.plot(ladder, means, 'o-', label='mean')
    ax.plot(ladder, p99s, 's--', label='p99')
    ax.set_xlabel("num_prop_iterations")
    ax.set_ylabel("luminance")
    ax.set_yscale('log')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()
    for j in range(1, n_cols):
        axes[1, j].axis('off')

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "B_ladder.png")
    plt.savefig(out, dpi=100); plt.close()
    print(f"\nsaved: {out}")

    # if multiple c values were swept, dump a comparison table
    if len(args.c) > 1:
        print("\n── cluster_c sweep summary ─────────────────────────────")
        print(f"  {'c':>4} " + " ".join(f"N={n:>3}" for n in ladder))
        for c in args.c:
            row = [f"{results[(c, n)][1]:.3e}" for n in ladder]
            print(f"  {c:>4} " + " ".join(f"{v:>7}" for v in row))


if __name__ == "__main__":
    main()
