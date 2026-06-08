"""
G. Hachisuka-style convergence study.

Renders the BLT scene over many iterations, dumping the running-mean
estimate after each one and computing RMSE against a reference EXR.
Produces a log-log plot of error vs iteration count, with theoretical
slope lines (-1/2 for standard MC averaging, -1/3 for progressive
density estimation with α=2/3 per Knaus-Zwicker 2011).

Each iteration shares the cluster/MMIS/propagation objects so the
optional progressive-kernel update has somewhere to count (if enabled).

NOTE on the latent bug in cluster.py: `Cluster._iteration` is never
incremented in the existing pipeline, so kernel shrinkage doesn't happen.
This script optionally bumps it per outer iteration via --progressive,
so you can directly compare both regimes on one plot.

Usage:
    # 32 iterations, no progressive (matches current renderer behavior):
    python G_convergence.py --max-iter 32

    # 32 iters, progressive kernel shrinkage enabled:
    python G_convergence.py --max-iter 32 --progressive

    # Smaller image for speed:
    python G_convergence.py --max-iter 32 --width 128
"""
from _setup import cornell_scene

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import mitsuba as mi
from segments.renderer import Renderer, _make_blt_components


DEFAULT_REF = "/Users/celine/Documents/projects/ray-maps/baseline/spp_renders-128x128-d-1/reference.exr"


def load_ref(path, W, H):
    arr = np.array(mi.Bitmap(path))
    if arr.ndim != 3 or arr.shape[-1] < 3:
        raise SystemExit(f"unexpected ref shape {arr.shape}")
    rgb = arr[..., :3].astype(np.float64)
    if rgb.shape[:2] != (H, W):
        raise SystemExit(
            f"ref is {rgb.shape[:2]}, requested {(H, W)} — "
            f"either re-render the reference at the new resolution, or pass --width/--height matching"
        )
    return rgb


def render_one_iter(renderer, scene, sensor, height, width, cluster,
                    mmis, propagation, samplers, iteration, beta=0.1):
    """Run one BLT iteration; returns the per-iter contribution image."""
    sampler = mi.load_dict({"type": "independent", "sample_count": 1})
    sampler.seed(iteration, width * height)
    return renderer._render_iterate_vec(
        scene, sensor, sampler, height, width,
        cluster, mmis, propagation, samplers,
        iteration=iteration, verbose=False, beta=beta, add_light_samples=False,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=DEFAULT_REF)
    ap.add_argument("--max-iter", type=int, default=32)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--height", type=int, default=128)
    ap.add_argument("--c", type=int, default=30, help="cluster_c")
    ap.add_argument("--N", type=int, default=8, help="num_prop_iterations")
    ap.add_argument("--no-progressive", action="store_true",
                    help="disable Knaus-Zwicker progressive kernel shrinkage "
                         "(by default progressive is ON, matching render_vec's new default)")
    ap.add_argument("--geom-clamp", default="none",
                    help="geom_term cap factor (× median). 'none' = off.")
    ap.add_argument("--geom-clamp-mode", default="hard", choices=("hard", "soft"))
    ap.add_argument("--alpha", type=float, default=0.25,
                    help="Knaus-Zwicker α. Smaller = slower shrinkage = better "
                         "coverage. Paper uses 2/3 but with 5-10× more sample "
                         "density. Default 0.2 matches our budget.")
    args = ap.parse_args()

    W, H = args.width, args.height
    ref = load_ref(args.ref, W, H)
    print(f"reference: {args.ref}   ref mean = {ref.sum(-1).mean():.4e}")

    scene = cornell_scene(W, H)
    sensor = scene.sensors()[0]
    renderer = Renderer()

    geom_f = None if str(args.geom_clamp).lower() in ("none", "off") else float(args.geom_clamp)
    cluster, mmis, propagation, samplers = _make_blt_components(
        scene, add_light_samples=False, num_prop_iterations=args.N, cluster_c=args.c,
        geom_clamp_factor=geom_f, geom_clamp_mode=args.geom_clamp_mode,
        alpha=args.alpha,
    )

    accum = np.zeros((H, W, 3), dtype=np.float64)
    iters = []
    rmses = []
    mean_ratios = []

    progressive = not args.no_progressive
    for i in range(args.max_iter):
        if progressive:
            cluster._iteration = i   # mirror what render_vec now does
        print(f"\n── iter {i+1}/{args.max_iter} "
              f"(c={args.c}, N={args.N}, progressive={progressive}) ──")
        img = render_one_iter(renderer, scene, sensor, H, W,
                              cluster, mmis, propagation, samplers, iteration=i)
        accum += img.astype(np.float64)
        running = accum / (i + 1)

        # linear-space RMSE per channel (matches baseline.rmse)
        rmse = float(np.sqrt(np.mean((running - ref) ** 2)))
        mr = float(running.sum(-1).mean() / max(ref.sum(-1).mean(), 1e-12))
        iters.append(i + 1)
        rmses.append(rmse)
        mean_ratios.append(mr)
        print(f"   running RMSE = {rmse:.4e}   mean_ratio = {mr:.4f}")

    iters = np.array(iters)
    rmses = np.array(rmses)
    mean_ratios = np.array(mean_ratios)

    # log-log slope estimate
    slope, intercept = np.polyfit(np.log(iters), np.log(rmses), 1)
    print(f"\nlog-log slope of RMSE vs iter = {slope:.3f}")
    print(f"  expected ≈ -0.5  (pure MC averaging, no progressive)")
    print(f"  expected ≈ -0.33 (progressive PM, α=2/3, Knaus-Zwicker)")

    # ── plot ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.loglog(iters, rmses, "o-", label=f"BLT (slope={slope:.2f})")
    # reference slopes anchored at iter=1
    ax.loglog(iters, rmses[0] * iters ** (-0.5), "k--", alpha=0.5, label="N^(-1/2) MC")
    ax.loglog(iters, rmses[0] * iters ** (-1/3), "k:",  alpha=0.5, label="N^(-1/3) progressive")
    ax.set_xlabel("iterations")
    ax.set_ylabel("RMSE vs reference")
    ax.set_title(f"convergence  "
                 f"(c={args.c}, N={args.N}, prog={progressive})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.plot(iters, mean_ratios, "o-")
    ax.axhline(1.0, color="k", linestyle="--", alpha=0.5, label="ref")
    ax.set_xlabel("iterations")
    ax.set_ylabel("mean(BLT) / mean(ref)")
    ax.set_title("mean ratio — should asymptote at 1.0 if MMIS is unbiased")
    ax.set_ylim(0.0, 1.5)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    suffix = "prog" if progressive else "noprog"
    out = os.path.join(os.path.dirname(__file__),
                       f"G_convergence_{suffix}_c{args.c}_N{args.N}.png")
    plt.savefig(out, dpi=110); plt.close()
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
