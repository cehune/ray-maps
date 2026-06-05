"""
H. Sweep geom and/or MMIS clamp factors and chart bias-vs-variance.

For each clamp setting, runs N iterations of BLT averaging and reports
the final (running) RMSE and mean_ratio vs reference. The sweet spot is
the loosest clamp where mean_ratio stays at ~1.0 and RMSE is minimized.

Bias/variance picture:
  • Tight clamp (e.g. 10×)  → low variance, high (negative) bias.
        mean_ratio < 1.0, RMSE floors out at the bias level.
  • Loose clamp (e.g. 200×) → near-unbiased, high variance.
        mean_ratio ≈ 1.0, RMSE dominated by firefly tail.
  • No clamp (None)         → unbiased in expectation but slow convergence.
        mean_ratio bounces above 1.0 early, drifts down as fireflies
        average out.

Usage:
    # default: geom-clamp sweep at fixed N iters
    python H_clamp_sweep.py --max-iter 16

    # custom factors:
    python H_clamp_sweep.py --geom-factors 10 50 100 200 none

    # also sweep mmis (does cross product; expensive):
    python H_clamp_sweep.py --geom-factors 50 none --mmis-factors 20 100 none
"""
from _setup import cornell_scene

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import mitsuba as mi
from segments.renderer import Renderer, _make_blt_components


DEFAULT_REF = "/Users/celine/Documents/projects/ray-maps/baseline/spp_renders-200x200-d-1/reference.exr"


def parse_factor_list(items):
    """Parse "10 50 none" → [10.0, 50.0, None]."""
    out = []
    for s in items:
        if str(s).lower() in ("none", "off", "null"):
            out.append(None)
        else:
            out.append(float(s))
    return out


def load_ref(path, W, H):
    arr = np.array(mi.Bitmap(path))[..., :3].astype(np.float64)
    if arr.shape[:2] != (H, W):
        raise SystemExit(f"ref shape {arr.shape[:2]} ≠ {(H, W)}")
    return arr


def run_one(scene, sensor, renderer, H, W, max_iter, c, N, geom_f, mmis_f, progressive=True):
    """Returns (per_iter_rmse, per_iter_mean_ratio) lists."""
    cluster, mmis, propagation, samplers = _make_blt_components(
        scene, add_light_samples=False,
        num_prop_iterations=N, cluster_c=c,
        geom_clamp_factor=geom_f, mmis_clamp_factor=mmis_f,
    )
    accum = np.zeros((H, W, 3), dtype=np.float64)
    rmses, mrs = [], []
    for i in range(max_iter):
        if progressive:
            cluster._iteration = i
        sampler = mi.load_dict({"type": "independent", "sample_count": 1})
        sampler.seed(i, W * H)
        img = renderer._render_iterate_vec(
            scene, sensor, sampler, H, W,
            cluster, mmis, propagation, samplers,
            iteration=i, verbose=False, beta=0.1, add_light_samples=False,
        )
        accum += img.astype(np.float64)
        running = accum / (i + 1)
        rmses.append(float(np.sqrt(((running - ref) ** 2).mean())))
        mrs.append(float(running.sum(-1).mean() / max(ref.sum(-1).mean(), 1e-12)))
    return rmses, mrs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=DEFAULT_REF)
    ap.add_argument("--max-iter", type=int, default=8)
    ap.add_argument("--width", type=int, default=128, help="lower for speed when sweeping")
    ap.add_argument("--c", type=int, default=30)
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--geom-factors", nargs="+", default=["20", "50", "100", "none"],
                    help="space-separated list. 'none' = clamp off.")
    ap.add_argument("--mmis-factors", nargs="+", default=["none"],
                    help="space-separated list, or just 'none' to skip the mmis axis.")
    args = ap.parse_args()

    geom_list = parse_factor_list(args.geom_factors)
    mmis_list = parse_factor_list(args.mmis_factors)

    W = H = args.width

    global ref
    ref = load_ref(args.ref, W, H)
    print(f"ref shape={ref.shape}  mean={ref.sum(-1).mean():.4e}")

    scene = cornell_scene(W, H)
    sensor = scene.sensors()[0]
    renderer = Renderer()

    results = {}  # (gf, mf) -> (rmses, mrs)
    for gf in geom_list:
        for mf in mmis_list:
            label = f"geom={gf}, mmis={mf}"
            print(f"\n── {label} ──")
            rmses, mrs = run_one(scene, sensor, renderer, H, W,
                                 args.max_iter, args.c, args.N, gf, mf)
            results[(gf, mf)] = (rmses, mrs)
            print(f"   final RMSE = {rmses[-1]:.4e}   final mean_ratio = {mrs[-1]:.4f}")

    # ── summary table ─────────────────────────────────────────────
    print("\n── summary @ final iter ─────────────────────────────")
    print(f"  {'geom':>6s} {'mmis':>6s} {'RMSE':>11s} {'mean_ratio':>11s}")
    for (gf, mf), (rmses, mrs) in results.items():
        print(f"  {str(gf):>6s} {str(mf):>6s} {rmses[-1]:>11.3e} {mrs[-1]:>11.4f}")

    # ── plot ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    iters = np.arange(1, args.max_iter + 1)
    for (gf, mf), (rmses, mrs) in results.items():
        label = f"g={gf} m={mf}"
        axes[0].loglog(iters, rmses, "o-", label=label)
        axes[1].plot(iters, mrs, "o-", label=label)
    axes[0].set_xlabel("iterations"); axes[0].set_ylabel("RMSE")
    axes[0].set_title("RMSE vs iter per clamp setting"); axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].axhline(1.0, color="k", linestyle="--", alpha=0.5)
    axes[1].set_xlabel("iterations"); axes[1].set_ylabel("mean(BLT)/mean(ref)")
    axes[1].set_title("mean ratio — sweet spot is asymptote=1.0"); axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    axes[1].set_ylim(0.5, 1.5)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "H_clamp_sweep.png")
    plt.savefig(out, dpi=110); plt.close()
    print(f"\nsaved: {out}")
    print("\nPick the (geom, mmis) row whose mean_ratio is closest to 1.0 with lowest RMSE,")
    print("then rerun F with those values:")
    print("    F renders directly with whatever render_vec defaults are; if you want F to")
    print("    apply the chosen clamps, pass them through your main.py or edit F to add the")
    print("    same params to its render_vec call.")


if __name__ == "__main__":
    main()
