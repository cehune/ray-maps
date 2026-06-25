"""
J. Sweep Knaus-Zwicker α over {0.2, 0.4, 2/3} and chart whether slower
   decay preserves coverage well enough for the paper's "smaller kernel
   = less bias" trend to actually appear.

Theory recap (Wang/West/Hachisuka 2025 §6.2):
  • Progressive kernel shrinkage with α ∈ (0, 1) reduces bias as iters
    grow — IF per-kernel sample count stays adequate.
  • α=2/3 is the Knaus-Zwicker default (paper's choice). Aggressive.
  • Smaller α → slower shrinkage → coverage stays healthier longer.
  • α=0 → no shrinkage at all (matches --no-progressive).

If our codebase is sample-density-limited (which the previous mean_ratio
descent suggested), a smaller α should make mean_ratio stop drifting
downward and ideally start climbing toward 1.0 as iters accumulate.

Usage:
    python J_alpha_sweep.py --max-iter 16
    python J_alpha_sweep.py --max-iter 32 --alphas 0.2 0.4 0.67
    python J_alpha_sweep.py --max-iter 32 --width 100   # smaller for speed
"""
from _setup import cornell_scene

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import mitsuba as mi
from segments.renderer import Renderer, _make_blt_components


DEFAULT_REF = "/Users/celine/Documents/projects/ray-maps/baseline/spp_renders-cbox-128x128/reference.exr"


def load_ref(path, W, H):
    arr = np.array(mi.Bitmap(path))[..., :3].astype(np.float64)
    if arr.shape[:2] != (H, W):
        raise SystemExit(f"ref shape {arr.shape[:2]} ≠ {(H, W)}")
    return arr


def reinhard_tonemap(img, exposure, gamma=2.2):
    scaled = np.asarray(img, dtype=np.float64) * exposure
    compressed = scaled / (1.0 + scaled)
    return np.clip(compressed ** (1.0 / gamma), 0.0, 1.0).astype(np.float32)


def run_one(scene, sensor, renderer, H, W, max_iter, c, N, alpha, ref,
            geom_clamp=200.0, geom_mode="hard"):
    """Run BLT for one alpha value, return per-iter (rmse, mean_ratio, voxel_size)."""
    cluster, mmis, propagation, samplers = _make_blt_components(
        scene, add_light_samples=False,
        num_prop_iterations=N, cluster_c=c, alpha=alpha,
        geom_clamp_factor=geom_clamp, geom_clamp_mode=geom_mode,
    )
    accum = np.zeros((H, W, 3), dtype=np.float64)
    rmses, mrs, voxel_sizes = [], [], []
    last_img = None
    for i in range(max_iter):
        cluster._iteration = i   # progressive on
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
        voxel_sizes.append(float(cluster.voxel_size))
        last_img = running
    return rmses, mrs, voxel_sizes, last_img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=DEFAULT_REF)
    ap.add_argument("--max-iter", type=int, default=16)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--c", type=int, default=30)
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.2, 0.4, 0.67],
                    help="Knaus-Zwicker α values to sweep. Smaller = slower shrinkage.")
    ap.add_argument("--geom-clamp", type=float, default=200.0)
    args = ap.parse_args()

    W = H = args.width
    ref = load_ref(args.ref, W, H)
    ref_mean = float(ref.sum(-1).mean())
    print(f"ref shape={ref.shape}  mean={ref_mean:.4e}")

    scene = cornell_scene(W, H)
    sensor = scene.sensors()[0]
    renderer = Renderer()

    results = {}
    for alpha in args.alphas:
        print(f"\n── α = {alpha} ──")
        rmses, mrs, vs, img = run_one(
            scene, sensor, renderer, H, W,
            args.max_iter, args.c, args.N, alpha, ref,
            geom_clamp=args.geom_clamp,
        )
        results[alpha] = (rmses, mrs, vs, img)
        print(f"   iter1  voxel={vs[0]:.3e}   mr={mrs[0]:.4f}   RMSE={rmses[0]:.4e}")
        print(f"   final  voxel={vs[-1]:.3e}   mr={mrs[-1]:.4f}   RMSE={rmses[-1]:.4e}")
        print(f"   voxel shrink ratio = {vs[-1]/vs[0]:.3f}")

    # ── summary table ────────────────────────────────────────────
    print("\n── summary @ final iter ─────────────────────────────")
    print(f"  {'α':>6s} {'voxel_final':>12s} {'shrink':>8s} {'RMSE':>11s} {'mean_ratio':>11s}")
    for alpha, (rmses, mrs, vs, _) in results.items():
        print(f"  {alpha:>6.3f} {vs[-1]:>12.3e} {vs[-1]/vs[0]:>8.3f} "
              f"{rmses[-1]:>11.3e} {mrs[-1]:>11.4f}")

    # ── plots ────────────────────────────────────────────────────
    iters = np.arange(1, args.max_iter + 1)
    n_alpha = len(args.alphas)
    fig, axes = plt.subplots(2, max(n_alpha, 2), figsize=(4 * max(n_alpha, 2), 8))
    if n_alpha == 1:
        axes = axes.reshape(2, -1)

    # row 0: convergence curves (one big plot, all alphas)
    ax = axes[0, 0]
    for alpha, (rmses, mrs, vs, _) in results.items():
        ax.loglog(iters, rmses, "o-", label=f"α={alpha}")
    ax.set_xlabel("iterations"); ax.set_ylabel("RMSE")
    ax.set_title("RMSE vs iter (lower better)")
    ax.grid(True, which="both", alpha=0.3); ax.legend()

    ax = axes[0, 1] if n_alpha > 1 else axes[0, 0]
    for alpha, (rmses, mrs, vs, _) in results.items():
        ax.plot(iters, mrs, "o-", label=f"α={alpha}")
    ax.axhline(1.0, color="k", linestyle="--", alpha=0.5)
    ax.set_xlabel("iterations"); ax.set_ylabel("mean(BLT)/mean(ref)")
    ax.set_title("mean ratio — should ASCEND to 1.0 if α works")
    ax.set_ylim(0.7, 1.1); ax.grid(True, alpha=0.3); ax.legend()

    # third axis on top row if available — voxel size vs iter (sanity)
    if n_alpha > 2:
        ax = axes[0, 2]
        for alpha, (rmses, mrs, vs, _) in results.items():
            ax.semilogy(iters, vs, "o-", label=f"α={alpha}")
        ax.set_xlabel("iterations"); ax.set_ylabel("voxel_size")
        ax.set_title("kernel shrinkage rate"); ax.grid(True, which="both", alpha=0.3); ax.legend()

    # row 1: tonemapped final image per alpha
    ref_lum = ref.sum(-1)
    med = float(np.median(ref_lum[ref_lum > 0])) if (ref_lum > 0).any() else 0.0
    exposure = (0.18 / med) if med > 0 else 1.0

    for j, (alpha, (rmses, mrs, vs, img)) in enumerate(results.items()):
        ax = axes[1, j] if n_alpha > 1 else axes[1, 0]
        ax.imshow(reinhard_tonemap(img, exposure))
        ax.set_title(f"α={alpha}  mr={mrs[-1]:.3f}"); ax.axis("off")
    # blank any extra panels
    for j in range(len(results), axes.shape[1]):
        axes[1, j].axis("off")

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "J_alpha_sweep.png")
    plt.savefig(out, dpi=110); plt.close()
    print(f"\nsaved: {out}")

    # ── interpretation ───────────────────────────────────────────
    print("\n── interpretation ──")
    print("  • If smaller α has mean_ratio CLOSER to 1.0 with comparable RMSE,")
    print("    coverage was the bottleneck — slow shrinkage is the right choice")
    print("    for your sample budget.")
    print("  • If all α track each other, the K-cancellation is already working")
    print("    and progressive simply isn't necessary at this resolution.")
    print("  • If mean_ratio descends for ALL α, the bias is sample-density-")
    print("    independent and you need more iters or to raise --c.")


if __name__ == "__main__":
    main()
