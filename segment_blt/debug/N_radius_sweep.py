"""
N. Radius sweep with FIXED lookup mechanism (exact ball search).

Holds the neighborhood enumeration constant (KD-tree ball query — no
voxel artifacts) and sweeps the kernel radius from large to small.
Logs RMSE and mean_ratio vs reference for each radius.

This isolates the *kernel-size* effect from the *lookup-mechanism* effect.
M tests "voxel vs ball at same r"; N tests "ball at varying r".
Together they decompose where the bias/variance is actually living.

Expected outcomes:
  • Large r (e.g. 0.5 in 2m Cornell — almost the whole room): high bias
    from kernel-area smearing, low variance from many neighbors.
    mean_ratio should drop notably below the clamp floor.
  • Medium r (0.05–0.1): the sweet spot we've been operating in. Bias
    floor at the clamp's 0.97, variance moderate.
  • Tiny r (0.01–0.02, near paper-faithful): very few neighbors per
    receiver, variance dominates. mean_ratio may stay near clamp floor,
    but RMSE will be much higher.

If RMSE vs r has a clean U shape with the minimum at moderate r, your
algorithm is behaving exactly as a density-estimation method should.
If the minimum is at r→∞ (unbounded smoothing), the kernel is doing
nothing useful and the clamp dominates everything.

Usage:
    # Make sure you have a 50×50 reference:
    python make_reference.py --width 50 --spp 8192

    python N_radius_sweep.py --max-iter 16
    python N_radius_sweep.py --max-iter 16 --radii 0.01 0.02 0.05 0.1 0.2 0.5
"""
from _setup import cornell_scene
from _cache import save_render, load_render
from K_ball_kernel import build_pair_cache_ball

import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import mitsuba as mi

from segments.renderer import Renderer, _make_blt_components, _sample_segments, _final_gather


def load_ref(path, W, H):
    arr = np.array(mi.Bitmap(path))[..., :3].astype(np.float64)
    if arr.shape[:2] != (H, W):
        raise SystemExit(
            f"ref shape {arr.shape[:2]} ≠ {(H, W)}.\n"
            f"Generate: python make_reference.py --width {W} --spp 8192"
        )
    return arr


def render_one_iter_ball(scene, sensor, renderer, H, W,
                         cluster, mmis, propagation, samplers,
                         iteration, radius, beta=0.1):
    sampler = mi.load_dict({"type": "independent", "sample_count": 1})
    sampler.seed(iteration, W * H)
    segment_pool, camera_first_segments = _sample_segments(
        scene, sensor, sampler, H, W, beta=beta, add_light_samples=False,
    )
    if not segment_pool.paths:
        return np.zeros((H, W, 3), dtype=np.float64), 0
    cluster.set_segments(segment_pool.paths)
    pair_cache = build_pair_cache_ball(cluster, samplers, radius=radius)
    mmis.compute_all_mmis_weights_vec(cluster, pair_cache)
    propagation.iterate_propogation_vec(cluster, pair_cache)
    img = _final_gather(camera_first_segments, H, W)
    return img, len(pair_cache.prop_i)


def run_radius(scene, sensor, renderer, H, W, max_iter, radius,
               geom_clamp, N, ref, ref_mean):
    cluster, mmis, propagation, samplers = _make_blt_components(
        scene, add_light_samples=False,
        num_prop_iterations=N, cluster_c=30,
        geom_clamp_factor=geom_clamp, geom_clamp_mode="hard",
    )
    accum = np.zeros((H, W, 3), dtype=np.float64)
    rmses, mrs, pairs = [], [], []
    last_img = None
    for i in range(max_iter):
        t0 = time.time()
        img, n_pairs = render_one_iter_ball(
            scene, sensor, renderer, H, W,
            cluster, mmis, propagation, samplers,
            iteration=i, radius=radius,
        )
        dt = time.time() - t0
        accum += img.astype(np.float64)
        running = accum / (i + 1)
        rmses.append(float(np.sqrt(((running - ref) ** 2).mean())))
        mrs.append(float(running.sum(-1).mean() / max(ref_mean, 1e-12)))
        pairs.append(n_pairs)
        last_img = running
        print(f"   iter {i+1}/{max_iter}  pairs={n_pairs:,}  "
              f"RMSE={rmses[-1]:.4e}  mr={mrs[-1]:.4f}  ({dt:.1f}s)")
    return rmses, mrs, pairs, last_img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=None)
    ap.add_argument("--max-iter", type=int, default=16)
    ap.add_argument("--width", type=int, default=50)
    ap.add_argument("--radii", nargs="+", type=float,
                    default=[0.50, 0.20, 0.10, 0.05, 0.02, 0.01],
                    help="radii to sweep (large to small)")
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--geom-clamp", type=float, default=200.0)
    ap.add_argument("--refresh", action="store_true",
                    help="ignore cached EXR/metrics and re-render every radius")
    args = ap.parse_args()

    W = H = args.width
    print(f"config: W={W}  max_iter={args.max_iter}  N={args.N}  "
          f"geom_clamp={args.geom_clamp}")
    print(f"sweeping radii: {args.radii}")

    ref_path = args.ref or (
        f"/Users/celine/Documents/projects/ray-maps/baseline/"
        f"spp_renders-cbox-{W}x{H}/reference.exr"
    )
    ref = load_ref(ref_path, W, H)
    ref_mean = float(ref.sum(-1).mean())
    print(f"reference: {ref_path}\n  mean = {ref_mean:.4e}")

    scene = cornell_scene(W, H)
    sensor = scene.sensors()[0]
    renderer = Renderer()

    results = {}  # radius → (rmses, mrs, pairs, last_img)
    for r in args.radii:
        print(f"\n── radius = {r} ──")
        cache_tag = (f"N_w{W}_i{args.max_iter}_r{r}_N{args.N}"
                     f"_g{int(args.geom_clamp)}")
        hit = None if args.refresh else load_render(cache_tag)
        if hit is not None and hit["meta"] is not None:
            m = hit["meta"]
            rmses, mrs, pairs, img = m["rmses"], m["mrs"], m["pairs"], hit["image"]
            print(f"   (cache hit: {cache_tag})")
        else:
            rmses, mrs, pairs, img = run_radius(
                scene, sensor, renderer, H, W,
                args.max_iter, r, args.geom_clamp, args.N, ref, ref_mean,
            )
            save_render(cache_tag, img, meta={
                "config": {"W": W, "max_iter": args.max_iter, "radius": r,
                           "N": args.N, "geom_clamp": args.geom_clamp,
                           "lookup": "ball", "c": 30},
                "ref": os.path.basename(ref_path),
                "rmses": rmses, "mrs": mrs, "pairs": pairs,
            })
        results[r] = (rmses, mrs, pairs, img)

    # ── summary ─────────────────────────────────────────────────
    print("\n── summary @ final iter ─────────────────────────────")
    print(f"  {'radius':>8s} {'pairs/iter':>12s} {'RMSE':>11s} {'mean_ratio':>11s}")
    for r in args.radii:
        rmses, mrs, pairs, _ = results[r]
        print(f"  {r:>8.4f} {np.mean(pairs):>12.0f} "
              f"{rmses[-1]:>11.3e} {mrs[-1]:>11.4f}")

    best_r = min(args.radii, key=lambda r: results[r][0][-1])
    print(f"\nlowest final RMSE: r={best_r}  RMSE={results[best_r][0][-1]:.4e}")

    # ── plots ───────────────────────────────────────────────────
    iters = np.arange(1, args.max_iter + 1)
    n_r = len(args.radii)
    fig = plt.figure(figsize=(15, 4 + 3 * ((n_r + 2) // 3)))
    gs = fig.add_gridspec(2 + ((n_r + 2) // 3), 3,
                          height_ratios=[2, 2] + [1.5] * ((n_r + 2) // 3))

    # Convergence panel
    ax = fig.add_subplot(gs[0, :])
    for r in args.radii:
        rmses, mrs, _, _ = results[r]
        slope, _ = np.polyfit(np.log(iters), np.log(rmses), 1)
        ax.loglog(iters, rmses, "o-", label=f"r={r}  (slope={slope:.2f})")
    ax.loglog(iters, results[args.radii[0]][0][0] * iters ** (-0.5),
              "k--", alpha=0.4, label="N^(-1/2)")
    ax.set_xlabel("iter"); ax.set_ylabel("RMSE")
    ax.set_title("RMSE vs iter, fixed ball lookup, swept r")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)

    # mean ratio + final-RMSE-vs-radius
    ax = fig.add_subplot(gs[1, 0])
    for r in args.radii:
        _, mrs, _, _ = results[r]
        ax.plot(iters, mrs, "o-", label=f"r={r}")
    ax.axhline(1.0, color="k", linestyle="--", alpha=0.5)
    ax.set_xlabel("iter"); ax.set_ylabel("mean_ratio")
    ax.set_title("mean ratio")
    ax.set_ylim(0.5, 1.2); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 1])
    final_rmses = [results[r][0][-1] for r in args.radii]
    final_mrs = [results[r][1][-1] for r in args.radii]
    ax.semilogx(args.radii, final_rmses, "o-")
    ax.set_xlabel("radius"); ax.set_ylabel("final RMSE")
    ax.set_title("RMSE vs radius (lower better)")
    ax.grid(True, which="both", alpha=0.3)

    ax = fig.add_subplot(gs[1, 2])
    ax.semilogx(args.radii, final_mrs, "o-")
    ax.axhline(1.0, color="k", linestyle="--", alpha=0.5)
    ax.set_xlabel("radius"); ax.set_ylabel("final mean_ratio")
    ax.set_title("bias vs radius")
    ax.set_ylim(0.5, 1.1); ax.grid(True, which="both", alpha=0.3)

    # Image grid
    ref_lum = ref.sum(-1)
    med = float(np.median(ref_lum[ref_lum > 0])) if (ref_lum > 0).any() else 0.0
    exposure = (0.18 / med) if med > 0 else 1.0
    def tonemap(x):
        s = x * exposure
        return np.clip((s / (1 + s)) ** (1/2.2), 0, 1)

    for idx, r in enumerate(args.radii):
        row = 2 + idx // 3
        col = idx % 3
        ax = fig.add_subplot(gs[row, col])
        _, mrs, _, img = results[r]
        ax.imshow(tonemap(img))
        ax.set_title(f"r={r}  mr={mrs[-1]:.3f}  RMSE={results[r][0][-1]:.3e}",
                     fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    tag = f"w{W}_i{args.max_iter}_N{args.N}_g{int(args.geom_clamp)}"
    out = os.path.join(os.path.dirname(__file__), f"N_radius_sweep_{tag}.png")
    plt.savefig(out, dpi=110); plt.close()
    print(f"\nsaved: {out}")

    print("\n── interpretation ──")
    print("  Clean U-shape in 'RMSE vs radius' = density estimation working as expected.")
    print("  Minimum at r→∞                   = kernel doing nothing useful.")
    print("  Minimum at r→0                   = variance-limited; kernel can't help.")
    print("  Flat mean_ratio across radii     = bias floor set by clamp, kernel irrelevant.")
    print("  Monotonic mean_ratio drop        = larger kernel = more bias (expected).")


if __name__ == "__main__":
    main()
