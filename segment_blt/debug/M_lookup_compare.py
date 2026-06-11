"""
M. Lookup-mechanism A/B at fixed kernel radius.

Tests whether the voxel-hash cluster lookup itself biases results vs
a true radius search (KD-tree). Both runs use the same kernel size:

  • Voxel side L (computed from cluster.cluster() with voxel_size=L)
  • Ball radius r matched so π r² ≈ L² (ball's cross-section matches
    voxel face area, since light transport is on 2D surfaces)
    → r = L / sqrt(π) ≈ 0.564 L

If results differ: voxel discretization (boundary jitter, asymmetric
voxel position per receiver, ≤5-endpoint skip) is biasing things.
If results agree: lookup mechanism doesn't matter, and any remaining
bias is from the kernel size or the clamp.

50×50 keeps the slow ball-search tractable. We render n_iter at the
same fixed radius for each mechanism and overlay convergence curves.

Usage:
    # Make sure you have a 50×50 reference first:
    python make_reference.py --width 50 --spp 8192

    # Then:
    python M_lookup_compare.py --max-iter 16 --voxel-size 0.085

The default voxel-size matches what cluster_c=30 produces for the
Cornell box, so the kernel area is comparable to what you've been
running at 128².
"""
from _setup import cornell_scene
from _cache import save_render, load_render
from K_ball_kernel import build_pair_cache_ball  # reuse the ball-search builder

import argparse
import math
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import mitsuba as mi

from segments.renderer import Renderer, _make_blt_components, _sample_segments, _final_gather
from segments.pair_cache import build_pair_cache as build_pair_cache_voxel


def load_ref(path, W, H):
    arr = np.array(mi.Bitmap(path))[..., :3].astype(np.float64)
    if arr.shape[:2] != (H, W):
        raise SystemExit(
            f"ref shape {arr.shape[:2]} ≠ {(H, W)}.\n"
            f"Generate: python make_reference.py --width {W} --spp 8192"
        )
    return arr


def render_one_iter(mode, scene, sensor, renderer, H, W,
                    cluster, mmis, propagation, samplers,
                    iteration, voxel_size, ball_radius, beta=0.1):
    """One iter using either voxel or ball lookup. Identical otherwise."""
    sampler = mi.load_dict({"type": "independent", "sample_count": 1})
    sampler.seed(iteration, W * H)

    segment_pool, camera_first_segments = _sample_segments(
        scene, sensor, sampler, H, W, beta=beta, add_light_samples=False,
    )
    if not segment_pool.paths:
        return np.zeros((H, W, 3), dtype=np.float64), 0

    cluster.set_segments(segment_pool.paths)

    if mode == "voxel":
        # Force fixed voxel size — no progressive shrinkage, matches ball's fixed r
        cluster.cluster(np.random.default_rng(seed=iteration), voxel_size=voxel_size)
        pair_cache = build_pair_cache_voxel(cluster, samplers)
    elif mode == "ball":
        # Skip cluster.cluster() — KD-tree lookup doesn't need voxel grid
        pair_cache = build_pair_cache_ball(cluster, samplers, radius=ball_radius)
    else:
        raise ValueError(f"unknown mode {mode}")

    mmis.compute_all_mmis_weights_vec(cluster, pair_cache)
    propagation.iterate_propogation_vec(cluster, pair_cache)
    img = _final_gather(camera_first_segments, H, W)
    return img, len(pair_cache.prop_i)


def run_mode(mode, scene, sensor, renderer, H, W,
             max_iter, voxel_size, ball_radius, geom_clamp, N, ref, ref_mean):
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
        img, n_pairs = render_one_iter(
            mode, scene, sensor, renderer, H, W,
            cluster, mmis, propagation, samplers,
            iteration=i, voxel_size=voxel_size, ball_radius=ball_radius,
        )
        dt = time.time() - t0
        accum += img.astype(np.float64)
        running = accum / (i + 1)
        rmses.append(float(np.sqrt(((running - ref) ** 2).mean())))
        mrs.append(float(running.sum(-1).mean() / max(ref_mean, 1e-12)))
        pairs.append(n_pairs)
        last_img = running
        print(f"   [{mode}] iter {i+1}/{max_iter}  pairs={n_pairs:,}  "
              f"RMSE={rmses[-1]:.4e}  mr={mrs[-1]:.4f}  ({dt:.1f}s)")
    return rmses, mrs, pairs, last_img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=None,
                    help="defaults to ~/.../baseline/spp_renders-WxW-d-1/reference.exr")
    ap.add_argument("--max-iter", type=int, default=16)
    ap.add_argument("--width", type=int, default=50)
    ap.add_argument("--voxel-size", type=float, default=0.085,
                    help="voxel side L. Ball radius set to L/√π to match cross-section area.")
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--geom-clamp", type=float, default=200.0)
    ap.add_argument("--refresh", action="store_true",
                    help="ignore cached EXR/metrics and re-render both modes")
    args = ap.parse_args()

    W = H = args.width
    L = args.voxel_size
    r_ball = L / math.sqrt(math.pi)  # match cross-sectional area πr² = L²
    print(f"config: W={W} max_iter={args.max_iter}  N={args.N}  geom_clamp={args.geom_clamp}")
    print(f"voxel side L = {L:.4e}    ball radius r = L/√π = {r_ball:.4e}")
    print(f"  voxel face area  = L²    = {L*L:.4e}")
    print(f"  ball cross-sect. = πr²  = {math.pi*r_ball*r_ball:.4e}  (matched)")

    ref_path = args.ref or (
        f"/Users/celine/Documents/projects/ray-maps/baseline/"
        f"spp_renders-{W}x{H}-d-1/reference.exr"
    )
    ref = load_ref(ref_path, W, H)
    ref_mean = float(ref.sum(-1).mean())
    print(f"reference: {ref_path}\n  mean = {ref_mean:.4e}")

    scene = cornell_scene(W, H)
    sensor = scene.sensors()[0]
    renderer = Renderer()

    def cached_mode(mode):
        cache_tag = (f"M_{mode}_w{W}_i{args.max_iter}_L{L}_r{r_ball:.3f}"
                     f"_N{args.N}_g{int(args.geom_clamp)}")
        hit = None if args.refresh else load_render(cache_tag)
        if hit is not None and hit["meta"] is not None:
            m = hit["meta"]
            print(f"   (cache hit: {cache_tag})")
            return m["rmses"], m["mrs"], m["pairs"], hit["image"]
        rmses, mrs, pairs, img = run_mode(
            mode, scene, sensor, renderer, H, W,
            args.max_iter, L, r_ball, args.geom_clamp, args.N, ref, ref_mean,
        )
        save_render(cache_tag, img, meta={
            "config": {"W": W, "max_iter": args.max_iter, "voxel_size": L,
                       "ball_radius": r_ball, "N": args.N,
                       "geom_clamp": args.geom_clamp, "mode": mode},
            "ref": os.path.basename(ref_path),
            "rmses": rmses, "mrs": mrs, "pairs": pairs,
        })
        return rmses, mrs, pairs, img

    print(f"\n── VOXEL lookup ────────────────────────────────")
    vr, vm, vp, vimg = cached_mode("voxel")

    print(f"\n── BALL lookup (exact radius) ──────────────────")
    br, bm, bp, bimg = cached_mode("ball")

    print(f"\n── summary @ final iter ─────────────────────────")
    print(f"  voxel: RMSE={vr[-1]:.4e}  mr={vm[-1]:.4f}  pairs/iter≈{np.mean(vp):.0f}")
    print(f"  ball : RMSE={br[-1]:.4e}  mr={bm[-1]:.4f}  pairs/iter≈{np.mean(bp):.0f}")
    print(f"  RMSE ratio (ball/voxel): {br[-1]/vr[-1]:.3f}")
    print(f"  mr   diff   (ball-voxel): {bm[-1]-vm[-1]:+.4f}")

    # ── plot ────────────────────────────────────────────────────
    iters = np.arange(1, args.max_iter + 1)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    v_slope, _ = np.polyfit(np.log(iters), np.log(vr), 1)
    b_slope, _ = np.polyfit(np.log(iters), np.log(br), 1)

    axes[0, 0].loglog(iters, vr, "o-", label=f"voxel (slope={v_slope:.2f})")
    axes[0, 0].loglog(iters, br, "s-", label=f"ball  (slope={b_slope:.2f})")
    axes[0, 0].loglog(iters, vr[0] * iters ** (-0.5), "k--", alpha=0.5, label="N^(-1/2)")
    axes[0, 0].set_xlabel("iter"); axes[0, 0].set_ylabel("RMSE")
    axes[0, 0].set_title(f"lookup A/B at L={L}, r={r_ball:.3f}")
    axes[0, 0].grid(True, which="both", alpha=0.3); axes[0, 0].legend()

    axes[0, 1].plot(iters, vm, "o-", label="voxel")
    axes[0, 1].plot(iters, bm, "s-", label="ball")
    axes[0, 1].axhline(1.0, color="k", linestyle="--", alpha=0.5)
    axes[0, 1].set_xlabel("iter"); axes[0, 1].set_ylabel("mean_ratio")
    axes[0, 1].set_title("mean ratio")
    axes[0, 1].set_ylim(0.7, 1.2); axes[0, 1].grid(True, alpha=0.3); axes[0, 1].legend()

    ref_lum = ref.sum(-1)
    med = float(np.median(ref_lum[ref_lum > 0])) if (ref_lum > 0).any() else 0.0
    exposure = (0.18 / med) if med > 0 else 1.0
    def tonemap(x):
        s = x * exposure
        return np.clip((s / (1 + s)) ** (1/2.2), 0, 1)
    axes[1, 0].imshow(tonemap(vimg))
    axes[1, 0].set_title(f"VOXEL  mr={vm[-1]:.3f}"); axes[1, 0].axis("off")
    axes[1, 1].imshow(tonemap(bimg))
    axes[1, 1].set_title(f"BALL   mr={bm[-1]:.3f}"); axes[1, 1].axis("off")

    plt.tight_layout()
    tag = f"w{W}_i{args.max_iter}_L{L}_r{r_ball:.3f}_N{args.N}_g{int(args.geom_clamp)}"
    out = os.path.join(os.path.dirname(__file__), f"M_lookup_compare_{tag}.png")
    plt.savefig(out, dpi=110); plt.close()
    print(f"\nsaved: {out}")

    # Interpretation hints
    print("\n── interpretation ──")
    print("  If RMSE curves track each other AND mean_ratio matches:")
    print("    lookup mechanism is irrelevant — clustering bias is benign.")
    print("  If voxel RMSE is consistently higher:")
    print("    voxel-specific artifacts (jitter, ≤5 skip, boundaries) are real.")
    print("  If they diverge in mean_ratio:")
    print("    different coverage patterns produce different bias floors.")


if __name__ == "__main__":
    main()
