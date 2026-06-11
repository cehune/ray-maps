"""
python debug/F_ab_vs_reference.py --n-iter 8 --N 8 --c 30

F. A/B BLT against a reference EXR.

Either renders a fresh BLT image OR loads one from disk, then:
  • aligns both to the SAME exposure (computed from the reference's median
    so the comparison isn't biased by BLT's own firefly tail);
  • saves a side-by-side tonemapped PNG;
  • prints per-pixel and aggregate error metrics.

Usage:
    # render BLT fresh and compare:
    python F_ab_vs_reference.py --n-iter 8 --c 30 --N 8

    # compare a saved BLT EXR/PNG (no rendering — fast):
    python F_ab_vs_reference.py --blt-path ../outputs/blt.exr

    # override reference path:
    python F_ab_vs_reference.py --ref /path/to/reference.exr
"""
from _setup import cornell_scene
from K_ball_kernel import build_pair_cache_ball

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import mitsuba as mi
from segments.renderer import Renderer, _make_blt_components, _sample_segments, _final_gather


DEFAULT_REF = "/Users/celine/Documents/projects/ray-maps/baseline/spp_renders-50x50-d-1/reference.exr"


def reinhard_tonemap(img, exposure, gamma=2.2):
    """Same pipeline as render.save_with_tonemap, but pure (no file write)."""
    scaled = np.asarray(img, dtype=np.float64) * exposure
    compressed = scaled / (1.0 + scaled)
    return np.clip(compressed ** (1.0 / gamma), 0.0, 1.0).astype(np.float32)


def load_rgb(path):
    arr = np.array(mi.Bitmap(path))
    if arr.ndim != 3 or arr.shape[-1] < 3:
        raise SystemExit(f"unexpected image shape {arr.shape}")
    return arr[..., :3].astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=DEFAULT_REF, help="reference EXR")
    ap.add_argument("--blt-path", default=None,
                    help="optional: load a saved BLT image instead of rendering")
    ap.add_argument("--n-iter", type=int, default=24, help="BLT render iterations (averaging)")
    ap.add_argument("--N", type=int, default=8, help="num_prop_iterations per render")
    ap.add_argument("--c", type=int, default=30, help="cluster_c")
    ap.add_argument("--target-grey", type=float, default=0.18,
                    help="middle-grey for auto-exposure (off ref median)")
    ap.add_argument("--geom-clamp", default="none",
                    help="geom_term cap factor (× median). 'none' = off.")
    ap.add_argument("--geom-clamp-mode", default="hard", choices=("hard", "soft"))
    ap.add_argument("--mmis-clamp", default="none",
                    help="mmis_weight cap factor (× median). 'none' = off.")
    ap.add_argument("--lookup", default="voxel", choices=("voxel", "ball"),
                    help="neighbor enumeration. 'voxel' = current cluster.cluster() "
                         "hash grid (fast). 'ball' = exact KD-tree radius query "
                         "(slow, but no voxel artifacts and tunable radius).")
    ap.add_argument("--radius", type=float, default=0.12,
                    help="ball radius in scene units. Default 0.12 = N-sweep "
                         "unbiased point. When --progressive, this is r₀.")
    ap.add_argument("--progressive", action="store_true",
                    help="(ball only) shrink radius via Knaus-Zwicker r_n = r_0 * n^(-α/2). "
                         "Drives local-bias floor toward 0 as iters grow.")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="K-Z α. Paper uses 2/3 (aggressive). We use 0.5 because "
                         "smaller r crashes coverage at our sample density.")
    args = ap.parse_args()

    geom_f = None if str(args.geom_clamp).lower() in ("none", "off") else float(args.geom_clamp)
    mmis_f = None if str(args.mmis_clamp).lower() in ("none", "off") else float(args.mmis_clamp)

    ref = load_rgb(args.ref)
    H, W = ref.shape[:2]
    print(f"reference: {args.ref}  shape={ref.shape}")

    # ── BLT side ────────────────────────────────────────────────────
    if args.blt_path:
        blt = load_rgb(args.blt_path)
        if blt.shape[:2] != (H, W):
            raise SystemExit(f"BLT size {blt.shape[:2]} ≠ ref {(H, W)} — re-render at matching res")
        print(f"BLT (loaded): {args.blt_path}")
    else:
        print(f"rendering BLT  W={W} H={H} n_iter={args.n_iter} N={args.N} "
              f"c={args.c} lookup={args.lookup}"
              f"{f' radius={args.radius}' if args.lookup == 'ball' else ''}")
        scene = cornell_scene(W, H)
        renderer = Renderer()

        if args.lookup == "voxel":
            blt = renderer.render_vec(
                scene, height=H, width=W, n_iterations=args.n_iter,
                num_prop_iterations=args.N, cluster_c=args.c,
                geom_clamp_factor=geom_f, mmis_clamp_factor=mmis_f,
                geom_clamp_mode=args.geom_clamp_mode,
                verbose=False, beta=0.1, add_light_samples=False,
            )
        else:
            # ball search via KD-tree — bypasses cluster.cluster() entirely
            sensor = scene.sensors()[0]
            cluster, mmis, propagation, samplers = _make_blt_components(
                scene, add_light_samples=False,
                num_prop_iterations=args.N, cluster_c=args.c,
                geom_clamp_factor=geom_f, mmis_clamp_factor=mmis_f,
                geom_clamp_mode=args.geom_clamp_mode,
            )
            accum = np.zeros((H, W, 3), dtype=np.float64)
            for i in range(args.n_iter):
                # Knaus-Zwicker shrinkage: r_n = r_0 * n^(-α/2). n=1 → r_0.
                if args.progressive:
                    r_i = args.radius * ((i + 1) ** (-args.alpha / 2))
                else:
                    r_i = args.radius

                sampler = mi.load_dict({"type": "independent", "sample_count": 1})
                sampler.seed(i, W * H)
                segment_pool, camera_first_segments = _sample_segments(
                    scene, sensor, sampler, H, W, beta=0.1, add_light_samples=False,
                )
                if not segment_pool.paths:
                    continue
                cluster.set_segments(segment_pool.paths)
                pair_cache = build_pair_cache_ball(cluster, samplers, radius=r_i)
                mmis.compute_all_mmis_weights_vec(cluster, pair_cache)
                propagation.iterate_propogation_vec(cluster, pair_cache)
                img = _final_gather(camera_first_segments, H, W)
                accum += img.astype(np.float64)
                print(f"  iter {i+1}/{args.n_iter}  r={r_i:.4f}  "
                      f"pairs={len(pair_cache.prop_i):,}")
            blt = (accum / max(args.n_iter, 1)).astype(np.float32)

    # ── matched exposure ────────────────────────────────────────────
    # Use ref median to pick exposure — firefly-immune AND tied to the
    # ground-truth distribution so the displayed BLT exposure isn't
    # biased by its own variance.
    ref_lum = ref.sum(-1)
    med = float(np.median(ref_lum[ref_lum > 0])) if (ref_lum > 0).any() else 0.0
    exposure = (args.target_grey / med) if med > 0 else 1.0
    print(f"\nmatched exposure (from ref median): {exposure:.4e}")
    print(f"  ref_median_lum = {med:.4e}   target_grey = {args.target_grey}")

    # ── error metrics (linear, pre-tonemap — only meaningful comparison) ──
    diff = blt - ref
    rmse = float(np.sqrt((diff ** 2).mean()))
    mae  = float(np.abs(diff).mean())

    blt_lum = blt.sum(-1)
    print("\n── aggregate stats (linear) ─────────────────────────────")
    print(f"  {'':14s} {'mean':>11s} {'med':>11s} {'p90':>11s} {'p99':>11s} {'max':>11s}")
    for name, arr in [("ref Σ-RGB", ref_lum), ("blt Σ-RGB", blt_lum)]:
        print(f"  {name:14s} "
              f"{arr.mean():>11.3e} {np.median(arr):>11.3e} "
              f"{np.percentile(arr,90):>11.3e} {np.percentile(arr,99):>11.3e} "
              f"{arr.max():>11.3e}")
    print(f"\n  mean ratio blt/ref  = {blt_lum.mean()/max(ref_lum.mean(),1e-12):.4f}")
    print(f"  RMSE (per channel)  = {rmse:.4e}")
    print(f"  MAE  (per channel)  = {mae:.4e}")
    print(f"  rel-RMSE / ref.mean = {rmse/max(ref.mean(),1e-12):.4f}")

    # ── tonemapped side-by-side ─────────────────────────────────────
    ref_tm = reinhard_tonemap(ref, exposure)
    blt_tm = reinhard_tonemap(blt, exposure)
    diff_tm = blt_tm.mean(-1) - ref_tm.mean(-1)
    dmax = float(np.abs(diff_tm).max() or 1.0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(ref_tm); axes[0].set_title("REFERENCE"); axes[0].axis("off")
    axes[1].imshow(blt_tm); axes[1].set_title("BLT");        axes[1].axis("off")
    im = axes[2].imshow(diff_tm, cmap="RdBu_r", vmin=-dmax, vmax=dmax)
    axes[2].set_title("BLT - REF  (Δ luma in display space)")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046)
    plt.tight_layout()

    if args.lookup == "ball":
        spatial = f"ball_r{args.radius}"
        if args.progressive:
            spatial += f"_prog_a{args.alpha}"
    else:
        spatial = f"voxel_c{args.c}"
    tag = (f"w{W}_i{args.n_iter}_N{args.N}_{spatial}"
           f"_g{args.geom_clamp}_m{args.mmis_clamp}")
    out = os.path.join(os.path.dirname(__file__), f"F_ab_vs_reference_{tag}.png")
    plt.savefig(out, dpi=110); plt.close()
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
