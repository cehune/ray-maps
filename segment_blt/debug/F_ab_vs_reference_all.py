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

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import mitsuba as mi
from segments.renderer import Renderer


DEFAULT_REF = "/Users/celine/Documents/projects/ray-maps/baseline/spp_renders-128x128-d-1/reference.exr"


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
        print(f"rendering BLT  W={W} H={H} n_iter={args.n_iter} N={args.N} c={args.c}")
        scene = cornell_scene(W, H)
        renderer = Renderer()
        blt = renderer.render_vec(
            scene, height=H, width=W, n_iterations=args.n_iter,
            num_prop_iterations=args.N, cluster_c=args.c,
            geom_clamp_factor=geom_f, mmis_clamp_factor=mmis_f,
            geom_clamp_mode=args.geom_clamp_mode,
            verbose=False, beta=0.1, add_light_samples=False,
        )

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
    # per-pixel signed luminance diff, in tonemapped space (what eye sees)
    diff_tm = blt_tm.mean(-1) - ref_tm.mean(-1)
    dmax = float(np.abs(diff_tm).max() or 1.0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(ref_tm); axes[0].set_title("REFERENCE"); axes[0].axis("off")
    axes[1].imshow(blt_tm); axes[1].set_title("BLT");        axes[1].axis("off")
    im = axes[2].imshow(diff_tm, cmap="RdBu_r", vmin=-dmax, vmax=dmax)
    axes[2].set_title(f"BLT - REF  (Δ luma in display space)")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046)
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "F_ab_vs_reference_100_iterations_15kernel.png")
    plt.savefig(out, dpi=110); plt.close()
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
