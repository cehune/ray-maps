"""
K. Exact-radius vs voxel-hash pair enumeration.

The question: is the voxel-bucket clustering approximating "all pairs within
radius r" well enough? An axis-aligned voxel of side r is NEITHER a ball of
radius r NOR centered on the query point, so:
  • Two endpoints 0.1·r apart can land in DIFFERENT voxels and never form
    a pair (false negative).
  • Two endpoints in opposite corners of the same voxel, up to r·√3 apart,
    DO form a pair (false positive — they're outside the intended kernel).

This test holds everything else fixed (same scene, same seed, same N, same
weights) and swaps ONLY the neighborhood enumeration:
  • Voxel-hash version  → segments/pair_cache.py:build_pair_cache
  • Exact-radius version → segments/pair_cache.py:build_pair_cache_exact

If the images differ meaningfully — different mean, splotchy regions in one
but not the other, large RMSE between the two — the voxel-hash is doing a bad
job approximating the kernel support, and any subsequent kernel/weight
diagnostic (script L) needs to use the exact version to remove that confound.

If the images agree, the voxel-hash is a faithful approximation at this r, and
you can use it everywhere without worry.

Small image (50×50) is sufficient — exact-radius is O(S²) and gets expensive
fast.  Adjust --r-factor to compare at different effective neighborhood sizes:
  --r-factor 1.0  : r = voxel_size_0 — equal effective scale
  --r-factor 0.5  : r = 0.5 × voxel_size_0 — tighter ball, exposes false +s
  --r-factor 2.0  : r = 2 × voxel_size_0 — looser ball, exposes false -s

CLI:
  python K_exact_vs_voxel.py --width 50 --n-iter 8 --r-factor 1.0
"""
from _setup import cornell_scene

import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import mitsuba as mi

from segments.renderer import _make_blt_components, _sample_segments, _final_gather
from segments.pair_cache import build_pair_cache, build_pair_cache_exact


# ─── Tonemap (identical to debug/F_ab_vs_reference.py:reinhard_tonemap) ───────
def reinhard_tonemap(img, exposure, gamma=2.2):
    scaled = np.asarray(img, dtype=np.float64) * exposure
    compressed = scaled / (1.0 + scaled)
    return np.clip(compressed ** (1.0 / gamma), 0.0, 1.0).astype(np.float32)


def auto_exposure(img, target_grey=0.18):
    """F's auto-exposure rule: target_grey / median Σ-RGB of positive pixels."""
    lum = np.asarray(img, dtype=np.float64).sum(-1)
    pos = lum[lum > 0]
    if pos.size == 0:
        return 1.0
    med = float(np.median(pos))
    return (target_grey / med) if med > 0 else 1.0


def parse_clamp(s):
    """Parse '200' or '20.0' as a float, 'none'/'off'/'null' as None."""
    if s is None or str(s).lower() in ("none", "off", "null"):
        return None
    return float(s)


def render_one_iter_swappable(
    scene, sensor, sampler, H, W,
    cluster, mmis, propagation, samplers,
    *,
    use_exact_radius: bool,
    r: float,
    iteration: int = 0,
    beta: float = 0.1,
    add_light_samples: bool = False,
):
    """One BLT iteration, but with a swappable pair-cache builder.

    The hand-rolled version of _render_iterate_vec, minus the timing prints
    and diagnostics, so we can substitute build_pair_cache_exact cleanly.
    """
    segment_pool, camera_first_segments = _sample_segments(
        scene, sensor, sampler, H, W, beta, add_light_samples
    )
    if not segment_pool.paths:
        return np.zeros((H, W, 3))

    cluster.set_segments(segment_pool.paths)
    cluster.cluster(np.random.default_rng(seed=iteration))

    if use_exact_radius:
        pair_cache = build_pair_cache_exact(cluster, samplers, r)
    else:
        pair_cache = build_pair_cache(cluster, samplers)

    mmis.compute_all_mmis_weights_vec(cluster, pair_cache)
    propagation.iterate_propogation_vec(cluster, pair_cache)
    return _final_gather(camera_first_segments, H, W), pair_cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=50, help="image size (kept small — exact-radius is O(S²))")
    ap.add_argument("--n-iter", type=int, default=8, help="averaging iterations")
    ap.add_argument("--N", type=int, default=8, help="num_prop_iterations per render")
    ap.add_argument("--c", type=int, default=30, help="cluster_c for voxel-hash version")
    ap.add_argument("--r-factor", type=float, default=1.0,
                    help="exact-radius r = r-factor × voxel_size_0 (default 1.0)")
    ap.add_argument("--geom-clamp", default="200",
                    help="geom_term cap factor (× median). 'none' = off. "
                         "Default 200 matches your F/H sweep settings — keeps "
                         "this script consistent with prior runs so the diff "
                         "between voxel and exact isn't masked by firefly noise.")
    ap.add_argument("--mmis-clamp", default="none",
                    help="mmis_weight cap factor (× median). 'none' = off.")
    args = ap.parse_args()

    geom_f = parse_clamp(args.geom_clamp)
    mmis_f = parse_clamp(args.mmis_clamp)
    print(f"clamps: geom={geom_f}  mmis={mmis_f}")

    W = H = args.width
    scene = cornell_scene(W, H)
    sensor = scene.sensors()[0]

    # ── Compute voxel_size_0 once, using a fresh cluster on a sample frame ──
    cluster_probe, _, _, samplers_probe = _make_blt_components(
        scene, add_light_samples=False, num_prop_iterations=args.N, cluster_c=args.c,
        geom_clamp_factor=geom_f, mmis_clamp_factor=mmis_f,
    )
    sampler = mi.load_dict({"type": "independent", "sample_count": 1})
    sampler.seed(0, W * H)
    probe_pool, _ = _sample_segments(scene, sensor, sampler, H, W, 0.1, False)
    cluster_probe.set_segments(probe_pool.paths)
    cluster_probe.cluster(np.random.default_rng(seed=0))
    voxel_size_0 = float(cluster_probe.voxel_size)
    r = args.r_factor * voxel_size_0

    print(f"voxel_size_0 = {voxel_size_0:.4e}")
    print(f"exact-radius r = {r:.4e}   (r-factor = {args.r_factor})")
    print(f"image: {W}×{H}, n_iter={args.n_iter}, N={args.N}\n")

    # ── Run both pipelines with same seed, fresh components per side ────────
    def fresh_components():
        return _make_blt_components(
            scene, add_light_samples=False,
            num_prop_iterations=args.N, cluster_c=args.c,
            geom_clamp_factor=geom_f, mmis_clamp_factor=mmis_f,
        )

    cluster_v, mmis_v, prop_v, samplers_v = fresh_components()
    cluster_e, mmis_e, prop_e, samplers_e = fresh_components()

    accum_v = np.zeros((H, W, 3), dtype=np.float64)
    accum_e = np.zeros((H, W, 3), dtype=np.float64)

    pair_counts_v = []  # # of prop pairs per iter, voxel
    pair_counts_e = []  # # of prop pairs per iter, exact

    for i in range(args.n_iter):
        # voxel
        s_v = mi.load_dict({"type": "independent", "sample_count": 1})
        s_v.seed(i, W * H)
        t0 = time.time()
        img_v, pc_v = render_one_iter_swappable(
            scene, sensor, s_v, H, W,
            cluster_v, mmis_v, prop_v, samplers_v,
            use_exact_radius=False, r=r, iteration=i,
        )
        t_v = time.time() - t0
        accum_v += img_v.astype(np.float64)
        pair_counts_v.append(len(pc_v.prop_i))

        # exact
        s_e = mi.load_dict({"type": "independent", "sample_count": 1})
        s_e.seed(i, W * H)
        t0 = time.time()
        img_e, pc_e = render_one_iter_swappable(
            scene, sensor, s_e, H, W,
            cluster_e, mmis_e, prop_e, samplers_e,
            use_exact_radius=True, r=r, iteration=i,
        )
        t_e = time.time() - t0
        accum_e += img_e.astype(np.float64)
        pair_counts_e.append(len(pc_e.prop_i))

        print(f"  iter {i+1}/{args.n_iter}  "
              f"voxel: {t_v:5.1f}s, P={pair_counts_v[-1]:>6d}   "
              f"exact: {t_e:5.1f}s, P={pair_counts_e[-1]:>6d}")

    img_v = accum_v / args.n_iter
    img_e = accum_e / args.n_iter

    lum_v = img_v.sum(-1)
    lum_e = img_e.sum(-1)
    diff = img_v - img_e  # signed

    # ── Statistics ───────────────────────────────────────────────────────────
    print(f"\n── Per-image stats (linear radiance, Σ-RGB per pixel) ──")
    fmt = "  {:6s}  mean={:.4e}   median={:.4e}   p99={:.4e}   max={:.4e}"
    print(fmt.format("voxel", lum_v.mean(), float(np.median(lum_v)),
                     float(np.percentile(lum_v, 99)), float(lum_v.max())))
    print(fmt.format("exact", lum_e.mean(), float(np.median(lum_e)),
                     float(np.percentile(lum_e, 99)), float(lum_e.max())))

    print(f"\n── Voxel vs exact (the diagnostic) ──")
    print(f"  mean_ratio voxel/exact = {lum_v.mean() / max(lum_e.mean(), 1e-12):.4f}")
    print(f"  RMSE(voxel, exact)     = {np.sqrt((diff ** 2).mean()):.4e}")
    print(f"  |diff| max             = {np.abs(diff).max():.4e}")
    print(f"  pairs: voxel avg = {np.mean(pair_counts_v):,.0f}   "
          f"exact avg = {np.mean(pair_counts_e):,.0f}   "
          f"ratio = {np.mean(pair_counts_e)/max(np.mean(pair_counts_v),1):.2f}×")

    # ── Plot ─────────────────────────────────────────────────────────────────
    # Match F's tonemap pipeline: shared exposure off the EXACT image's median
    # (the more-trusted reference of this A/B), apply Reinhard + gamma 2.2.
    # Sharing the exposure across both panels makes them directly comparable —
    # any brightness difference you see is a real difference, not exposure drift.
    exposure = auto_exposure(img_e, target_grey=0.18)
    print(f"\nshared tonemap exposure (from exact-radius median) = {exposure:.4e}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(reinhard_tonemap(img_v, exposure))
    axes[0].set_title(f"voxel-hash\nmean={lum_v.mean():.3e}, p99={np.percentile(lum_v,99):.3e}")
    axes[0].axis("off")

    axes[1].imshow(reinhard_tonemap(img_e, exposure))
    axes[1].set_title(f"exact r={r:.3e}\nmean={lum_e.mean():.3e}, p99={np.percentile(lum_e,99):.3e}")
    axes[1].axis("off")

    # Diff panel: compute in tonemapped (display) space, like F does — that's
    # what the eye sees. Stays signed so the diverging colormap works.
    diff_tm = (reinhard_tonemap(img_v, exposure).mean(-1)
               - reinhard_tonemap(img_e, exposure).mean(-1))
    dmax = float(np.abs(diff_tm).max() or 1.0)
    im = axes[2].imshow(diff_tm, cmap="RdBu_r", vmin=-dmax, vmax=dmax)
    axes[2].set_title("voxel − exact  (Δ luma in display space)\n"
                      "red = voxel brighter, blue = voxel darker")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046)

    # Filename encodes every CLI setting that affects the result, so you can
    # eyeball which run is which without opening a notes file.
    parts = [
        f"w{W}", f"n{args.n_iter}", f"N{args.N}", f"c{args.c}",
        f"rf{args.r_factor}", f"g{args.geom_clamp}",
    ]
    if args.mmis_clamp != "none":
        parts.append(f"m{args.mmis_clamp}")
    suffix = "_".join(parts)
    out = os.path.join(os.path.dirname(__file__), f"K_exact_vs_voxel_{suffix}.png")
    plt.tight_layout()
    plt.savefig(out, dpi=110)
    plt.close()
    print(f"\nsaved {out}")

    # ── Interpretation guide ─────────────────────────────────────────────────
    print("\n── How to read this ──")
    print("  • If voxel and exact images look IDENTICAL and RMSE is tiny:")
    print("      voxel hash is faithfully approximating the radius-r ball.")
    print("      Splotches & convergence floor are NOT from the data structure.")
    print("      → Proceed to L (r-sweep) using the voxel version freely.")
    print("  • If voxel image is splotchier OR much brighter/darker than exact:")
    print("      voxel hash is approximating badly.")
    print("      → Run L with EXACT radius to remove this confound.")
    print("      → Long term, replace cluster lookup with a real KD-tree or")
    print("        ball-tree (or sphere-test inside the voxel before keeping a pair).")


if __name__ == "__main__":
    main()
