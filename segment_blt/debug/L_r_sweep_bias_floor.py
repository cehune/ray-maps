"""
L. r-sweep with EXACT-radius pairs — looking for the bias floor.

For a well-formed kernel density estimator, the per-pixel error decomposes:

    RMSE(r, N) ≈ C_bias · r^β   +   C_var · r^(-d/2) · N^(-1/2)
                  └── bias ──┘       └────── variance ──────┘

  • β > 0 (typically 2 for a smoothing kernel) — bias scales DOWN as r shrinks.
  • d is the effective dimension of the merge manifold (≈ 2 for surface pairs).
  • Variance grows as r shrinks (fewer pairs per receiver → noisier estimate).

So sweeping r gives a U-shape: bias-dominated at large r (right side, drops as
r shrinks), variance-dominated at small r (left side, rises as r shrinks),
minimum at the sweet spot.

THE DIAGNOSTIC: as you shrink r AND push N large enough to make the variance
term small, the BIAS term should approach zero. If RMSE plateaus at a positive
floor that no amount of r-shrinkage or sample budget can lower, your weights
or PDFs are wrong — the MMIS denominator or kernel normalization isn't doing
what the math says it should. THAT floor is the signature of "splotchy and
never converges."

Critical: this script uses build_pair_cache_EXACT (an exact Euclidean ball of
radius r), NOT the voxel-hash version. This removes the data-structure as a
confound. Run K first to decide whether the voxel-hash matters; once you know
it doesn't (or once you accept the slowdown), L tells you about the kernel
math itself.

Slow: O(S²) pair enumeration per iteration. Default 50×50 image is feasible;
larger sizes get expensive fast.

Output: log-log plot of RMSE vs r, plus mean_ratio vs r and a per-r table.
Reads the result automatically — prints the bias-floor estimate.

CLI:
  python L_r_sweep_bias_floor.py --width 50 --n-iter 16 --N 8

  --r-factors  : space-separated list of r/voxel_size_0 multipliers to sweep.
                 Default sweeps coarsely from 4× down to 0.25×.
"""
from _setup import cornell_scene

import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import mitsuba as mi

from segments.renderer import _make_blt_components, _sample_segments, _final_gather
from segments.pair_cache import build_pair_cache_exact


# ─── Tonemap (identical to debug/F_ab_vs_reference.py:reinhard_tonemap) ───────
def reinhard_tonemap(img, exposure, gamma=2.2):
    scaled = np.asarray(img, dtype=np.float64) * exposure
    compressed = scaled / (1.0 + scaled)
    return np.clip(compressed ** (1.0 / gamma), 0.0, 1.0).astype(np.float32)


def auto_exposure(img, target_grey=0.18):
    lum = np.asarray(img, dtype=np.float64).sum(-1)
    pos = lum[lum > 0]
    if pos.size == 0:
        return 1.0
    med = float(np.median(pos))
    return (target_grey / med) if med > 0 else 1.0


def parse_clamp(s):
    if s is None or str(s).lower() in ("none", "off", "null"):
        return None
    return float(s)


DEFAULT_REF = "/Users/mckalechung/Documents/proj/ray-maps/baseline/spp_renders-200x200-d-1/reference.exr"


def load_ref_downsampled(path, target_W, target_H):
    """Load the 200×200 reference and block-average down to (target_W, target_H).

    Block-averaging preserves the linear mean exactly — what we want for an
    RMSE comparison at a different resolution. The downsampled ref's variance
    is lower than a fresh native-resolution render (by 16× for 4× shrinkage),
    so absolute RMSE values are biased slightly LOW vs a true 50² reference;
    but the SHAPE of RMSE(r) is what matters for the bias-floor check.
    """
    arr = np.array(mi.Bitmap(path))[..., :3].astype(np.float64)
    src_H, src_W = arr.shape[:2]
    fy, fx = src_H // target_H, src_W // target_W
    if fy * target_H != src_H or fx * target_W != src_W:
        raise SystemExit(
            f"ref {src_H}×{src_W} not an integer multiple of target {target_H}×{target_W}"
        )
    # block-average: reshape to (target_H, fy, target_W, fx, 3), mean over (fy, fx)
    return arr.reshape(target_H, fy, target_W, fx, 3).mean(axis=(1, 3))


def render_one_iter_exact(
    scene, sensor, sampler, H, W,
    cluster, mmis, propagation, samplers,
    *,
    r: float, iteration: int = 0, beta: float = 0.1,
    add_light_samples: bool = False,
):
    """One BLT iteration with exact-radius pair enumeration. Returns image."""
    segment_pool, camera_first_segments = _sample_segments(
        scene, sensor, sampler, H, W, beta, add_light_samples
    )
    if not segment_pool.paths:
        return np.zeros((H, W, 3)), 0

    cluster.set_segments(segment_pool.paths)
    cluster.cluster(np.random.default_rng(seed=iteration))
    pair_cache = build_pair_cache_exact(cluster, samplers, r)
    mmis.compute_all_mmis_weights_vec(cluster, pair_cache)
    propagation.iterate_propogation_vec(cluster, pair_cache)
    return _final_gather(camera_first_segments, H, W), len(pair_cache.prop_i)


def parse_floats(items):
    return [float(s) for s in items]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=DEFAULT_REF)
    ap.add_argument("--width", type=int, default=50, help="image size — keep small")
    ap.add_argument("--n-iter", type=int, default=16, help="averaging iters PER r")
    ap.add_argument("--N", type=int, default=8, help="num_prop_iterations")
    ap.add_argument("--c", type=int, default=30, help="cluster_c (used only for voxel_size_0)")
    ap.add_argument("--r-factors", nargs="+", default=["4.0", "2.0", "1.0", "0.5", "0.25"],
                    help="r/voxel_size_0 multipliers to sweep, large → small")
    ap.add_argument("--geom-clamp", default="200",
                    help="geom_term cap factor (× median). 'none' = off. "
                         "Default 200 matches your F/H/G sweep settings so this "
                         "diagnostic measures kernel/PDF bias rather than the "
                         "firefly tail you'd already characterized in H.")
    ap.add_argument("--mmis-clamp", default="none",
                    help="mmis_weight cap factor (× median). 'none' = off.")
    ap.add_argument("--save-previews", action="store_true",
                    help="also save a tonemapped PNG per r (handy for spotting "
                         "splotches that don't show in scalar metrics).")
    args = ap.parse_args()

    geom_f = parse_clamp(args.geom_clamp)
    mmis_f = parse_clamp(args.mmis_clamp)
    print(f"clamps: geom={geom_f}  mmis={mmis_f}")

    W = H = args.width
    r_factors = parse_floats(args.r_factors)

    # Filename suffix used for every artifact saved by this run.
    rf_tag = f"rf{min(r_factors):g}-{max(r_factors):g}" if len(r_factors) > 1 else f"rf{r_factors[0]:g}"
    suffix_parts = [
        f"w{W}", f"n{args.n_iter}", f"N{args.N}", f"c{args.c}",
        rf_tag, f"g{args.geom_clamp}",
    ]
    if args.mmis_clamp != "none":
        suffix_parts.append(f"m{args.mmis_clamp}")
    SETTINGS_SUFFIX = "_".join(suffix_parts)

    ref = load_ref_downsampled(args.ref, W, H)
    print(f"ref (downsampled to {H}×{W}):  mean Σ-RGB = {ref.sum(-1).mean():.4e}, "
          f"median = {float(np.median(ref.sum(-1))):.4e}")

    scene = cornell_scene(W, H)
    sensor = scene.sensors()[0]

    # ── Compute voxel_size_0 for the r reference scale ───────────────────────
    cluster_probe, _, _, samplers_probe = _make_blt_components(
        scene, add_light_samples=False, num_prop_iterations=args.N, cluster_c=args.c,
        geom_clamp_factor=geom_f, mmis_clamp_factor=mmis_f,
    )
    s = mi.load_dict({"type": "independent", "sample_count": 1})
    s.seed(0, W * H)
    probe_pool, _ = _sample_segments(scene, sensor, s, H, W, 0.1, False)
    cluster_probe.set_segments(probe_pool.paths)
    cluster_probe.cluster(np.random.default_rng(seed=0))
    voxel_size_0 = float(cluster_probe.voxel_size)
    print(f"voxel_size_0 = {voxel_size_0:.4e}\n")

    # Shared tonemap exposure for any preview PNGs — computed from the ref so
    # every r-preview lives on the same display scale (and is directly
    # comparable to F's BLT-vs-ref output).
    shared_exposure = auto_exposure(ref, target_grey=0.18)
    print(f"shared tonemap exposure (ref median) = {shared_exposure:.4e}\n")

    # ── Sweep r ──────────────────────────────────────────────────────────────
    results = []  # list of (r, r_factor, rmse, mean_ratio, p99, pair_count, image)
    for rf in r_factors:
        r = rf * voxel_size_0
        print(f"── r = {r:.4e}  (r/voxel = {rf}) ──")

        cluster, mmis, propagation, samplers = _make_blt_components(
            scene, add_light_samples=False,
            num_prop_iterations=args.N, cluster_c=args.c,
            geom_clamp_factor=geom_f, mmis_clamp_factor=mmis_f,
        )

        accum = np.zeros((H, W, 3), dtype=np.float64)
        n_pairs_list = []
        t0 = time.time()
        for i in range(args.n_iter):
            sampler_i = mi.load_dict({"type": "independent", "sample_count": 1})
            sampler_i.seed(i, W * H)
            img, n_pairs = render_one_iter_exact(
                scene, sensor, sampler_i, H, W,
                cluster, mmis, propagation, samplers,
                r=r, iteration=i,
            )
            accum += img.astype(np.float64)
            n_pairs_list.append(n_pairs)
        elapsed = time.time() - t0

        running = accum / args.n_iter
        rmse = float(np.sqrt(((running - ref) ** 2).mean()))
        mr = float(running.sum(-1).mean() / max(ref.sum(-1).mean(), 1e-12))
        p99 = float(np.percentile(running.sum(-1), 99))
        avg_pairs = float(np.mean(n_pairs_list))

        results.append((r, rf, rmse, mr, p99, avg_pairs, running))
        print(f"   RMSE = {rmse:.4e}   mean_ratio = {mr:.4f}   "
              f"p99 = {p99:.3e}   avg_pairs = {avg_pairs:,.0f}   "
              f"({elapsed:.1f}s)")

        if args.save_previews:
            preview = reinhard_tonemap(running, shared_exposure)
            # The per-r preview filename pins the exact r value AND all sweep
            # settings, so two L runs with different N/clamp don't overwrite
            # each other's previews.
            preview_path = os.path.join(
                os.path.dirname(__file__),
                f"L_r_sweep_preview_thisrf{rf:g}_{SETTINGS_SUFFIX}.png",
            )
            plt.imsave(preview_path, preview)
            print(f"   preview: {preview_path}")

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n── summary ────────────────────────────────────────────────────")
    print(f"  {'r':>10s} {'r/vox':>7s} {'RMSE':>11s} {'mean_ratio':>11s} "
          f"{'p99':>10s} {'pairs':>10s}")
    for r, rf, rmse, mr, p99, np_, _img in results:
        print(f"  {r:>10.3e} {rf:>7.2f} {rmse:>11.3e} {mr:>11.4f} "
              f"{p99:>10.3e} {np_:>10,.0f}")

    # ── Bias-floor estimate ──────────────────────────────────────────────────
    # Fit RMSE² = a · r^(2β) + b² · (variance_constant_at_fixed_N)
    # Cheap proxy: take the minimum RMSE across the sweep — that's the floor of
    # the U.  If it's >> what variance alone would give, it's the bias floor.
    rmses = np.array([r[2] for r in results])
    mrs = np.array([r[3] for r in results])
    r_arr = np.array([r[0] for r in results])

    min_rmse = float(rmses.min())
    min_idx = int(np.argmin(rmses))
    print(f"\n  minimum RMSE = {min_rmse:.4e} at r = {r_arr[min_idx]:.4e} "
          f"(r/voxel = {results[min_idx][1]})")

    # Estimate the bias floor from the LEFT side of the U (small r, large N).
    # If we had infinite N: var → 0 and RMSE → bias. Our finite-N variance is
    # roughly constant in N at fixed r, scaling as N^{-1/2}. So a quick lower
    # bound on the bias is: min(RMSE) corrected by removing the expected
    # variance contribution. We can't do this rigorously without more N
    # samples, but we can flag the warning sign:
    if mrs.min() < 0.85 or mrs.max() > 1.15:
        print(f"  ⚠ mean_ratio range [{mrs.min():.3f}, {mrs.max():.3f}] — "
              f"some r values are far from 1.0, suggesting bias.")
    if (rmses[1:] / rmses[:-1]).min() > 0.7:  # RMSE not falling fast as r shrinks
        print(f"  ⚠ RMSE plateaus across r values — possible bias floor.")
        print(f"     Run with more --n-iter and smaller --r-factors to confirm.")

    # ── Plots ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.loglog([r[0] for r in results], rmses, "o-", label=f"RMSE (N={args.n_iter})")
    ax.set_xlabel("r  (exact-radius)")
    ax.set_ylabel("RMSE vs ref")
    ax.set_title(f"RMSE vs r — U-shape if bias and variance both matter")
    ax.grid(True, which="both", alpha=0.3)
    ax.axhline(min_rmse, color="k", linestyle="--", alpha=0.5,
               label=f"min observed = {min_rmse:.2e}")
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.plot([r[0] for r in results], mrs, "o-")
    ax.axhline(1.0, color="k", linestyle="--", alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("r  (exact-radius)")
    ax.set_ylabel("mean_ratio (BLT / ref)")
    ax.set_title("mean_ratio vs r — heads to 1.0 if unbiased")
    ax.set_ylim(0.5, 1.5)
    ax.grid(True, alpha=0.3)

    out = os.path.join(os.path.dirname(__file__), f"L_r_sweep_{SETTINGS_SUFFIX}.png")
    plt.tight_layout()
    plt.savefig(out, dpi=110)
    plt.close()
    print(f"\nsaved {out}")

    # ── Interpretation guide ─────────────────────────────────────────────────
    print("\n── How to read the L plot ──")
    print("  Left panel — RMSE vs r (log-log):")
    print("    • Clear U-shape, minimum at some r* > 0:  HEALTHY.")
    print("      Right side (large r) ~ r^β:  pure bias from kernel width.")
    print("      Left side (small r) growing: pure variance — N^(-1/2) r^(-d/2).")
    print("    • Floor on the left side that doesn't drop with more --n-iter:")
    print("      BIAS BUG — weights or PDFs are wrong, not a sample-count problem.")
    print("    • Monotone descent all the way to smallest r: no bias yet visible,")
    print("      variance not yet dominant. Push smaller r-factors or larger N.")
    print()
    print("  Right panel — mean_ratio vs r:")
    print("    • Heads to 1.0 as r shrinks: unbiased in the limit.")
    print("    • Plateaus at ratio != 1.0: bias floor in the integrand.")
    print()
    print("  If L shows a bias floor: the lever is the MMIS denominator")
    print("  (mmis.py:compute_all_mmis_weights_vec) and the conditional PDF math")
    print("  (sampler_types.py:conditional_pdf — especially the n_y/(distance²)")
    print("  area-measure conversion). Sub-stochastic mmis_w → energy deficit.")


if __name__ == "__main__":
    main()
