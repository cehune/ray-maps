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
from _cache import save_render, load_render

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import mitsuba as mi
from segments.renderer import Renderer, _make_blt_components


DEFAULT_REF = "/Users/celine/Documents/projects/ray-maps/baseline/spp_renders-128x128-d-1/reference.exr"


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
            geom_clamp=200.0, geom_mode="hard", geom_growth=0.0):
    """Run BLT for one alpha value, return per-iter (rmse, mean_ratio, voxel_size)."""
    cluster, mmis, propagation, samplers = _make_blt_components(
        scene, add_light_samples=False,
        num_prop_iterations=N, cluster_c=c, alpha=alpha,
        geom_clamp_factor=geom_clamp, geom_clamp_mode=geom_mode,
        geom_clamp_growth=geom_growth,
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
    ap.add_argument("--geom-clamp-growth", type=float, default=0.0,
                    help="progressive clamp relaxation: cap_n = geom_clamp · n**growth "
                         "· median(geom). 0 = fixed clamp (old behavior). Try 0.5–1.0 to "
                         "let the cap grow each iter so the clamp's ~3%% darkening floor "
                         "vanishes in the running average.")
    ap.add_argument("--refresh", action="store_true",
                    help="ignore cached EXR/metrics and re-render every alpha")
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
        # Cache key encodes everything that changes the IMAGE (not the ref).
        # growth suffix only appended when nonzero, so existing g0 caches stay valid.
        tag = (f"J_w{W}_i{args.max_iter}_c{args.c}_N{args.N}"
               f"_a{alpha}_g{args.geom_clamp}"
               + (f"_gg{args.geom_clamp_growth}" if args.geom_clamp_growth else ""))
        hit = None if args.refresh else load_render(tag)
        if hit is not None and hit["meta"] is not None:
            m = hit["meta"]
            rmses, mrs, vs, img = m["rmses"], m["mrs"], m["voxel_sizes"], hit["image"]
            print(f"   (cache hit: {tag})")
            cached_ref = m.get("ref")
            if cached_ref != os.path.basename(args.ref):
                print(f"   ⚠ cached metrics computed vs '{cached_ref}', "
                      f"current ref is '{os.path.basename(args.ref)}' — "
                      f"pass --refresh to recompute against it")
        else:
            rmses, mrs, vs, img = run_one(
                scene, sensor, renderer, H, W,
                args.max_iter, args.c, args.N, alpha, ref,
                geom_clamp=args.geom_clamp, geom_growth=args.geom_clamp_growth,
            )
            save_render(tag, img, meta={
                "config": {"W": W, "H": H, "max_iter": args.max_iter,
                           "c": args.c, "N": args.N, "alpha": alpha,
                           "geom_clamp": args.geom_clamp,
                           "geom_clamp_growth": args.geom_clamp_growth,
                           "beta": 0.1, "lookup": "voxel"},
                "ref": os.path.basename(args.ref),
                "rmses": rmses, "mrs": mrs, "voxel_sizes": vs,
            })
            print(f"   (saved cache: {tag}.exr / .json)")
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

    # ── row 0, left: convergence with bias/variance decomposition ────
    # A running average converges as  RMSE(N)² = bias² + variance/N.
    #   • variance/N  → falls as 1/N  (slope -1/2 on log-log)
    #   • bias²       → CONSTANT      (horizontal asymptote)
    # so RMSE flattens onto a floor = |bias|; it does NOT reach zero. A single
    # power-law fit over the whole curve is meaningless here — it blends the
    # early variance-limited slope with the late bias-limited flat. Instead we
    # fit RMSE² = a + b/N (linear in 1/N): a = floor², b = variance scale, and
    # overlay the fitted model (dashed) + the floor (dotted) so you can read
    # off what it converges TO and whether variance is still falling.
    ax = axes[0, 0]
    print("\n── convergence fit (RMSE² = floor² + var/N) ─────────")
    for alpha, (rmses_a, _mrs, _vs, _) in results.items():
        r = np.asarray(rmses_a, dtype=np.float64)
        line, = ax.loglog(iters, r, "o-", label=f"α={alpha}")
        col = line.get_color()

        b, a = np.polyfit(1.0 / iters, r ** 2, 1)          # RMSE² vs 1/N
        floor = float(np.sqrt(max(a, 0.0)))
        model = np.sqrt(np.maximum(a + b / iters, 0.0))
        ax.loglog(iters, model, "--", color=col, alpha=0.6)
        if floor > 0:
            ax.axhline(floor, color=col, ls=":", alpha=0.5)

        # asymptotic slope = fit only the back half (drops the early transient)
        h = max(len(iters) // 2, 2)
        tail_slope = np.polyfit(np.log(iters[-h:]), np.log(r[-h:]), 1)[0]
        full_slope = np.polyfit(np.log(iters), np.log(r), 1)[0]
        # extrapolated RMSE if variance vanished entirely (N→∞)
        print(f"  α={alpha:<5}: final={r[-1]:.4f}  floor≈{floor:.4f}  "
              f"({100*floor/max(r[-1],1e-9):.0f}% of final is irreducible)  "
              f"tail-slope={tail_slope:+.2f}  full-slope={full_slope:+.2f}")

    r0 = float(np.asarray(next(iter(results.values()))[0])[0])
    ax.loglog(iters, r0 * iters ** (-0.5), "k--", alpha=0.4, label="N^(-1/2) MC")
    ax.set_xlabel("iterations"); ax.set_ylabel("RMSE")
    ax.set_title("convergence — dashed=fit, dotted=bias floor")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)

    # grid is always ≥2 cols (max(n_alpha, 2)), so axes[0, 1] always exists —
    # do NOT fall back to axes[0, 0] for n_alpha==1 or the mean-ratio plot
    # overdraws the convergence chart and the top-right panel renders blank.
    ax = axes[0, 1]
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
    out = os.path.join(os.path.dirname(__file__), f"J_alpha_sweep_w{args.width}_a{args.alphas}_geom_clamp{args.geom_clamp}_geom_growth{args.geom_clamp_growth}.png")
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
