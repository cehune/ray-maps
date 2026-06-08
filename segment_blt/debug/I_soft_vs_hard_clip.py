"""
I. A/B hard vs soft geom-clip at the same clamp factor.

Renders the same scene twice with the same `geom_clamp_factor` but the
two clip MODES:
  • hard: geom = min(geom, cap)            ← brick wall, what you had
  • soft: geom = geom * cap / (cap + geom) ← Reinhard-style asymptote

Reports per-iter RMSE and mean_ratio for both, plus a side-by-side
tonemapped image at matched exposure.

Expected:
  • soft mean_ratio closer to 1.0 than hard (less energy lost to tail)
  • soft RMSE comparable or slightly higher than hard at low iter
    (it lets a bit more variance through), but converges to lower
    asymptote because the bias floor is closer to zero
  • visually nearly identical at this scale

Usage:
    python I_soft_vs_hard_clip.py --max-iter 8 --factor 200
    python I_soft_vs_hard_clip.py --max-iter 16 --factor 300 --width 128
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
    arr = np.array(mi.Bitmap(path))[..., :3].astype(np.float64)
    if arr.shape[:2] != (H, W):
        raise SystemExit(f"ref shape {arr.shape[:2]} ≠ {(H, W)}")
    return arr


def reinhard_tonemap(img, exposure, gamma=2.2):
    scaled = np.asarray(img, dtype=np.float64) * exposure
    compressed = scaled / (1.0 + scaled)
    return np.clip(compressed ** (1.0 / gamma), 0.0, 1.0).astype(np.float32)


def run(scene, sensor, renderer, H, W, max_iter, c, N, factor, mode, ref):
    cluster, mmis, propagation, samplers = _make_blt_components(
        scene, add_light_samples=False,
        num_prop_iterations=N, cluster_c=c,
        geom_clamp_factor=factor, geom_clamp_mode=mode,
    )
    accum = np.zeros((H, W, 3), dtype=np.float64)
    rmses, mrs = [], []
    last_img = None
    for i in range(max_iter):
        cluster._iteration = i  # progressive on
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
        last_img = running
    return rmses, mrs, last_img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=DEFAULT_REF)
    ap.add_argument("--max-iter", type=int, default=8)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--c", type=int, default=30)
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--factor", type=float, default=200.0,
                    help="geom_clamp_factor (× median geom). Default 200.")
    args = ap.parse_args()

    W = H = args.width
    ref = load_ref(args.ref, W, H)
    print(f"ref shape={ref.shape}  mean={ref.sum(-1).mean():.4e}")

    scene = cornell_scene(W, H)
    sensor = scene.sensors()[0]
    renderer = Renderer()

    results = {}
    for mode in ("hard", "soft"):
        print(f"\n── geom-clamp mode={mode}, factor={args.factor} ──")
        rmses, mrs, img = run(scene, sensor, renderer, H, W,
                              args.max_iter, args.c, args.N, args.factor, mode, ref)
        results[mode] = (rmses, mrs, img)
        print(f"   final RMSE={rmses[-1]:.4e}   final mean_ratio={mrs[-1]:.4f}")

    print("\n── summary ─────────────────────────────")
    print(f"  factor = {args.factor}")
    print(f"  hard:  RMSE={results['hard'][0][-1]:.4e}   mean_ratio={results['hard'][1][-1]:.4f}")
    print(f"  soft:  RMSE={results['soft'][0][-1]:.4e}   mean_ratio={results['soft'][1][-1]:.4f}")

    # ── plots ────────────────────────────────────────────────────
    iters = np.arange(1, args.max_iter + 1)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.loglog(iters, results["hard"][0], "o-", label="hard")
    ax.loglog(iters, results["soft"][0], "s-", label="soft")
    ax.set_xlabel("iterations"); ax.set_ylabel("RMSE"); ax.set_title("RMSE vs iter")
    ax.grid(True, which="both", alpha=0.3); ax.legend()

    ax = axes[0, 1]
    ax.plot(iters, results["hard"][1], "o-", label="hard")
    ax.plot(iters, results["soft"][1], "s-", label="soft")
    ax.axhline(1.0, color="k", linestyle="--", alpha=0.5)
    ax.set_xlabel("iterations"); ax.set_ylabel("mean(BLT)/mean(ref)")
    ax.set_title("mean ratio — closer to 1.0 = less bias")
    ax.set_ylim(0.85, 1.15); ax.grid(True, alpha=0.3); ax.legend()

    # tonemapped images at matched exposure (from ref median)
    ref_lum = ref.sum(-1)
    med = float(np.median(ref_lum[ref_lum > 0])) if (ref_lum > 0).any() else 0.0
    exposure = (0.18 / med) if med > 0 else 1.0

    axes[1, 0].imshow(reinhard_tonemap(results["hard"][2], exposure))
    axes[1, 0].set_title(f"hard  (mr={results['hard'][1][-1]:.3f})"); axes[1, 0].axis("off")
    axes[1, 1].imshow(reinhard_tonemap(results["soft"][2], exposure))
    axes[1, 1].set_title(f"soft  (mr={results['soft'][1][-1]:.3f})"); axes[1, 1].axis("off")

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__),
                       f"I_soft_vs_hard_factor{int(args.factor)}.png")
    plt.savefig(out, dpi=110); plt.close()
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
