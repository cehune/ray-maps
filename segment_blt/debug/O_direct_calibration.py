"""
O. Direct-light calibration — ONE application of the gather operator.

The flat overshoot (~+3-4% at converged N) is either:
  (a) per-application: kernel/MMIS geometry inflates every gather, or
  (b) injection: Le sources enter at the wrong scale.
A single application isolates this. We compare:

  BLT direct at first hit:  pixel = T0 · D(v1)
      D = merged gather over LIGHT-TERMINAL sources only (direct_in),
      i.e. injection + one gather + MMIS weights, zero multi-hop.
  Ground truth direct:      mitsuba path integrator, max_depth=2
      (emitter visible + one bounce = direct lighting), high spp.

Verdict:
  ratio_direct ~= 1.00      -> injection & single gather calibrated;
                               the excess accumulates in MULTI-hop transport
                               (kernel false-positive geometry is prime suspect)
  ratio_direct ~= 1.03-1.04 -> the gather itself is hot; every hop inherits
                               it; look at MMIS weights / kernel value.

Run:  python O_direct_calibration.py --iters 16
"""
from _setup import cornell_scene
from _cache import RenderCache, pt_reference

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import mitsuba as mi

from segments.renderer import _make_blt_components, _sample_segments
from segments.pair_cache import build_pair_cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=16)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--ref-spp", type=int, default=512)
    ap.add_argument("--refresh", action="store_true",
                    help="ignore cached iterations / reference and re-render")
    args = ap.parse_args()

    W = H = args.width

    # ── ground-truth direct: mitsuba path, max_depth=2 (cached) ──────────────
    ref = pt_reference(W, H, max_depth=2, spp=args.ref_spp, refresh=args.refresh)
    print(f"  ref direct mean = {ref.sum(-1).mean():.4e}")

    # ── BLT: injection + ONE gather (direct_in), averaged over iterations ────
    scene = cornell_scene(W, H)
    sensor = scene.sensors()[0]
    cluster, mmis, propagation, samplers = _make_blt_components(
        scene, add_light_samples=False, num_prop_iterations=1,
        cluster_c=30, geom_clamp_factor=None, mmis_clamp_factor=None,
    )

    cache = RenderCache(
        params=dict(W=W, H=H, c=30, N=1, beta=0.1, light=False, nee=True,
                    mode="O_direct_readout"),
        tag="O_direct", enabled=not args.refresh)

    accum = np.zeros((H, W, 3), dtype=np.float64)
    for i in range(args.iters):
        cached = cache.load_iter(i)
        if cached is not None:
            accum += cached
            running = accum / (i + 1)
            mr = running.sum(-1).mean() / max(ref.sum(-1).mean(), 1e-12)
            print(f"iter {i+1:3d}/{args.iters}   direct mean_ratio = {mr:.4f}")
            continue
        sampler = mi.load_dict({"type": "independent", "sample_count": 1})
        sampler.seed(i, W * H)
        pool, cam_first = _sample_segments(scene, sensor, sampler, H, W,
                                           beta=0.1, add_light_samples=False)
        cluster.set_segments(pool.paths)
        cluster.cluster(np.random.default_rng(seed=i))
        pair_cache = build_pair_cache(
            cluster, samplers,
            merge_min_len_factor=propagation.merge_min_len_factor)
        mmis.compute_all_mmis_weights_vec(cluster, pair_cache)
        propagation.iterate_propogation_vec(cluster, pair_cache)  # fills direct_in

        img = np.zeros((H * W, 3))
        for px, fs in enumerate(cam_first):
            if fs is None:
                continue
            if fs.y.is_light:                       # emitter directly visible
                v = fs.throughput * fs.radiance_in  # pinned Le
            else:
                D = getattr(fs, "direct_in", None)
                if D is None:
                    continue
                v = fs.throughput * D
            img[px] = [float(v.x), float(v.y), float(v.z)]
        it_img = img.reshape(H, W, 3)
        cache.save_iter(i, it_img, stats={"mean": float(it_img.mean())})
        accum += it_img

        running = accum / (i + 1)
        mr = running.sum(-1).mean() / max(ref.sum(-1).mean(), 1e-12)
        print(f"iter {i+1:3d}/{args.iters}   direct mean_ratio = {mr:.4f}")

    # side-by-side
    blt = accum / args.iters
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for a, im, t in zip(ax, [ref, blt, blt - ref],
                        ["ref direct (PT d=2)", "BLT direct (1 gather)", "diff"]):
        if t == "diff":
            m = a.imshow(im.sum(-1), cmap="RdBu_r", vmin=-0.2, vmax=0.2)
            plt.colorbar(m, ax=a, fraction=0.046)
        else:
            a.imshow(np.clip(im ** (1 / 2.2), 0, 1))
        a.set_title(t); a.axis("off")
    out = os.path.join(os.path.dirname(__file__), "O_direct_calibration.png")
    plt.tight_layout(); plt.savefig(out, dpi=110); plt.close()
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
