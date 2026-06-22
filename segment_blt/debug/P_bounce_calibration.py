"""
P. Bounce-resolved calibration — localize the +3.4% by bounce order.

Direct (one gather) is calibrated at 0.993 (script O). The excess is in
indirect and is c-independent (G c-sweep). This script compares BLT's
cumulative energy per bounce order against a Mitsuba path-tracer depth
ladder rendered fresh, so nothing depends on the saved reference.

Mapping (split readout, camera-only):
  pixel(N=0) = emitter-visible + D(v1)            <-> mitsuba max_depth=2
  pixel(N)   adds one more gather of indirect      <-> mitsuba max_depth=N+2

Output: a table of mean-energy ratios BLT(N) / PT(max_depth=N+2).
  ratio flat ~1.00 then jumping at one N  -> that bounce order is broken
  ratio creeping up by ~x% per N          -> per-hop inflation of ~x%
  all ~1.00 but full-depth hot            -> the tail / pinning is wrong

Run:  python P_bounce_calibration.py --iters 8 --width 64
"""
from _setup import cornell_scene

import argparse
import numpy as np
import mitsuba as mi

from _cache import RenderCache, pt_reference
from segments.renderer import _make_blt_components, _sample_segments, _final_gather
from segments.pair_cache import build_pair_cache

N_LIST = [0, 1, 2, 3, 4, 8]


def pt_ladder_mean(W, H, depths, spp, refresh=False):
    means = {}
    for d_ in depths:
        img = pt_reference(W, H, max_depth=d_, spp=spp, refresh=refresh)
        means[d_] = float(img.sum(-1).mean())
        print(f"  PT max_depth={d_:3d}: mean={means[d_]:.4e}")
    return means


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--ref-spp", type=int, default=256)
    ap.add_argument("--refresh", action="store_true",
                    help="ignore cached iterations / references and re-render")
    args = ap.parse_args()
    W = H = args.width

    print("PT depth ladder...")
    depths = [n + 2 for n in N_LIST] + [-1]
    pt = pt_ladder_mean(W, H, depths, args.ref_spp, refresh=args.refresh)

    scene = cornell_scene(W, H)
    sensor = scene.sensors()[0]

    print("\nBLT N ladder (split readout)...")
    blt_mean = {}
    for N in N_LIST:
        cluster, mmis, propagation, samplers = _make_blt_components(
            scene, add_light_samples=False, num_prop_iterations=N,
            cluster_c=30, geom_clamp_factor=None, mmis_clamp_factor=None,
        )
        cache = RenderCache(
            params=dict(W=W, H=H, c=30, N=N, beta=0.1, light=False, nee=True,
                        fg_depth=2, mode="P_bounce_readout"),
            tag=f"P_blt_N{N}", enabled=not args.refresh)
        acc = 0.0
        for i in range(args.iters):
            cached = cache.load_iter(i)
            if cached is not None:
                acc += float(np.asarray(cached).sum(-1).mean())
                continue
            sampler = mi.load_dict({"type": "independent", "sample_count": 1})
            sampler.seed(i, W * H)
            pool, cam_first = _sample_segments(scene, sensor, sampler, H, W,
                                               beta=0.1, add_light_samples=False)
            cluster.set_segments(pool.paths)
            cluster.cluster(np.random.default_rng(seed=i))
            pc = build_pair_cache(
                cluster, samplers,
                merge_min_len_factor=propagation.merge_min_len_factor)
            mmis.compute_all_mmis_weights_vec(cluster, pc)
            propagation.iterate_propogation_vec(cluster, pc)
            img = np.asarray(_final_gather(cam_first, H, W, depth=2))
            cache.save_iter(i, img, stats={"mean": float(img.mean())})
            acc += float(img.sum(-1).mean())
        blt_mean[N] = acc / args.iters
        print(f"  BLT N={N}: mean={blt_mean[N]:.4e}")

    print("\n── cumulative calibration: BLT(N) vs PT(max_depth=N+2) ──")
    prev_b = prev_p = None
    for N in N_LIST:
        b, p = blt_mean[N], pt[N + 2]
        line = f"  N={N:2d} (≤{N+2}-vertex paths): BLT/PT = {b/p:.4f}"
        if prev_b is not None and p - prev_p > 1e-9:
            line += f"   marginal bounce ratio = {(b-prev_b)/(p-prev_p):.4f}"
        print(line)
        prev_b, prev_p = b, p
    print(f"  full transport:  BLT(N=8)/PT(inf) = {blt_mean[8]/pt[-1]:.4f}")
    print("\n  'marginal bounce ratio' is the calibration of EACH added bounce —")
    print("  the first N where it leaves ~1.0 is where the excess enters.")


if __name__ == "__main__":
    main()
