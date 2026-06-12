"""
O3: bucket the depth-99 split readout by level and calibrate each bucket.

The contradiction (post-NEE, one code epoch):
    O  : T0·D(v1)        vs PT(2)          ~ 0.99   cold
    O2 : T0·T1·D(v2)     vs PT(3)-PT(2)    ~ 0.956  cold
    G  : fg_depth=99 readout (= sum of all such terms + tail)  = 1.072 HOT

A sum of cold terms cannot be hot, so either a deep bucket is wildly hot or
one instrument measures something other than what it claims. This script
removes all interpretation:

  * walks the EXACT _readout_path(d=99) recursion per pixel, but accumulates
    each level's contribution into its own image bucket:
        bucket 0     : emitter directly visible (T0·Le)
        bucket ℓ>=1  : P_{ℓ-1} · D(v_ℓ)   (prefix throughput × cached direct)
        bucket 'tail': P · rad_in(s_last) for paths cut by max_depth
  * asserts Σ buckets == _readout_path(path, 99) per pixel (exactness check —
    if this trips, the bucketing or the readout has a bug, fix before
    trusting anything);
  * compares cumulative sums against the cached PT depth ladder:
        Σ_{ℓ<=L} bucket_ℓ  vs  PT(max_depth=L+1)
    and prints per-level ratios bucket_ℓ / (PT(L+1)-PT(L)).

Whichever level's ratio leaves ~1.0 is where the d=99 heat enters.

Run:  python O3_readout_ladder.py --width 64 --iters 16
      (use the same width as your O/O2 runs for apples-to-apples)
"""
from _setup import cornell_scene
from _cache import RenderCache, pt_reference

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import mitsuba as mi

from segments.renderer import (_make_blt_components, _sample_segments,
                               _readout_path)
from segments.pair_cache import build_pair_cache

MAX_LEVEL = 6   # buckets 0..MAX_LEVEL, deeper levels fold into 'tail+'


def bucket_pixel(fs, max_level):
    """Mirror _readout_path(path, 99) but return per-level contributions.
    Returns (levels[0..max_level], deep) as float arrays (3,), where `deep`
    collects everything past max_level plus max-depth tails."""
    levels = np.zeros((max_level + 1, 3))
    deep = np.zeros(3)
    path = getattr(fs, "_path", None)
    if path is None:
        return levels, deep

    def c3(v):
        return np.array([float(v.x), float(v.y), float(v.z)])

    s1 = path[0]
    if s1.y.is_light or s1.x.is_light:
        levels[0] = c3(s1.throughput * s1.radiance_in)   # pinned Le
        return levels, deep

    P = c3(s1.throughput)                  # prefix throughput T0
    k = 0                                  # index into path; s_{k+1} = path[k]
    while True:
        seg = path[k]                      # gather vertex v_{k+1}
        D = getattr(seg, "direct_in", None)
        if D is None:
            break
        contrib = P * c3(D)
        if k + 1 <= max_level:
            levels[k + 1] = contrib
        else:
            deep += contrib
        nxt = path[k + 1] if k + 1 < len(path) else None
        if nxt is None:
            # path cut by max_depth: the d=99 readout reads the full cache
            # at the last segment MINUS its direct (already bucketed above)
            tail = P * (c3(seg.radiance_in) - c3(D))
            deep += tail
            break
        if nxt.y.is_light or nxt.x.is_light:
            break                          # own emitter hit lives inside D
        P = P * c3(nxt.throughput)
        k += 1
    return levels, deep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=16)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--ref-spp", type=int, default=512)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    W = H = args.width

    # PT cumulative ladder: PT(max_depth=L+1) covers buckets 0..L
    pt_depths = list(range(1, MAX_LEVEL + 2)) + [-1]
    pt = {d_: pt_reference(W, H, max_depth=d_, spp=args.ref_spp,
                           refresh=args.refresh).sum(-1).mean()
          for d_ in pt_depths}

    scene = cornell_scene(W, H)
    sensor = scene.sensors()[0]
    cluster, mmis, propagation, samplers = _make_blt_components(
        scene, add_light_samples=False, num_prop_iterations=8,
        cluster_c=30, geom_clamp_factor=None, mmis_clamp_factor=None,
    )

    lvl_mean = np.zeros(MAX_LEVEL + 1)
    deep_mean = 0.0
    total_mean = 0.0
    worst_mismatch = 0.0

    for i in range(args.iters):
        sampler = mi.load_dict({"type": "independent", "sample_count": 1})
        sampler.seed(i, W * H)
        pool, cam_first = _sample_segments(scene, sensor, sampler, H, W,
                                           beta=0.1, add_light_samples=False)
        cluster.set_segments(pool.paths)
        cluster.cluster(np.random.default_rng(seed=i))
        pc = build_pair_cache(cluster, samplers,
                              merge_min_len_factor=propagation.merge_min_len_factor)
        mmis.compute_all_mmis_weights_vec(cluster, pc)
        propagation.iterate_propogation_vec(cluster, pc)

        for fs in cam_first:
            if fs is None:
                continue
            levels, deep = bucket_pixel(fs, MAX_LEVEL)
            # exactness: buckets must reproduce the production d=99 readout
            ref_val = fs.throughput * _readout_path(fs._path, 99) \
                if getattr(fs, "_path", None) else None
            if ref_val is not None:
                rv = np.array([float(ref_val.x), float(ref_val.y), float(ref_val.z)])
                tot = levels.sum(axis=0) + deep
                worst_mismatch = max(worst_mismatch,
                                     float(np.abs(tot - rv).max()))
            # channel SUM, matching the PT slices' .sum(-1).mean()
            # (an earlier version divided by 3 here — channel mean vs
            # channel sum — which scaled every printed ratio by exactly 1/3)
            lvl_mean += levels.sum(axis=1)
            deep_mean += float(deep.sum())
            total_mean += float(levels.sum() + deep.sum())
        print(f"iter {i+1}/{args.iters} done   (bucket-vs-readout max "
              f"mismatch so far: {worst_mismatch:.2e})")

    n = args.iters * W * H
    lvl_mean /= n; deep_mean /= n; total_mean /= n

    # Color3f is float32: products of ~16 terms accumulate ~1e-6 abs error.
    # Real bucketing bugs are O(0.1) — six orders above this tolerance.
    assert worst_mismatch < 1e-4, (
        f"buckets do not reproduce _readout_path (max err {worst_mismatch:.3e})"
        " — fix the bucketing before reading the table!")

    print("\n── d=99 readout, calibrated per level ──")
    print(f"{'lvl':>4} | {'BLT bucket':>11} | {'PT slice':>11} | {'ratio':>6}")
    cum_blt = 0.0
    for L in range(MAX_LEVEL + 1):
        pt_slice = pt[L + 1] - (pt[L] if L >= 1 else 0.0)
        cum_blt += lvl_mean[L]
        r = lvl_mean[L] / pt_slice if pt_slice > 1e-12 else float('nan')
        cum_r = cum_blt / pt[L + 1]
        print(f"{L:>4} | {lvl_mean[L]:>11.4e} | {pt_slice:>11.4e} | {r:>6.3f}"
              f"   (cumulative vs PT({L+1}): {cum_r:.4f})")
    deep_truth = pt[-1] - pt[MAX_LEVEL + 1]
    print(f"deep+tail | {deep_mean:>9.4e} | {deep_truth:>11.4e} | "
          f"{deep_mean/max(deep_truth,1e-12):>6.3f}")
    print(f"\nTOTAL vs PT(inf): {total_mean/pt[-1]:.4f}   "
          f"(G's fg99 mean_ratio should match this)")


if __name__ == "__main__":
    main()
