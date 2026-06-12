"""
Q2: Disjoint hop-pools — the corrected reuse-compounding test.

Q's flaw: arm B re-jittered the CLUSTERING per hop but reused the same
SEGMENTS, so correlation flowing through the shared sample pool
(E[M_1·M_2·...·Le] > E[M]^k·Le when the M_k share samples) was present in
both arms and invisible. The scale-free ~1.3x marginal bounce heat (same at
64^2 and 128^2 voxel scales, per O/P) is exactly the signature such
compounding would leave: (1 + relvar)^k, independent of kernel size.

Here arm B gives every hop a genuinely independent operator:
  - paths are split round-robin into k disjoint pools (BY PATH, so each
    child's parent is in the same pool and its pdf term stays in p_sum —
    the w*G <= pi bound survives);
  - hop m builds its pair cache with pool_mask = pool m: merge sources and
    MMIS auxiliaries restricted to pool m, receivers unrestricted. The
    denominator then counts exactly the techniques active in the sub-pool,
    so weights scale up by ~k automatically and each M_m is an unbiased
    estimate of the full one-hop operator from 1/k of the samples.

Arms (same sampled pool per iteration — paired):
  A "production"  one full-pool cache, k hops (M applied k times).
  B "hop-pools"   independent M_1..M_k, one hop each.

Reading the table:
  A/B ratio grows ~constant factor per hop  -> same-pool compounding
      confirmed; production fix = hop-pool decorrelation (or keep fg-depth
      high so the image reads only deep cache bounces).
  A/B stays ~= 1                            -> compounding refuted for real
      this time; the heat is a deterministic per-hop calibration error —
      hunt for a constant factor in the gather math.

Note B is noisier per hop (1/k sources per gather); compare MEANS over
iterations, and run with --iters >= 4 before trusting a small difference.

Usage:
    python debug/Q2_hop_pools.py --wh 32 --iters 4 --hops 4 --c 60
(--c is the FULL-pool occupancy; each hop-pool sees ~c/hops of it.)
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _setup import cornell_scene  # noqa: E402

import mitsuba as mi  # noqa: E402

from segments.cluster import Cluster                           # noqa: E402
from segments.mmis import MMIS                                 # noqa: E402
from segments.pair_cache import build_pair_cache               # noqa: E402
from segments.sampler_types import Sampler, SequentialSampler  # noqa: E402
from segments.segment_gen_sampler import sample_camera_path    # noqa: E402

LUM = np.array([0.2126, 0.7152, 0.0722])


def sample_pool(scene, W, H, seed, max_depth, n_pools):
    """Returns segments, camera-first indices+segs, and per-segment pool id
    (round-robin by PATH so parent/child always share a pool)."""
    sensor = scene.sensors()[0]
    sampler = mi.load_dict({"type": "independent", "sample_count": 1})
    sampler.seed(seed, W * H)
    segments, camera_first, pool_id = [], [], []
    path_count = 0
    for y in range(H):
        for x in range(W):
            pos = mi.Point2f(x + sampler.next_1d(), y + sampler.next_1d())
            ap = mi.Point2f(sampler.next_1d(), sampler.next_1d())
            film = mi.Point2f(pos.x / W, pos.y / H)
            ray, ray_w = sensor.sample_ray(
                time=0.0, sample1=sampler.next_1d(), sample2=film, sample3=ap)
            si = scene.ray_intersect(ray)
            path = sample_camera_path(scene, sampler, ray, ray_w, si,
                                      max_depth=max_depth)
            if path:
                camera_first.append((len(segments), path[0]))
                segments.extend(path)
                pool_id.extend([path_count % n_pools] * len(path))
                path_count += 1
    return segments, camera_first, np.array(pool_id, dtype=np.int32)


def build_operator(segments, cluster, samplers, merge_min_len_factor,
                   row_gain_cap, pool_mask=None):
    """Pair cache + MMIS + row-cap → (weighted_fr, cache). Clustering is the
    caller's (shared between arms so only pooling differs)."""
    cache = build_pair_cache(cluster, samplers,
                             merge_min_len_factor=merge_min_len_factor,
                             pool_mask=pool_mask)
    MMIS().compute_all_mmis_weights_vec(cluster, cache)
    mmis_w = np.array([float(s.mmis_weight) for s in segments])
    geom = np.array([float(s.geom_term) for s in segments])
    wfr = cache.prop_fr * (mmis_w[cache.prop_j] * geom[cache.prop_j])[:, None]
    if row_gain_cap is not None and len(cache.prop_i):
        row = np.zeros(len(segments))
        np.add.at(row, cache.prop_i, wfr @ LUM)
        over = row > row_gain_cap
        if over.any():
            scale = np.where(over, row_gain_cap / np.maximum(row, 1e-300), 1.0)
            wfr = wfr * scale[cache.prop_i][:, None]
            print(f"    [row cap] {int(over.sum())} rows capped "
                  f"(max {row.max():.2f})")
    return wfr, cache


def one_hop(rad_in, Le, is_light, wfr, cache):
    S = rad_in.shape[0]
    rad_out = np.zeros((S, 3))
    if len(cache.prop_i):
        np.add.at(rad_out, cache.prop_i, wfr * rad_in[cache.prop_j])
    if len(cache.cls_i):
        np.add.at(rad_out, cache.cls_i, cache.cls_w * rad_in[cache.cls_j])
    return np.where(is_light[:, None], Le, rad_out)


def run(args):
    scene = cornell_scene(args.wh, args.wh)
    samplers = Sampler.build_registry(SequentialSampler())
    k = args.hops

    pool_lum = {a: np.zeros(k) for a in "AB"}
    cam_lum = {a: np.zeros(k) for a in "AB"}

    for it in range(args.iters):
        segments, camera_first, pid = sample_pool(
            scene, args.wh, args.wh, seed=it, max_depth=args.max_depth,
            n_pools=k)
        S = len(segments)
        Le = np.array([[float(s.Le.x), float(s.Le.y), float(s.Le.z)]
                       for s in segments])
        is_light = np.array([s.y.is_light or s.x.is_light for s in segments],
                            dtype=bool)
        cam_idx = np.array([i for i, _ in camera_first], dtype=np.int64)
        T0 = np.array([[float(fs.throughput.x), float(fs.throughput.y),
                        float(fs.throughput.z)] for _, fs in camera_first])

        cluster = Cluster(c=args.c, alpha=0.25)
        cluster.set_scene_aabb(scene.bbox())
        cluster.set_segments(segments)
        cluster.cluster(np.random.default_rng(seed=1000 + it))
        print(f"[iter {it}] S={S}, voxel={float(cluster.voxel_size):.3e}")

        # Arm A: one full-pool operator, k hops
        wfr, cache = build_operator(segments, cluster, samplers,
                                    args.merge_min_len_factor,
                                    args.row_gain_cap)
        rad = Le.copy()
        for m in range(k):
            rad = one_hop(rad, Le, is_light, wfr, cache)
            pool_lum["A"][m] += float((rad[~is_light] @ LUM).sum())
            cam_lum["A"][m] += float(((T0 * rad[cam_idx]) @ LUM).mean())

        # Arm B: independent operator per hop from disjoint sub-pools
        # (same clustering as A — only the source/auxiliary pooling differs)
        ops = []
        for m in range(k):
            ops.append(build_operator(segments, cluster, samplers,
                                      args.merge_min_len_factor,
                                      args.row_gain_cap,
                                      pool_mask=(pid == m)))
        rad = Le.copy()
        for m in range(k):
            wfr_m, cache_m = ops[m]
            rad = one_hop(rad, Le, is_light, wfr_m, cache_m)
            pool_lum["B"][m] += float((rad[~is_light] @ LUM).sum())
            cam_lum["B"][m] += float(((T0 * rad[cam_idx]) @ LUM).mean())

    print(f"\n{'hop':>4} | {'pool A':>11} {'pool B':>11} {'A/B':>6} | "
          f"{'gain A':>7} {'gain B':>7} | {'cam A/B':>7}")
    print("-" * 72)
    def _r(a, b):
        return a / b if b > 0 else float('nan')
    for m in range(k):
        ga = _r(pool_lum['A'][m], pool_lum['A'][m-1]) if m else float('nan')
        gb = _r(pool_lum['B'][m], pool_lum['B'][m-1]) if m else float('nan')
        print(f"{m+1:>4} | {pool_lum['A'][m]:>11.4e} {pool_lum['B'][m]:>11.4e} "
              f"{_r(pool_lum['A'][m], pool_lum['B'][m]):>6.3f} | "
              f"{ga:>7.3f} {gb:>7.3f} | "
              f"{_r(cam_lum['A'][m], cam_lum['B'][m]):>7.3f}", flush=True)
    print("\nA/B growing by a ~constant factor per hop ⇒ same-pool hop "
          "compounding confirmed. A/B ≈ 1 throughout ⇒ deterministic per-hop "
          "calibration error; hunt a constant factor in the gather math.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wh", type=int, default=32)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--hops", type=int, default=4)
    ap.add_argument("--c", type=int, default=60)
    ap.add_argument("--max-depth", type=int, default=8)
    ap.add_argument("--merge-min-len-factor", type=float, default=1.0)
    ap.add_argument("--row-gain-cap", type=float, default=1.0)
    run(ap.parse_args())
