"""
Q: Is the hot indirect (P's 1.26–1.59 marginal bounce ratios) hop-reuse
correlation, or per-hop calibration bias?

Mechanism under test: propagation applies the SAME operator M (one clustering,
one pair cache, one set of MMIS denominators) num_prop_iterations times.
E[M^k · Le] ≠ E[M]^k · Le when the hops share randomness — each hop re-uses
the same realized denominators and the same pair set, so per-hop estimation
error compounds multiplicatively instead of averaging. If that's the hot-
bounce mechanism, decorrelating the hops should pull the per-hop gains down.

Arms (same sampled segment pool per iteration — paired comparison):
  A "reused"      one clustering + one pair cache, k hops (production).
  B "rejittered"  fresh jitter → fresh clustering → fresh pair cache → fresh
                  MMIS denominators for EVERY hop; radiance is carried on the
                  segments across hops. Same kernel size, same pool — only the
                  hop-to-hop correlation is removed.

Read the table:
  gains equal            → reuse correlation is NOT the mechanism; the heat is
                           per-hop calibration (kernel/pdf/MMIS) — go look at
                           the density estimate itself.
  A gains > B gains,
  growing with hop count → reuse correlation confirmed; fix direction is
                           per-hop decorrelation (rejitter between hops, or
                           disjoint source sub-pools per hop), not weights.

Cost: arm B rebuilds the pair cache per hop (BSDF/pdf evals). Keep --wh small.

Usage:
    python debug/Q_reuse_correlation.py --wh 32 --iters 2 --hops 5 --c 30
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _setup import cornell_scene  # noqa: E402  (patches src onto sys.path)

import mitsuba as mi  # noqa: E402

from segments.cluster import Cluster                           # noqa: E402
from segments.mmis import MMIS                                 # noqa: E402
from segments.pair_cache import build_pair_cache               # noqa: E402
from segments.sampler_types import Sampler, SequentialSampler  # noqa: E402
from segments.segment_gen_sampler import sample_camera_path    # noqa: E402

LUM = np.array([0.2126, 0.7152, 0.0722])


def sample_pool(scene, W, H, seed, max_depth):
    sensor = scene.sensors()[0]
    sampler = mi.load_dict({"type": "independent", "sample_count": 1})
    sampler.seed(seed, W * H)
    segments, camera_first = [], []
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
    return segments, camera_first


def build_operator(segments, scene, samplers, jitter_seed, c,
                   merge_min_len_factor, row_gain_cap):
    """Cluster with the given jitter seed and return (weighted_fr, cache).
    Mirrors build_pair_cache + MMIS + the row-gain projection in
    iterate_propogation_vec."""
    cluster = Cluster(c=c, alpha=0.25)
    cluster.set_scene_aabb(scene.bbox())
    cluster.set_segments(segments)
    cluster.cluster(np.random.default_rng(seed=jitter_seed))
    cache = build_pair_cache(cluster, samplers,
                             merge_min_len_factor=merge_min_len_factor)
    MMIS().compute_all_mmis_weights_vec(cluster, cache)

    mmis_w = np.array([float(s.mmis_weight) for s in segments])
    geom = np.array([float(s.geom_term) for s in segments])
    wfr = cache.prop_fr * (mmis_w[cache.prop_j] * geom[cache.prop_j])[:, None]

    if row_gain_cap is not None and len(cache.prop_i):
        S = len(segments)
        row = np.zeros(S)
        np.add.at(row, cache.prop_i, wfr @ LUM)
        over = row > row_gain_cap
        if over.any():
            scale = np.where(over, row_gain_cap / np.maximum(row, 1e-300), 1.0)
            wfr = wfr * scale[cache.prop_i][:, None]
    return wfr, cache


def one_hop(rad_in, Le, is_light, wfr, cache):
    """One Jacobi hop: rad_out = M·rad_in (+ classical links), lights pinned."""
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

    # per (arm, hop): pool luminance + camera-first readout luminance
    pool_lum = {a: np.zeros(args.hops) for a in "AB"}
    cam_lum = {a: np.zeros(args.hops) for a in "AB"}

    for it in range(args.iters):
        segments, camera_first = sample_pool(
            scene, args.wh, args.wh, seed=it, max_depth=args.max_depth)
        S = len(segments)
        Le = np.array([[float(s.Le.x), float(s.Le.y), float(s.Le.z)]
                       for s in segments])
        is_light = np.array([s.y.is_light or s.x.is_light for s in segments],
                            dtype=bool)
        cam_idx = np.array([i for i, _ in camera_first], dtype=np.int64)
        T0 = np.array([[float(fs.throughput.x), float(fs.throughput.y),
                        float(fs.throughput.z)] for _, fs in camera_first])
        print(f"[iter {it}] S={S} segments")

        # Arm A: one operator, k hops
        wfr_A, cache_A = build_operator(
            segments, scene, samplers, jitter_seed=1000 + it, c=args.c,
            merge_min_len_factor=args.merge_min_len_factor,
            row_gain_cap=args.row_gain_cap)
        rad = Le.copy()
        for k in range(args.hops):
            rad = one_hop(rad, Le, is_light, wfr_A, cache_A)
            pool_lum["A"][k] += float((rad[~is_light] @ LUM).sum())
            cam_lum["A"][k] += float(((T0 * rad[cam_idx]) @ LUM).mean())

        # Arm B: fresh operator per hop (same pool, same kernel size)
        rad = Le.copy()
        for k in range(args.hops):
            wfr_B, cache_B = build_operator(
                segments, scene, samplers,
                jitter_seed=5000 + 100 * it + k, c=args.c,
                merge_min_len_factor=args.merge_min_len_factor,
                row_gain_cap=args.row_gain_cap)
            rad = one_hop(rad, Le, is_light, wfr_B, cache_B)
            pool_lum["B"][k] += float((rad[~is_light] @ LUM).sum())
            cam_lum["B"][k] += float(((T0 * rad[cam_idx]) @ LUM).mean())

    print(f"\n{'hop':>4} | {'pool A':>11} {'pool B':>11} {'A/B':>6} | "
          f"{'gain A':>7} {'gain B':>7} | {'cam A':>10} {'cam B':>10} {'A/B':>6}")
    print("-" * 92)
    for k in range(args.hops):
        ga = (pool_lum['A'][k] / pool_lum['A'][k-1]) if k else float('nan')
        gb = (pool_lum['B'][k] / pool_lum['B'][k-1]) if k else float('nan')
        print(f"{k+1:>4} | {pool_lum['A'][k]:>11.4e} {pool_lum['B'][k]:>11.4e} "
              f"{pool_lum['A'][k]/max(pool_lum['B'][k],1e-300):>6.3f} | "
              f"{ga:>7.3f} {gb:>7.3f} | "
              f"{cam_lum['A'][k]:>10.4e} {cam_lum['B'][k]:>10.4e} "
              f"{cam_lum['A'][k]/max(cam_lum['B'][k],1e-300):>6.3f}")
    print("\nIf the A/B ratio grows with hop index, hop-reuse correlation is "
          "compounding (the hot-bounce mechanism). If it stays ≈ 1, the heat "
          "is per-hop calibration — same in both arms.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wh", type=int, default=32)
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--hops", type=int, default=5)
    ap.add_argument("--c", type=int, default=30)
    ap.add_argument("--max-depth", type=int, default=8)
    ap.add_argument("--merge-min-len-factor", type=float, default=1.0)
    ap.add_argument("--row-gain-cap", type=float, default=1.0)
    run(ap.parse_args())
