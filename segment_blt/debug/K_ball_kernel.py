"""
K. Ball-kernel BLT — replaces voxel-grid clustering with a per-receiver
   ball query (KD-tree). Standalone — does not modify core code.

Per the paper §3.1: "the uniform kernel over a finite region K(y_i)
centered at the vertex y_i". That's a ball, not a voxel cell. Our
voxel implementation is an *approximation* of this that introduces:
  • discontinuous boundaries (1/A inside, 0 outside step-function)
  • per-receiver asymmetric kernel shapes (corner vs center of voxel)
  • the hard "skip small clusters" optimization in pair_cache.py:94
    that drops whole cluster's worth of receivers to zero coverage

All three are removed by using a true ball around each receiver. The
math is identical (K = 1/V_ball, ∫K dV = 1), only the spatial structure
differs. Cost: O(S log S) for the KD-tree vs O(S) for voxel sort, but
no boundary artifacts to fight per iteration.

Usage:
    python K_ball_kernel.py --max-iter 16 --radius 0.05
    python K_ball_kernel.py --max-iter 16 --radius 0.03 --width 128
    python K_ball_kernel.py --max-iter 32 --radius 0.05 --progressive
"""
from _setup import cornell_scene

import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import mitsuba as mi
from scipy.spatial import cKDTree

from segments.renderer import Renderer, _make_blt_components, _sample_segments, _final_gather
from segments.pair_cache import PairCache
from segments.primitives import SegmentTechnique


DEFAULT_REF = "/Users/celine/Documents/projects/ray-maps/baseline/spp_renders-128x128-d-1/reference.exr"


# ─────────────────────────────────────────────────────────────────────────────
# Ball-kernel pair cache builder — drop-in replacement for build_pair_cache
# ─────────────────────────────────────────────────────────────────────────────

def build_pair_cache_ball(cluster, samplers, radius):
    """
    Build PairCache via per-receiver ball queries instead of voxel grid.

    Propagation pairs: (i, j) where ||y_i - x_j|| < radius and i ≠ j.
    MMIS pairs: (j, t) where ||x_j - y_t|| < radius, t ≠ j, both have
        samplers, and j is not trivially-weighted.
    """
    segs = cluster.segments
    S = len(segs)
    if S == 0:
        return PairCache()

    # Endpoint positions as flat arrays
    y_positions = np.array(
        [[float(s.y.p.x), float(s.y.p.y), float(s.y.p.z)] for s in segs],
        dtype=np.float64,
    )
    x_positions = np.array(
        [[float(s.x.p.x), float(s.x.p.y), float(s.x.p.z)] for s in segs],
        dtype=np.float64,
    )

    # KD-trees over each endpoint set
    tree_x = cKDTree(x_positions)
    tree_y = cKDTree(y_positions)

    # Trivial set — same convention as build_pair_cache
    trivial_set = set()
    for seg_idx, s in enumerate(segs):
        if s.x.is_camera or (s.technique == SegmentTechnique.LIGHT and s.y.is_light):
            trivial_set.add(seg_idx)
    trivial_mask = np.zeros(S, dtype=bool)
    if trivial_set:
        trivial_mask[np.fromiter(trivial_set, dtype=np.int32)] = True

    has_sampler = np.array(
        [samplers.get(s.technique) is not None for s in segs], dtype=bool
    )
    light_terminal = np.array(
        [(s.technique == SegmentTechnique.LIGHT and s.y.is_light) for s in segs],
        dtype=bool,
    )

    # ── Propagation pairs: for each receiver i, find sources j with x_j ∈ ball(y_i, r)
    # Vectorized via query_ball_tree (tree-to-tree): returns lists.
    # Iterate over receivers since output is variable-length per i.
    prop_i_list, prop_j_list = [], []
    # query_ball_tree returns list-of-lists: tree_x.query_ball_tree(tree_y_pts, r)
    # We want: for each y_i point, which x_j points are within r.
    neighbors_per_i = tree_x.query_ball_point(y_positions, r=radius)
    for i, j_neighbors in enumerate(neighbors_per_i):
        if light_terminal[i]:
            continue  # light-terminal i doesn't receive any propagation
        for j in j_neighbors:
            if j == i:
                continue
            if light_terminal[j]:
                continue
            prop_i_list.append(i)
            prop_j_list.append(j)

    # Compute prop_fr for each pair — same BSDF call as build_pair_cache
    P = len(prop_i_list)
    prop_fr = np.zeros((P, 3), dtype=np.float64)
    cam_sampler = samplers.get(SegmentTechnique.CAMERA)
    if cam_sampler is not None:
        for k, (i, j) in enumerate(zip(prop_i_list, prop_j_list)):
            fr = cam_sampler.shift_invariant_bsdf(segs[i], segs[j])
            prop_fr[k] = [float(fr.x), float(fr.y), float(fr.z)]

    prop_i_arr = np.array(prop_i_list, dtype=np.int32) if P > 0 else np.empty(0, np.int32)
    prop_j_arr = np.array(prop_j_list, dtype=np.int32) if P > 0 else np.empty(0, np.int32)

    # ── MMIS pairs: for each segment j (non-trivial), find auxiliaries t with y_t ∈ ball(x_j, r)
    mmis_j_list, mmis_t_list = [], []
    neighbors_per_j = tree_y.query_ball_point(x_positions, r=radius)
    for j, t_neighbors in enumerate(neighbors_per_j):
        if trivial_mask[j] or not has_sampler[j]:
            continue
        for t in t_neighbors:
            if t == j:
                continue
            if not has_sampler[t]:
                continue
            mmis_j_list.append(j)
            mmis_t_list.append(t)

    mmis_j_arr = np.array(mmis_j_list, dtype=np.int32) if mmis_j_list else np.empty(0, np.int32)
    mmis_t_arr = np.array(mmis_t_list, dtype=np.int32) if mmis_t_list else np.empty(0, np.int32)

    M = len(mmis_j_arr)
    mmis_pdf = np.zeros(M, dtype=np.float64)
    for k in range(M):
        j_idx = int(mmis_j_arr[k])
        t_idx = int(mmis_t_arr[k])
        sampler = samplers.get(segs[t_idx].technique)
        if sampler is None:
            continue
        mmis_pdf[k] = sampler.conditional_pdf(segs[j_idx], segs[t_idx])

    return PairCache(
        prop_i=prop_i_arr,
        prop_j=prop_j_arr,
        prop_fr=prop_fr,
        mmis_j=mmis_j_arr,
        mmis_t=mmis_t_arr,
        mmis_pdf=mmis_pdf,
        trivial_mmis=np.array(sorted(trivial_set), dtype=np.int32),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Render loop — substitutes ball pair-cache, keeps everything else identical
# ─────────────────────────────────────────────────────────────────────────────

def render_one_iter_ball(scene, sensor, renderer, H, W,
                         cluster, mmis, propagation, samplers,
                         iteration, radius, beta=0.1):
    """One iteration of BLT using ball-kernel pair_cache."""
    sampler = mi.load_dict({"type": "independent", "sample_count": 1})
    sampler.seed(iteration, W * H)

    segment_pool, camera_first_segments = _sample_segments(
        scene, sensor, sampler, H, W, beta=beta, add_light_samples=False,
    )
    if not segment_pool.paths:
        return np.zeros((H, W, 3), dtype=np.float64), 0

    cluster.set_segments(segment_pool.paths)
    # NOTE: we don't call cluster.cluster() — no voxel grid needed.
    # But MMIS/propagation read cluster.segments only, so that's enough.

    pair_cache = build_pair_cache_ball(cluster, samplers, radius)
    mmis.compute_all_mmis_weights_vec(cluster, pair_cache)
    propagation.iterate_propogation_vec(cluster, pair_cache)

    img = _final_gather(camera_first_segments, H, W)
    return img, len(pair_cache.prop_i)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=DEFAULT_REF)
    ap.add_argument("--max-iter", type=int, default=8)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--radius", type=float, default=0.05,
                    help="ball-kernel radius in scene units. Cornell box is "
                         "~2m wide, so 0.05 = 5cm is roughly comparable to "
                         "the voxel_size we had at c=30.")
    ap.add_argument("--N", type=int, default=8, help="num_prop_iterations")
    ap.add_argument("--geom-clamp", type=float, default=200.0)
    ap.add_argument("--progressive", action="store_true",
                    help="shrink radius via Knaus-Zwicker α=0.2 per iter")
    ap.add_argument("--alpha", type=float, default=0.2)
    args = ap.parse_args()

    W = H = args.width
    print(f"loading reference: {args.ref}")
    ref = np.array(mi.Bitmap(args.ref))[..., :3].astype(np.float64)
    if ref.shape[:2] != (H, W):
        raise SystemExit(
            f"ref shape {ref.shape[:2]} ≠ {(H, W)}. Use make_reference.py to "
            f"generate one at this resolution, or pass --width."
        )
    ref_mean = float(ref.sum(-1).mean())
    print(f"  ref shape={ref.shape}   mean = {ref_mean:.4e}")

    scene = cornell_scene(W, H)
    sensor = scene.sensors()[0]
    renderer = Renderer()

    # Build BLT components with whatever clamp settings are chosen.
    # We don't use cluster.voxel_size, but Cluster instance is still needed
    # as a segment container for MMIS/propagation.
    cluster, mmis, propagation, samplers = _make_blt_components(
        scene, add_light_samples=False,
        num_prop_iterations=args.N, cluster_c=30,
        geom_clamp_factor=args.geom_clamp, geom_clamp_mode="hard",
    )

    accum = np.zeros((H, W, 3), dtype=np.float64)
    rmses, mrs, pair_counts = [], [], []
    radii = []

    r0 = args.radius
    for i in range(args.max_iter):
        # Knaus-Zwicker progressive on the BALL radius
        if args.progressive and i > 0:
            r = r0 * (i ** (-args.alpha / 2))
        else:
            r = r0
        radii.append(r)

        print(f"\n── iter {i+1}/{args.max_iter}  radius={r:.4e} ──")
        t0 = time.time()
        img, n_pairs = render_one_iter_ball(
            scene, sensor, renderer, H, W,
            cluster, mmis, propagation, samplers,
            iteration=i, radius=r,
        )
        dt = time.time() - t0
        accum += img.astype(np.float64)
        running = accum / (i + 1)
        rmse = float(np.sqrt(((running - ref) ** 2).mean()))
        mr = float(running.sum(-1).mean() / max(ref_mean, 1e-12))
        rmses.append(rmse)
        mrs.append(mr)
        pair_counts.append(n_pairs)
        print(f"   pairs={n_pairs:,}  RMSE={rmse:.4e}  mean_ratio={mr:.4f}  ({dt:.1f}s)")

    # ── plot ──────────────────────────────────────────────────────
    iters = np.arange(1, args.max_iter + 1)
    rmses_arr = np.array(rmses)
    slope, _ = np.polyfit(np.log(iters), np.log(rmses_arr), 1)
    print(f"\nlog-log slope of RMSE vs iter = {slope:.3f}  (MC ideal ≈ -0.5)")

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    axes[0, 0].loglog(iters, rmses_arr, "o-", label=f"ball (slope={slope:.2f})")
    axes[0, 0].loglog(iters, rmses_arr[0] * iters ** (-0.5),
                      "k--", alpha=0.5, label="N^(-1/2) MC")
    axes[0, 0].loglog(iters, rmses_arr[0] * iters ** (-1/3),
                      "k:", alpha=0.5, label="N^(-1/3) progressive")
    axes[0, 0].set_xlabel("iter"); axes[0, 0].set_ylabel("RMSE")
    axes[0, 0].set_title(f"ball-kernel  radius={args.radius}")
    axes[0, 0].grid(True, which="both", alpha=0.3)
    axes[0, 0].legend(fontsize=9)

    axes[0, 1].plot(iters, mrs, "o-")
    axes[0, 1].axhline(1.0, color="k", linestyle="--", alpha=0.5)
    axes[0, 1].set_xlabel("iter"); axes[0, 1].set_ylabel("mean(BLT)/mean(ref)")
    axes[0, 1].set_title("mean ratio (closer to 1.0 = less bias)")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0.7, 1.2)

    # Tonemap & show running final image, side-by-side with ref
    ref_lum = ref.sum(-1)
    med = float(np.median(ref_lum[ref_lum > 0])) if (ref_lum > 0).any() else 0.0
    exposure = (0.18 / med) if med > 0 else 1.0

    def tonemap(x):
        scaled = x * exposure
        return np.clip((scaled / (1 + scaled)) ** (1/2.2), 0, 1)

    axes[1, 0].imshow(tonemap(ref))
    axes[1, 0].set_title("REFERENCE"); axes[1, 0].axis("off")
    axes[1, 1].imshow(tonemap(running))
    axes[1, 1].set_title(f"ball-kernel BLT  mr={mrs[-1]:.3f}  RMSE={rmses[-1]:.3e}")
    axes[1, 1].axis("off")

    plt.tight_layout()
    prog_tag = f"prog_a{args.alpha}" if args.progressive else "fixed"
    tag = (f"w{args.width}_i{args.max_iter}"
           f"_r{args.radius}_{prog_tag}"
           f"_N{args.N}_g{int(args.geom_clamp)}")
    out = os.path.join(os.path.dirname(__file__), f"K_ball_{tag}.png")
    plt.savefig(out, dpi=110); plt.close()
    print(f"\nsaved: {out}")

    # Coverage stats — how many pairs/receiver on average?
    avg_pairs = np.mean(pair_counts) / max(len(cluster.segments), 1)
    print(f"\naverage pairs / segment (last iter) = {pair_counts[-1] / max(len(cluster.segments), 1):.1f}")
    print(f"compare against voxel-grid baseline at c=30: ~50 pairs/segment")


if __name__ == "__main__":
    main()
