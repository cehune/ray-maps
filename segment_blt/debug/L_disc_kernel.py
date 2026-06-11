"""
L. Disc-kernel segment-based renderer — restricts neighbors to the
   tangent plane at the receiver point.

Light transport happens on 2D surfaces. A 3D ball (or voxel) kernel
includes neighbors that are spatially close but on different surfaces —
e.g. a wall point and a floor point near the wall-floor corner. Those
cross-surface "neighbors" are conceptually invalid for density
estimation: their BSDF interactions are unrelated to the receiver's.

A disc kernel placed on the receiver's tangent plane fixes this:

    auxiliary y_t is in disc(y_i) iff
        1. ||y_t - y_i|| ≤ radius              (spatial extent)
        2. |n_i · (y_t - y_i)| ≤ height_tol    (near tangent plane)
        3. n_i · n_t ≥ cos(theta_max)          (similar normal orientation)

Conditions (2) and (3) exclude cross-surface contamination. K = 1/(πr²)
inside the disc, 0 outside; ∫K dA = 1, so the paper's kernel constraint
(eq 5) holds.

This is the canonical density-estimation kernel in photon mapping.

Usage:
    python L_disc_kernel.py --max-iter 16 --radius 0.05
    python L_disc_kernel.py --max-iter 16 --radius 0.05 --height-tol 0.01
    python L_disc_kernel.py --max-iter 16 --radius 0.05 --normal-deg 30
"""
from _setup import cornell_scene
from _cache import save_render, load_render

import argparse
import math
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
# Disc-kernel pair cache builder
# ─────────────────────────────────────────────────────────────────────────────

def _filter_disc(receiver_pos, receiver_normal,
                 cand_positions, cand_normals,
                 radius, height_tol, cos_theta_max):
    """
    From a list of candidate points already within `radius` of receiver_pos,
    return a boolean mask for which ones are inside the disc.

    Disc plane: n_i · (y_t - y_i) = 0, |height| ≤ height_tol.
    Normal cone: n_i · n_t ≥ cos_theta_max.
    Returns boolean array of length len(cand_positions).
    """
    delta = cand_positions - receiver_pos  # (M, 3)
    height = np.abs(delta @ receiver_normal)
    height_ok = height <= height_tol
    normal_dot = cand_normals @ receiver_normal
    normal_ok = normal_dot >= cos_theta_max
    return height_ok & normal_ok


def build_pair_cache_disc(cluster, samplers, radius, height_tol, normal_deg):
    """
    Build PairCache via per-receiver disc queries.

    Each receiver's disc lies on its tangent plane:
      • spatial radius `radius`
      • height tolerance `height_tol` (perpendicular to receiver's normal)
      • normal cone aperture `normal_deg` (degrees) for the auxiliary's
        own normal to count as the same surface
    """
    segs = cluster.segments
    S = len(segs)
    if S == 0:
        return PairCache()

    cos_theta_max = math.cos(math.radians(normal_deg))

    # Endpoint positions + normals
    y_pos = np.array([[float(s.y.p.x), float(s.y.p.y), float(s.y.p.z)] for s in segs],
                     dtype=np.float64)
    y_nrm = np.array([[float(s.y.n.x), float(s.y.n.y), float(s.y.n.z)] for s in segs],
                     dtype=np.float64)
    x_pos = np.array([[float(s.x.p.x), float(s.x.p.y), float(s.x.p.z)] for s in segs],
                     dtype=np.float64)
    x_nrm = np.array([[float(s.x.n.x), float(s.x.n.y), float(s.x.n.z)] for s in segs],
                     dtype=np.float64)

    tree_x = cKDTree(x_pos)
    tree_y = cKDTree(y_pos)

    # Trivial set
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

    # ── Propagation pairs: receiver i, source j with x_j in disc(y_i)
    prop_i_list, prop_j_list = [], []
    neighbors_per_i = tree_x.query_ball_point(y_pos, r=radius)
    for i, j_candidates in enumerate(neighbors_per_i):
        if light_terminal[i] or len(j_candidates) == 0:
            continue
        cand_idx = np.array(j_candidates, dtype=np.int32)
        mask = _filter_disc(
            y_pos[i], y_nrm[i],
            x_pos[cand_idx], x_nrm[cand_idx],
            radius, height_tol, cos_theta_max,
        )
        for j_local, ok in enumerate(mask):
            if not ok:
                continue
            j = int(cand_idx[j_local])
            if j == i or light_terminal[j]:
                continue
            prop_i_list.append(i)
            prop_j_list.append(j)

    P = len(prop_i_list)
    prop_fr = np.zeros((P, 3), dtype=np.float64)
    cam_sampler = samplers.get(SegmentTechnique.CAMERA)
    if cam_sampler is not None:
        for k, (i, j) in enumerate(zip(prop_i_list, prop_j_list)):
            fr = cam_sampler.shift_invariant_bsdf(segs[i], segs[j])
            prop_fr[k] = [float(fr.x), float(fr.y), float(fr.z)]

    prop_i_arr = np.array(prop_i_list, dtype=np.int32) if P > 0 else np.empty(0, np.int32)
    prop_j_arr = np.array(prop_j_list, dtype=np.int32) if P > 0 else np.empty(0, np.int32)

    # ── MMIS pairs: receiver j, auxiliary t with y_t in disc(x_j)
    mmis_j_list, mmis_t_list = [], []
    neighbors_per_j = tree_y.query_ball_point(x_pos, r=radius)
    for j, t_candidates in enumerate(neighbors_per_j):
        if trivial_mask[j] or not has_sampler[j] or len(t_candidates) == 0:
            continue
        cand_idx = np.array(t_candidates, dtype=np.int32)
        mask = _filter_disc(
            x_pos[j], x_nrm[j],
            y_pos[cand_idx], y_nrm[cand_idx],
            radius, height_tol, cos_theta_max,
        )
        for t_local, ok in enumerate(mask):
            if not ok:
                continue
            t = int(cand_idx[t_local])
            if t == j or not has_sampler[t]:
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
# Render loop
# ─────────────────────────────────────────────────────────────────────────────

def render_one_iter_disc(scene, sensor, renderer, H, W,
                         cluster, mmis, propagation, samplers,
                         iteration, radius, height_tol, normal_deg, beta=0.1):
    sampler = mi.load_dict({"type": "independent", "sample_count": 1})
    sampler.seed(iteration, W * H)

    segment_pool, camera_first_segments = _sample_segments(
        scene, sensor, sampler, H, W, beta=beta, add_light_samples=False,
    )
    if not segment_pool.paths:
        return np.zeros((H, W, 3), dtype=np.float64), 0

    cluster.set_segments(segment_pool.paths)
    pair_cache = build_pair_cache_disc(cluster, samplers, radius, height_tol, normal_deg)
    mmis.compute_all_mmis_weights_vec(cluster, pair_cache)
    propagation.iterate_propogation_vec(cluster, pair_cache)
    img = _final_gather(camera_first_segments, H, W)
    return img, len(pair_cache.prop_i)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=DEFAULT_REF)
    ap.add_argument("--max-iter", type=int, default=48)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--radius", type=float, default=0.05,
                    help="disc radius in scene units (Cornell ~2m).")
    ap.add_argument("--height-tol", type=float, default=0.01,
                    help="max perpendicular distance from receiver's tangent plane "
                         "(scene units). Smaller = stricter same-surface check. "
                         "Default 1cm; try 5mm for tighter, 2cm for looser.")
    ap.add_argument("--normal-deg", type=float, default=30.0,
                    help="max angle (degrees) between receiver normal and auxiliary "
                         "normal. 30° accepts gently curved surfaces; 90° disables "
                         "the cone check entirely.")
    ap.add_argument("--N", type=int, default=8, help="num_prop_iterations")
    ap.add_argument("--geom-clamp", type=float, default=200.0)
    ap.add_argument("--refresh", action="store_true",
                    help="ignore cached EXR/metrics and re-render")
    args = ap.parse_args()

    W = H = args.width
    print(f"loading reference: {args.ref}")
    ref = np.array(mi.Bitmap(args.ref))[..., :3].astype(np.float64)
    if ref.shape[:2] != (H, W):
        raise SystemExit(
            f"ref shape {ref.shape[:2]} ≠ {(H, W)}. Generate with make_reference.py."
        )
    ref_mean = float(ref.sum(-1).mean())
    print(f"  ref shape={ref.shape}   mean={ref_mean:.4e}")
    print(f"  config: radius={args.radius}  height_tol={args.height_tol}  "
          f"normal_deg={args.normal_deg}")

    scene = cornell_scene(W, H)
    sensor = scene.sensors()[0]
    renderer = Renderer()
    cluster, mmis, propagation, samplers = _make_blt_components(
        scene, add_light_samples=False,
        num_prop_iterations=args.N, cluster_c=30,
        geom_clamp_factor=args.geom_clamp, geom_clamp_mode="hard",
    )

    cache_tag = (f"L_w{W}_i{args.max_iter}_r{args.radius}_h{args.height_tol}"
                 f"_n{int(args.normal_deg)}_N{args.N}_g{int(args.geom_clamp)}")
    hit = None if args.refresh else load_render(cache_tag)
    if hit is not None and hit["meta"] is not None:
        m = hit["meta"]
        rmses, mrs, pair_counts = m["rmses"], m["mrs"], m["pair_counts"]
        running = hit["image"]
        n_segments = m.get("n_segments", 0)
        print(f"   (cache hit: {cache_tag})")
        if m.get("ref") != os.path.basename(args.ref):
            print(f"   ⚠ cached metrics vs '{m.get('ref')}', current ref "
                  f"'{os.path.basename(args.ref)}' — pass --refresh to recompute")
    else:
        accum = np.zeros((H, W, 3), dtype=np.float64)
        rmses, mrs, pair_counts = [], [], []

        for i in range(args.max_iter):
            print(f"\n── iter {i+1}/{args.max_iter} ──")
            t0 = time.time()
            img, n_pairs = render_one_iter_disc(
                scene, sensor, renderer, H, W,
                cluster, mmis, propagation, samplers,
                iteration=i, radius=args.radius,
                height_tol=args.height_tol, normal_deg=args.normal_deg,
            )
            dt = time.time() - t0
            accum += img.astype(np.float64)
            running = accum / (i + 1)
            rmse = float(np.sqrt(((running - ref) ** 2).mean()))
            mr = float(running.sum(-1).mean() / max(ref_mean, 1e-12))
            rmses.append(rmse); mrs.append(mr); pair_counts.append(n_pairs)
            print(f"   pairs={n_pairs:,}  RMSE={rmse:.4e}  mean_ratio={mr:.4f}  ({dt:.1f}s)")
        n_segments = len(cluster.segments)
        save_render(cache_tag, running, meta={
            "config": {"W": W, "max_iter": args.max_iter, "radius": args.radius,
                       "height_tol": args.height_tol, "normal_deg": args.normal_deg,
                       "N": args.N, "geom_clamp": args.geom_clamp, "lookup": "disc"},
            "ref": os.path.basename(args.ref),
            "rmses": rmses, "mrs": mrs, "pair_counts": pair_counts,
            "n_segments": n_segments,
        })

    # ── plot ──────────────────────────────────────────────────────
    iters = np.arange(1, args.max_iter + 1)
    rmses_arr = np.array(rmses)
    slope, _ = np.polyfit(np.log(iters), np.log(rmses_arr), 1)
    print(f"\nlog-log slope of RMSE vs iter = {slope:.3f}  (MC ideal ≈ -0.5)")

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    axes[0, 0].loglog(iters, rmses_arr, "o-", label=f"disc (slope={slope:.2f})")
    axes[0, 0].loglog(iters, rmses_arr[0] * iters ** (-0.5),
                      "k--", alpha=0.5, label="N^(-1/2) MC")
    axes[0, 0].loglog(iters, rmses_arr[0] * iters ** (-1/3),
                      "k:", alpha=0.5, label="N^(-1/3) progressive")
    axes[0, 0].set_xlabel("iter"); axes[0, 0].set_ylabel("RMSE")
    axes[0, 0].set_title(f"disc-kernel  r={args.radius}  h={args.height_tol}  θ={args.normal_deg}°")
    axes[0, 0].grid(True, which="both", alpha=0.3)
    axes[0, 0].legend(fontsize=9)

    axes[0, 1].plot(iters, mrs, "o-")
    axes[0, 1].axhline(1.0, color="k", linestyle="--", alpha=0.5)
    axes[0, 1].set_xlabel("iter"); axes[0, 1].set_ylabel("mean(BLT)/mean(ref)")
    axes[0, 1].set_title("mean ratio")
    axes[0, 1].grid(True, alpha=0.3); axes[0, 1].set_ylim(0.7, 1.2)

    ref_lum = ref.sum(-1)
    med = float(np.median(ref_lum[ref_lum > 0])) if (ref_lum > 0).any() else 0.0
    exposure = (0.18 / med) if med > 0 else 1.0

    def tonemap(x):
        scaled = x * exposure
        return np.clip((scaled / (1 + scaled)) ** (1/2.2), 0, 1)

    axes[1, 0].imshow(tonemap(ref))
    axes[1, 0].set_title("REFERENCE"); axes[1, 0].axis("off")
    axes[1, 1].imshow(tonemap(running))
    axes[1, 1].set_title(f"disc-kernel  mr={mrs[-1]:.3f}  RMSE={rmses[-1]:.3e}")
    axes[1, 1].axis("off")

    plt.tight_layout()
    tag = (f"w{args.width}_i{args.max_iter}"
           f"_r{args.radius}_h{args.height_tol}_n{int(args.normal_deg)}"
           f"_N{args.N}_g{int(args.geom_clamp)}")
    out = os.path.join(os.path.dirname(__file__), f"L_disc_{tag}.png")
    plt.savefig(out, dpi=110); plt.close()
    print(f"\nsaved: {out}")

    avg_pairs = pair_counts[-1] / max(n_segments, 1)
    print(f"\naverage pairs / segment (last iter) = {avg_pairs:.1f}")
    print(f"  compare to ball-kernel at r={args.radius}: ~24 pairs/segment")
    print(f"  reduction = {1 - avg_pairs/24:.1%}")
    print("If you're losing too many pairs (avg < 10), try larger --radius or"
          " --height-tol.")


if __name__ == "__main__":
    main()
