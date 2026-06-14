"""
O4: why is D at the LAST vertex of truncated paths ~300x hot?

O3 (128^2): levels 1..16 calibrated (cum 0.9966), but the deep bucket — a
single term, P16 * D(v17) on max-depth-truncated paths — reads 3.7e-2 vs a
1.2e-4 truth slice. Same gather, same code path, hot only at the final
vertex. This script answers two questions with one pool:

  1. Is D itself discontinuous in receiver depth?  Print mean lum(direct_in)
     for camera segments grouped by their depth index k = 1..17. A smooth
     curve with a spike at 17 indicts the last segment structurally; a
     smooth curve everywhere means the heat is P-D correlation instead.

  2. WHAT is inside D at depth 17?  Decompose the direct gather of depth-17
     receivers pair-by-pair (w * f * G * Le), categorized:
        own-nee   : the receiver vertex's own NEE segment
        other-nee : NEE segments from other vertices in the voxel
        bsdf-hit  : BSDF-found emitter hits
        classical : light-terminal classical links
     and dump the 10 largest single pair contributions with their factors
     (w, G, f, len_j, parent-share of the denominator) so a broken weight
     is visible directly.

Run:  python O4_last_vertex.py --width 64
"""
from _setup import cornell_scene

import argparse
import numpy as np
import mitsuba as mi

from segments.renderer import _make_blt_components, _sample_segments
from segments.pair_cache import build_pair_cache

LUM = np.array([0.2126, 0.7152, 0.0722])


def lum3(c):
    return 0.2126 * float(c.x) + 0.7152 * float(c.y) + 0.0722 * float(c.z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    W = H = args.width

    scene = cornell_scene(W, H)
    sensor = scene.sensors()[0]
    cluster, mmis, propagation, samplers = _make_blt_components(
        scene, add_light_samples=False, num_prop_iterations=8,
        cluster_c=30, geom_clamp_factor=None, mmis_clamp_factor=None,
    )
    sampler = mi.load_dict({"type": "independent", "sample_count": 1})
    sampler.seed(args.seed, W * H)
    pool, cam_first = _sample_segments(scene, sensor, sampler, H, W,
                                       beta=0.1, add_light_samples=False)
    cluster.set_segments(pool.paths)
    cluster.cluster(np.random.default_rng(seed=args.seed))
    pc = build_pair_cache(cluster, samplers,
                          merge_min_len_factor=propagation.merge_min_len_factor)
    mmis.compute_all_mmis_weights_vec(cluster, pc)
    propagation.iterate_propogation_vec(cluster, pc)

    segs = cluster.segments
    seg_index = {id(s): i for i, s in enumerate(segs)}

    # ── 1. D by receiver depth ────────────────────────────────────────────────
    by_depth = {}
    truncated_last = []          # pool indices of s_last on truncated paths
    for fs in cam_first:
        if fs is None:
            continue
        path = getattr(fs, "_path", None)
        if path is None:
            continue
        for k, seg in enumerate(path, start=1):
            if seg.y.is_light or seg.x.is_light:
                continue
            by_depth.setdefault(k, []).append(lum3(seg.direct_in))
        last = path[-1]
        if not (last.y.is_light or last.x.is_light):
            truncated_last.append(seg_index[id(last)])

    print("\n── mean lum(direct_in) by receiver depth ──")
    print(f"{'depth':>6} | {'count':>6} | {'mean D':>10} | {'p99 D':>10} | {'max D':>10}")
    for k in sorted(by_depth):
        v = np.array(by_depth[k])
        print(f"{k:>6} | {len(v):>6} | {v.mean():>10.4e} | "
              f"{np.percentile(v, 99):>10.4e} | {v.max():>10.4e}")

    # ── 2. pair-level decomposition of D at the truncated last vertices ──────
    is_light = np.array([s.y.is_light or s.x.is_light for s in segs])
    mmis_w = np.array([float(s.mmis_weight) for s in segs])
    geom = np.array([float(s.geom_term) for s in segs])
    Le_lum = np.array([lum3(s.Le) for s in segs])
    wfr_lum = (pc.prop_fr @ LUM) * mmis_w[pc.prop_j] * geom[pc.prop_j]

    recv_set = set(truncated_last)
    cats = {"own-nee": 0.0, "other-nee": 0.0, "bsdf-hit": 0.0}
    rows = []
    for p_idx in range(len(pc.prop_i)):
        i, j = int(pc.prop_i[p_idx]), int(pc.prop_j[p_idx])
        if i not in recv_set or not is_light[j]:
            continue
        contrib = wfr_lum[p_idx] * Le_lum[j]
        sj, si_ = segs[j], segs[i]
        if getattr(sj, "is_nee", False):
            cat = "own-nee" if id(sj.x) == id(si_.y) else "other-nee"
        else:
            cat = "bsdf-hit"
        cats[cat] += contrib
        rows.append((contrib, i, j, cat))

    cls_total = 0.0
    for k in range(len(pc.cls_i)):
        if int(pc.cls_i[k]) in recv_set and is_light[int(pc.cls_j[k])]:
            cls_total += float((pc.cls_w[k] @ LUM)) * Le_lum[int(pc.cls_j[k])]

    n_recv = max(len(recv_set), 1)
    print(f"\n── D decomposition at {n_recv} truncated last vertices ──")
    for cat, tot in cats.items():
        print(f"  {cat:>10}: total={tot:.4e}   per-receiver={tot/n_recv:.4e}")
    print(f"  {'classical':>10}: total={cls_total:.4e}   per-receiver={cls_total/n_recv:.4e}")

    rows.sort(reverse=True)
    print("\n── 10 largest single pair contributions into last-vertex D ──")
    print(f"{'contrib':>10} | {'cat':>9} | {'w_j':>9} | {'G_j':>10} | "
          f"{'w*G':>8} | {'len_j':>9} | {'Le_j':>7}")
    for contrib, i, j, cat in rows[:10]:
        sj = segs[j]
        print(f"{contrib:>10.3e} | {cat:>9} | {mmis_w[j]:>9.3e} | "
              f"{geom[j]:>10.3e} | {mmis_w[j]*geom[j]:>8.3f} | "
              f"{float(sj.len):>9.3e} | {Le_lum[j]:>7.2f}")
    print("\n(w*G should never exceed pi=3.1416 — anything above means the "
          "parent term is missing from that source's denominator.)")


if __name__ == "__main__":
    main()
