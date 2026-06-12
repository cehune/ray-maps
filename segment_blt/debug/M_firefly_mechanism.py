"""
M. Firefly mechanism validation — tests WHY the clamp was needed.

Three claims, each tested directly on one sampled iteration (camera-only,
no clamps anywhere):

  1. GUARD ASYMMETRY: conditional_pdf() zeroes the *parent* pdf term via
     `len_sq < 1e-10` / `cos_at_aux_y < 1e-6` guards, while Segment.__post_init__
     admits segments down to len 1e-8 and the propagation numerator keeps
     f·G > 0. Any segment whose parent term is zeroed but whose G survives
     has an unbounded w·G ratio — a literal firefly generator.

  2. PARENT DOMINANCE: for a segment with len << voxel_size, p_sum is
     dominated by its own parent's term p* = p_w · cos / len² (every other
     auxiliary sits a voxel away, contributing ~p_w·cos/voxel²). Then
     w·G = G/p* ≈ π·cos — each short pair transfers its FULL ρ·L with no
     1/N averaging, unlike long segments where N comparable denominator
     terms turn the cluster sum into a mean.

  3. LOOP GAIN: corner voxels holding m such short segments form a dense
     propagation subgraph with per-hop gain ≈ m·ρ > 1. iterate_propogation
     re-uses the SAME segments for all N prop iterations, so the loop
     compounds geometrically: (m·ρ)^N. Measured here as the spectral
     radius of the linear propagation operator via power iteration.

Run:  python M_firefly_mechanism.py --width 64 --c 30 --N 8
"""
from _setup import cornell_scene

import argparse
import numpy as np
import mitsuba as mi

from segments.renderer import _make_blt_components, _sample_segments
from segments.pair_cache import build_pair_cache
from segments.primitives import SegmentTechnique


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--c", type=int, default=30)
    ap.add_argument("--N", type=int, default=8, help="prop iterations (for gain^N report)")
    ap.add_argument("--beta", type=float, default=0.1)
    args = ap.parse_args()

    W = H = args.width
    scene = cornell_scene(W, H)
    cluster, mmis, propagation, samplers = _make_blt_components(
        scene, add_light_samples=False, num_prop_iterations=args.N,
        cluster_c=args.c, geom_clamp_factor=None, mmis_clamp_factor=None,
    )
    sensor = scene.sensors()[0]
    sampler = mi.load_dict({"type": "independent", "sample_count": 1})
    sampler.seed(0, W * H)

    pool, cam_first = _sample_segments(scene, sensor, sampler, H, W,
                                       beta=args.beta, add_light_samples=False)
    cluster.set_segments(pool.paths)
    cluster.cluster(np.random.default_rng(seed=0))
    pair_cache = build_pair_cache(cluster, samplers)
    mmis.compute_all_mmis_weights_vec(cluster, pair_cache)

    segs = cluster.segments
    S = len(segs)
    vox = float(cluster.voxel_size)
    print(f"\nS={S} segments, voxel_size={vox:.4e}")

    lens = np.array([float(s.len) for s in segs])
    geom = np.array([float(s.geom_term) for s in segs])
    w    = np.array([float(s.mmis_weight) for s in segs])

    # ── parent lookup: j.x IS the parent's y SurfacePoint object ──────────────
    y_owner = {id(s.y): i for i, s in enumerate(segs)}
    parent = np.full(S, -1, dtype=np.int64)
    for j, s in enumerate(segs):
        parent[j] = y_owner.get(id(s.x), -1)

    # p_sum per segment from the cache
    p_sum = np.zeros(S)
    if len(pair_cache.mmis_j) > 0:
        np.add.at(p_sum, pair_cache.mmis_j, pair_cache.mmis_pdf)

    # parent term recomputed directly
    seq = samplers[SegmentTechnique.CAMERA]
    p_parent = np.zeros(S)
    for j in range(S):
        t = parent[j]
        if t >= 0:
            p_parent[j] = seq.conditional_pdf(segs[j], segs[t])

    nontrivial = np.ones(S, dtype=bool)
    nontrivial[pair_cache.trivial_mmis] = False
    has_par = (parent >= 0) & nontrivial

    # ── CLAIM 1: guard asymmetry ──────────────────────────────────────────────
    killed = has_par & (p_parent == 0.0)
    danger = killed & (geom > np.median(geom[geom > 0]) * 100)
    print("\n[1] guard asymmetry")
    print(f"    non-trivial segs with a parent:        {int(has_par.sum())}")
    print(f"    parent pdf term zeroed by guards:      {int(killed.sum())}")
    print(f"    ...of those, G > 100x median (DANGER): {int(danger.sum())}")
    if danger.any():
        idx = np.argsort(-geom * danger)[:5]
        for j in idx[danger[idx]]:
            print(f"      seg {j}: len={lens[j]:.3e}  G={geom[j]:.3e}  "
                  f"p_sum={p_sum[j]:.3e}  w={w[j]:.3e}  w*G={w[j]*geom[j]:.3e}")

    # ── CLAIM 2: parent dominance for short segments ──────────────────────────
    share = np.where(p_sum > 0, p_parent / np.maximum(p_sum, 1e-300), 0.0)
    short = has_par & (lens < vox)
    longs = has_par & (lens > 3 * vox)
    print("\n[2] parent share of p_sum (1.0 = no averaging, full-strength transfer)")
    for name, m in (("len < voxel", short), ("len > 3*voxel", longs)):
        if m.any():
            print(f"    {name:14s} n={int(m.sum()):6d}  median share={np.median(share[m]):.3f}  "
                  f"median w*G={np.median((w*geom)[m]):.3f}")

    # ── CLAIM 3: spectral radius of the propagation operator ─────────────────
    print("\n[3] propagation loop gain (power iteration on M[i,j] = lum(f)*w_j*G_j)")
    if len(pair_cache.prop_i) > 0:
        LUM = np.array([0.2126, 0.7152, 0.0722])
        lum_f = pair_cache.prop_fr @ LUM
        pj = pair_cache.prop_j
        edge = lum_f * w[pj] * geom[pj]
        prop_i_all = pair_cache.prop_i
        # classical near-field links are part of the operator now
        if len(getattr(pair_cache, "cls_i", [])) > 0:
            prop_i_all = np.concatenate([prop_i_all, pair_cache.cls_i])
            pj         = np.concatenate([pj, pair_cache.cls_j])
            edge       = np.concatenate([edge, pair_cache.cls_w @ LUM])
        pair_cache.prop_i = prop_i_all  # local view for the loops below
        v = np.ones(S); rho = 0.0
        for _ in range(30):
            v2 = np.zeros(S)
            np.add.at(v2, pair_cache.prop_i, edge * v[pj])
            n2 = np.linalg.norm(v2)
            if n2 == 0: break
            rho = n2 / np.linalg.norm(v)
            v = v2 / n2
        print(f"    spectral radius ~= {rho:.3f}   ->  gain over N={args.N} hops: {rho**args.N:.3e}")
        print(f"    (rho > 1 means propagation DIVERGES on this segment pool)")
        # where does the dominant mode live?
        top = np.argsort(-v)[:5]
        for j in top:
            p = [float(segs[j].y.p[k]) for k in range(3)]
            print(f"      mode seg {j}: len={lens[j]:.3e} ({lens[j]/vox:.2f} voxels)  "
                  f"w*G={w[j]*geom[j]:.2f}  y={np.round(p,3)}")
        row_gain = np.zeros(S)
        np.add.at(row_gain, pair_cache.prop_i, edge)
        n_amp = int((row_gain > 1.0).sum())
        print(f"    segments with per-hop gain > 1: {n_amp}/{S} "
              f"({100*n_amp/S:.1f}%)  max={row_gain.max():.2f}")

        # ── same operator AFTER the energy-conservation projection ───────────
        cap = 1.0
        scale = np.where(row_gain > cap, cap / np.maximum(row_gain, 1e-300), 1.0)
        edge_c = edge * scale[pair_cache.prop_i]
        v = np.ones(S); rho_c = 0.0
        for _ in range(30):
            v2 = np.zeros(S)
            np.add.at(v2, pair_cache.prop_i, edge_c * v[pj])
            n2 = np.linalg.norm(v2)
            if n2 == 0: break
            rho_c = n2 / np.linalg.norm(v)
            v = v2 / n2
        print(f"    AFTER row-gain cap at {cap:g}: spectral radius ~= {rho_c:.3f}  "
              f"->  gain over N={args.N} hops: {rho_c**args.N:.3e}")


if __name__ == "__main__":
    main()
