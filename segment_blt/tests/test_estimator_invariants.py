"""
Estimator invariants — regression locks for the firefly/convergence audit.

Each test pins one of the fixed defects so it cannot silently return:

  1. Guard symmetry        → test_no_guard_kills, test_parent_term_bound
  2. Sub-kernel exclusion  → test_subkernel_sources_excluded, test_spectral_radius
  3. Classical links       → test_classical_links
  4. Row-gain projection   → test_row_gain_cap
  6. Progressive ownership → test_progressive_schedule_owned_by_caller
  7. Split final gather    → test_direct_split_readout

Key math (diffuse-only scene): for any segment j whose path-parent t* is in
the pool, the parent's conditional-pdf term is

    q(j | t*) = (|n_x · ŝ_j| / π) · |n_y · ŝ_j| / len_j²  =  G(j) / π ,

and the parent shares j.x's SurfacePoint (same position → same jitter →
same voxel), so p_sum ≥ G(j)/π and therefore

    mmis_weight(j) · geom_term(j)  ≤  π .

Any violation means a denominator term was deleted while the numerator kept
f·G — the guard-asymmetry firefly class. This bound is PER PAIR; divergence
through multi-hop reuse is bounded separately by the spectral-radius test.

Assumes a diffuse-only scene (cornell box). Pool size is kept small so the
whole module runs in ~1 minute.
"""
import os
import sys
import math

import numpy as np
import pytest

_SRC = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import mitsuba as mi

mi.set_variant("scalar_rgb")

from segments.primitives import SegmentTechnique, MIN_SEG_LEN  # noqa: E402
from segments.cluster import Cluster                            # noqa: E402
from segments.mmis import MMIS                                  # noqa: E402
from segments.pair_cache import build_pair_cache                # noqa: E402
from segments.propagation import Propagation                    # noqa: E402
from segments.sampler_types import Sampler, SequentialSampler   # noqa: E402
from segments.segment_gen_sampler import sample_camera_path     # noqa: E402

W, H = 20, 20
MAX_DEPTH = 6
CLUSTER_C = 30
MERGE_MIN_LEN_FACTOR = 1.0
ROW_GAIN_CAP = 1.0
LUM = np.array([0.2126, 0.7152, 0.0722])


def _lum(c) -> float:
    return 0.2126 * float(c.x) + 0.7152 * float(c.y) + 0.0722 * float(c.z)


@pytest.fixture(scope="module")
def pool():
    """One sampled camera-only pool + clustering + pair cache + MMIS weights."""
    d = mi.cornell_box()
    d["sensor"]["film"]["width"] = W
    d["sensor"]["film"]["height"] = H
    d["integrator"] = {"type": "path", "max_depth": -1}
    scene = mi.load_dict(d)
    sensor = scene.sensors()[0]

    sampler = mi.load_dict({"type": "independent", "sample_count": 1})
    sampler.seed(0, W * H)

    segments = []
    camera_first = []
    for y in range(H):
        for x in range(W):
            pos = mi.Point2f(x + sampler.next_1d(), y + sampler.next_1d())
            ap = mi.Point2f(sampler.next_1d(), sampler.next_1d())
            film = mi.Point2f(pos.x / W, pos.y / H)
            ray, ray_w = sensor.sample_ray(
                time=0.0, sample1=sampler.next_1d(), sample2=film, sample3=ap)
            si = scene.ray_intersect(ray)
            path = sample_camera_path(scene, sampler, ray, ray_w, si,
                                      max_depth=MAX_DEPTH)
            if path:
                path[0]._next_seg = path[1] if len(path) > 1 else None
                path[0]._path = path
                camera_first.append(path[0])
                segments.extend(path)
            else:
                camera_first.append(None)

    samplers = Sampler.build_registry(SequentialSampler())
    cluster = Cluster(c=CLUSTER_C, alpha=0.25)
    cluster.set_scene_aabb(scene.bbox())
    cluster.set_segments(segments)
    cluster.cluster(np.random.default_rng(seed=0))

    cache = build_pair_cache(cluster, samplers,
                             merge_min_len_factor=MERGE_MIN_LEN_FACTOR)
    mmis = MMIS()
    mmis.compute_all_mmis_weights_vec(cluster, cache)

    # parent lookup: parent.y IS child.x (same SurfacePoint object on a path)
    y_owner = {id(s.y): i for i, s in enumerate(segments)}
    parent = {j: y_owner.get(id(s.x)) for j, s in enumerate(segments)}

    return dict(scene=scene, segments=segments, camera_first=camera_first,
                cluster=cluster, cache=cache, samplers=samplers,
                parent=parent,
                min_len=MERGE_MIN_LEN_FACTOR * float(cluster.voxel_size))


def _weighted_fr(pool_):
    """Mirror of propagation's per-pair weight: prop_fr · mmis_w[j] · geom[j]."""
    segs = pool_["segments"]
    cache = pool_["cache"]
    mmis_w = np.array([float(s.mmis_weight) for s in segs])
    geom = np.array([float(s.geom_term) for s in segs])
    return cache.prop_fr * (mmis_w[cache.prop_j] * geom[cache.prop_j])[:, None]


# ──────────────────────────────────────────────────────────────────────────────
# 1. Guard symmetry
# ──────────────────────────────────────────────────────────────────────────────

def test_no_guard_kills(pool):
    """Every mergeable, non-trivial camera segment with a parent in the pool
    must have p_sum > 0 (the parent term alone guarantees it). A zero weight
    here means a pdf guard deleted the parent term — the firefly class."""
    segs, parent = pool["segments"], pool["parent"]
    min_len = pool["min_len"]
    kills = []
    for j, s in enumerate(segs):
        if s.technique != SegmentTechnique.CAMERA or s.x.is_camera:
            continue
        if float(s.len) < min_len:        # excluded from merging: weight 0 by design
            continue
        p = parent.get(j)
        if p is None or p == j:
            continue
        if segs[p].y.is_light or segs[p].y.is_camera:
            continue                       # parent can't be an auxiliary
        if s.mmis_weight == 0.0:
            kills.append(j)
    assert not kills, (
        f"{len(kills)} reachable segments have mmis_weight == 0 despite a live "
        f"parent term — a denominator guard fires where Segment admission "
        f"does not (guard asymmetry). Examples: {kills[:5]}")


def test_parent_term_bound(pool):
    """w_j · G_j ≤ π for diffuse: p_sum ≥ q(j|parent) = G_j/π exactly."""
    segs, parent = pool["segments"], pool["parent"]
    worst = 0.0
    n_checked = 0
    viol = []
    for j, s in enumerate(segs):
        if s.technique != SegmentTechnique.CAMERA or s.x.is_camera:
            continue
        p = parent.get(j)
        if p is None or p == j or segs[p].y.is_light or segs[p].y.is_camera:
            continue
        wg = float(s.mmis_weight) * float(s.geom_term)
        worst = max(worst, wg)
        n_checked += 1
        if wg > math.pi * 1.01:
            viol.append((j, wg))
    print(f"\n  parent-term bound: {n_checked} segments checked, "
          f"max w·G = {worst:.4f} (bound π = {math.pi:.4f})")
    assert not viol, (
        f"{len(viol)} segments violate w·G ≤ π — the parent's pdf term is "
        f"missing or undervalued in p_sum. Worst: {sorted(viol, key=lambda v: -v[1])[:5]}")


# ──────────────────────────────────────────────────────────────────────────────
# 2. Sub-kernel exclusion + spectral radius
# ──────────────────────────────────────────────────────────────────────────────

def test_subkernel_sources_excluded(pool):
    segs, cache, min_len = pool["segments"], pool["cache"], pool["min_len"]
    src_lens = np.array([float(segs[int(j)].len) for j in cache.prop_j])
    n_short = int((src_lens < min_len).sum())
    assert n_short == 0, (
        f"{n_short} propagation sources are shorter than the kernel "
        f"(min_len={min_len:.3e}) — sub-kernel exclusion is not applied.")


def test_spectral_radius_below_one(pool):
    """Power-iterate the merged propagation operator (with light pinning and
    classical links, mirroring iterate_propogation_vec). ρ < 1 is the
    convergence condition; the corner-firefly bug had ρ ≈ 1.3."""
    segs, cache = pool["segments"], pool["cache"]
    S = len(segs)
    wfr = _weighted_fr(pool)
    lum_fr = wfr @ LUM
    cls_lum = (cache.cls_w @ LUM) if len(cache.cls_i) else np.empty(0)
    is_light = np.array([s.y.is_light or s.x.is_light for s in segs], dtype=bool)

    rng = np.random.default_rng(1)
    x = rng.random(S)
    x[is_light] = 0.0
    rho = 0.0
    for _ in range(50):
        x_new = np.zeros(S)
        np.add.at(x_new, cache.prop_i, lum_fr * x[cache.prop_j])
        if len(cache.cls_i):
            np.add.at(x_new, cache.cls_i, cls_lum * x[cache.cls_j])
        x_new[is_light] = 0.0
        n = np.linalg.norm(x_new)
        if n == 0.0:
            rho = 0.0
            break
        rho = n / np.linalg.norm(x)
        x = x_new / n * np.linalg.norm(x)  # keep scale stable
    print(f"\n  spectral radius ρ(M) ≈ {rho:.3f}")
    assert rho < 1.0, (
        f"ρ(M) = {rho:.3f} ≥ 1: propagation is a divergent fixed-point "
        f"iteration (corner-firefly mechanism).")
    if rho > 0.9:
        print("  WARNING: ρ > 0.9 — close to divergence; check exclusion factor.")


# ──────────────────────────────────────────────────────────────────────────────
# 3. Classical links
# ──────────────────────────────────────────────────────────────────────────────

def test_classical_links(pool):
    segs, cache, min_len = pool["segments"], pool["cache"], pool["min_len"]
    for k in range(len(cache.cls_i)):
        p, j = int(cache.cls_i[k]), int(cache.cls_j[k])
        sj = segs[j]
        # only sub-kernel camera segments are routed classically
        assert float(sj.len) < min_len, f"cls child {j} is not sub-kernel"
        assert sj.technique == SegmentTechnique.CAMERA
        # exact parentage by object identity
        assert id(segs[p].y) == id(sj.x), f"cls pair ({p},{j}) is not parent/child"
        # weight = f·cosθ/p_ω ≤ albedo ≤ 1 per channel for diffuse
        assert np.all(cache.cls_w[k] <= 1.0 + 1e-6), (
            f"cls weight {cache.cls_w[k]} exceeds albedo bound — not a "
            f"throughput?")
    # no double counting: classical edges must not also be merge edges
    merge_edges = set(zip(cache.prop_i.tolist(), cache.prop_j.tolist()))
    cls_edges = set(zip(cache.cls_i.tolist(), cache.cls_j.tolist()))
    overlap = merge_edges & cls_edges
    assert not overlap, f"{len(overlap)} edges appear in BOTH merged and classical transport"
    print(f"\n  classical links: {len(cache.cls_i)} verified, no overlap with merge pairs")


# ──────────────────────────────────────────────────────────────────────────────
# 4. Row-gain projection
# ──────────────────────────────────────────────────────────────────────────────

def test_row_gain_cap(pool):
    """Pre-cap row gains should be (mostly) below the cap — the projection is
    a stabilizer, not a load-bearing clamp. If many rows exceed it, divergence
    pressure has returned and the cap is silently eating energy again."""
    segs, cache = pool["segments"], pool["cache"]
    S = len(segs)
    lum_fr = _weighted_fr(pool) @ LUM
    row_gain = np.zeros(S)
    np.add.at(row_gain, cache.prop_i, lum_fr)
    over = row_gain > ROW_GAIN_CAP
    frac = over.mean()
    print(f"\n  row gains: max={row_gain.max():.3f}, "
          f"over-cap rows={int(over.sum())}/{S} ({100*frac:.2f}%)")
    assert frac < 0.05, (
        f"{100*frac:.1f}% of rows exceed the energy-conservation cap pre-"
        f"projection — the cap is doing clamp-like work again. Find the new "
        f"divergence source instead of relying on the projection.")


# ──────────────────────────────────────────────────────────────────────────────
# 6. Progressive schedule ownership
# ──────────────────────────────────────────────────────────────────────────────

def test_progressive_schedule_owned_by_caller(pool):
    cluster = Cluster(c=CLUSTER_C, alpha=0.25)
    cluster.set_scene_aabb(pool["scene"].bbox())
    cluster.set_segments(pool["segments"])
    cluster.cluster(np.random.default_rng(0))
    v0 = float(cluster.voxel_size)
    cluster.cluster(np.random.default_rng(1))   # no _iteration touch
    v1 = float(cluster.voxel_size)
    assert v0 == pytest.approx(v1), (
        "voxel_size changed across cluster() calls without the caller setting "
        "_iteration — hidden progressive shrinkage is back.")
    cluster._iteration = 8
    cluster.cluster(np.random.default_rng(2))
    assert float(cluster.voxel_size) < v0, (
        "caller-set _iteration did not shrink the kernel — progressive "
        "schedule broken in the other direction.")


# ──────────────────────────────────────────────────────────────────────────────
# 7. Split final gather
# ──────────────────────────────────────────────────────────────────────────────

def test_direct_split_readout(pool):
    """Run propagation, then check: direct_in is populated, direct ≤ total
    radiance elementwise (both are nonnegative sums and direct is a sub-sum
    of the final hop), and the depth-1 / depth-2 readouts agree in the mean
    (same estimand, different correlation structure)."""
    segs = pool["segments"]
    prop = Propagation(num_prop_iterations=4, row_gain_cap=ROW_GAIN_CAP,
                       merge_min_len_factor=MERGE_MIN_LEN_FACTOR)
    prop.iterate_propogation_vec(pool["cluster"], pool["cache"])

    for s in segs:
        d = getattr(s, "direct_in", None)
        assert d is not None, "direct_in missing after iterate_propogation_vec"
        for ch in range(3):
            assert float(d[ch]) <= float(s.radiance_in[ch]) + 1e-9, (
                "direct_in exceeds radiance_in — the direct sub-sum is not a "
                "subset of the final gather")

    # depth-1 vs depth-2 readout (inline mirror of renderer._final_gather)
    d1, d2 = [], []
    for fs in pool["camera_first"]:
        if fs is None:
            continue
        v1 = fs.throughput * fs.radiance_in * fs.mmis_weight
        d1.append(_lum(v1))
        second = getattr(fs, "_next_seg", None)
        if second is not None:
            if second.y.is_light or second.x.is_light:
                v2 = fs.throughput * fs.direct_in
            else:
                v2 = fs.throughput * (fs.direct_in
                                      + second.throughput * second.radiance_in)
        else:
            v2 = v1
        d2.append(_lum(v2))
    m1, m2 = float(np.mean(d1)), float(np.mean(d2))
    print(f"\n  readout means: depth-1 = {m1:.4e}, depth-2 = {m2:.4e}, "
          f"ratio = {m2/max(m1,1e-12):.3f}")
    assert m1 > 0 and m2 > 0
    # same estimand; generous band because this is one small iteration
    assert 0.6 < m2 / m1 < 1.6, (
        f"depth-2 readout mean is {m2/m1:.2f}× depth-1 — the split gather is "
        f"not estimating the same quantity (double-count or dropped term).")


def test_depth_d_readout(pool):
    """Generalized depth-d split readout (renderer._readout_path):
    depth=2 must reproduce the inline depth-2 formula exactly, and deeper
    splits must estimate the same quantity (band check, single iteration)."""
    from segments.renderer import _readout_path

    if getattr(pool["segments"][0], "direct_in", None) is None:
        # standalone run: populate radiance_in/direct_in first
        Propagation(num_prop_iterations=4, row_gain_cap=ROW_GAIN_CAP,
                    merge_min_len_factor=MERGE_MIN_LEN_FACTOR
                    ).iterate_propogation_vec(pool["cluster"], pool["cache"])

    means = {}
    for d in (2, 3, 4, 99):
        vals = []
        for fs in pool["camera_first"]:
            if fs is None:
                continue
            R = _readout_path(fs._path, d)
            v = fs.throughput * R
            for ch in range(3):
                assert np.isfinite(float(v[ch])) and float(v[ch]) >= 0.0
            vals.append(_lum(v))
        means[d] = float(np.mean(vals))

    # exact parity with the inline depth-2 mirror
    inline = []
    for fs in pool["camera_first"]:
        if fs is None:
            continue
        second = getattr(fs, "_next_seg", None)
        if second is not None:
            if second.y.is_light or second.x.is_light:
                v = fs.throughput * fs.direct_in
            else:
                v = fs.throughput * (fs.direct_in
                                     + second.throughput * second.radiance_in)
        else:
            v = fs.throughput * fs.radiance_in * fs.mmis_weight
        inline.append(_lum(v))
    assert means[2] == pytest.approx(float(np.mean(inline)), rel=1e-12), (
        "depth-2 readout no longer matches the original split-gather formula")

    print(f"\n  depth-d readout means: " +
          ", ".join(f"d={d}: {m:.4e}" for d, m in means.items()))
    for d in (3, 4, 99):
        assert 0.6 < means[d] / means[2] < 1.6, (
            f"depth-{d} readout mean is {means[d]/means[2]:.2f}× depth-2 — "
            f"split levels are double-counting or dropping a term.")
