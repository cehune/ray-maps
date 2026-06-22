"""
PairCache: precompute all valid segment pairs once per render iteration.

After clustering, the set of (i, j) pairs is stable — same cluster assignment,
 — for the entire iteration. We:

  1. Build all candidate (i, j) pairs via cross-product within each cluster
     (i has y-endpoint in cluster, j has x-endpoint in cluster, i ≠ j).
  2. For the surviving pairs, call shift_invariant_bsdf / conditional_pdf once
     and cache the results.

Propagation then loops over num_prop_iterations as pure numpy (no BSDF calls).
MMIS is a single np.add.at + inversion.

Total savings: O(num_prop_iterations) factor on BSDF calls for propagation,
plus elimination of Python loop overhead from the inner propagation pass.

for the record, kernel is only enforced by voxel.
"""

from dataclasses import dataclass, field
import numpy as np
from segments.primitives import SegmentTechnique


@dataclass
class PairCache:
    # propagation pairs: segment i receives from segment j
    # (i's y-endpoint is near j's x-endpoint in the same cluster)
    prop_i: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    prop_j: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    prop_fr: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float64))

    # mmis pairs: when computing the weight for segment j, auxiliary t contributes p(j|t)
    # (t's y-endpoint is in the same cluster as j's x-endpoint)
    mmis_j: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    mmis_t: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    mmis_pdf: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))

    # segments whose MMIS weight is trivially 1.0 (x.is_camera or x.is_light)
    trivial_mmis: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))

    # classical (non-merged) links for sub-kernel segments: parent cls_i
    # receives child cls_j's radiance through the EXACT BSDF-sampled link,
    # with weight cls_w = child.throughput (= f·cosθ/p_ω, bounded by albedo).
    # Short segments are excluded from kernel merging (their atom/pdf mismatch
    # is the corner-firefly bias), but their transport is real — ceiling↔light
    # gaps, box↔floor contact, corners. The classical link estimates exactly
    # the near-field part of the parent's gather integral, unbiased, with the
    # 1/len² singularity cancelled by the BSDF sample's own pdf.
    cls_i: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    cls_j: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    cls_w: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float64))


def build_pair_cache(cluster, samplers, merge_min_len_factor: float | None = 1.0,
                     pool_mask=None) -> PairCache:
    """
    Build PairCache for the current clustering.

    Parameters
    ----------
    cluster       : Cluster (already clustered, reverse map built)
    samplers      : dict[SegmentTechnique, Sampler]
    merge_min_len_factor : exclude segments with len < factor * voxel_size
        from acting as merge SOURCES (prop_j and MMIS receivers). The
        virtual-perturbation pdf prices x' as Uniform(voxel) but the sample
        sits exactly at the parent's endpoint; for len << voxel the integrand
        G is evaluated at its spike, overestimating transfer by
        ~(voxel/len)^2 — the corner-firefly bias. Sub-kernel segments still
        RECEIVE radiance and still act as MMIS auxiliaries (p(j|t) does not
        depend on t's length). Bias from the exclusion -> 0 as the
        progressive kernel shrinks. None disables.
    pool_mask : optional (S,) bool array — DIAGNOSTIC (hop-pool decorrelation).
        When given, only masked segments may act as merge SOURCES (prop_j,
        classical children) or as MMIS AUXILIARIES (mmis_t); receivers stay
        unrestricted. Restricting the auxiliary set to the same sub-pool as
        the sources keeps the estimator unbiased (the denominator then counts
        exactly the techniques active in this sub-pool — weights scale up by
        the pool fraction automatically). The mask MUST keep parent/child on
        the same side (split by path!) or the parent term vanishes from
        p_sum and the w·G ≤ π bound breaks.
    """
    from segments.sampler_types import SequentialSampler

    segs = cluster.segments
    S = len(segs)
    if S == 0:
        return PairCache()

    # ── Sub-kernel exclusion mask ─────────────────────────────────────────────
    if merge_min_len_factor is not None and float(cluster.voxel_size) > 0:
        min_len = float(merge_min_len_factor) * float(cluster.voxel_size)
        mergeable = np.array([float(s.len) >= min_len for s in segs], dtype=bool)
        n_excl = int((~mergeable).sum())
        if n_excl > 0:
            print(f"  [sub-kernel exclusion] {n_excl}/{S} segments "
                  f"({100*n_excl/S:.1f}%) shorter than {min_len:.3e} excluded as merge sources")
    else:
        mergeable = np.ones(S, dtype=bool)

    if pool_mask is None:
        pool_mask = np.ones(S, dtype=bool)
    else:
        pool_mask = np.asarray(pool_mask, dtype=bool)
        assert pool_mask.shape == (S,), "pool_mask must be (S,) bool"
    source_ok = mergeable & pool_mask   # merge sources: mergeable AND in pool

    # ── Classical links for the excluded segments ─────────────────────────────
    # parent is identified by object identity: child.x IS parent.y (same
    # SurfacePoint instance along a sampled path).
    cls_i_list: list[int] = []
    cls_j_list: list[int] = []
    cls_w_list: list[list[float]] = []
    if (~mergeable & pool_mask).any():
        y_owner = {id(s.y): i for i, s in enumerate(segs)}
        for j in np.where(~mergeable & pool_mask)[0]:
            sj = segs[int(j)]
            # TODO: light-technique parentage is flipped (parent.x == child.y);
            # add when the bidirectional path is repaired.
            # Bridge (NEE) segments are technique=BRIDGE, already excluded by the
            # CAMERA check below — they are not classical near-field children
            # (their near-field stratum is the bridge itself).
            if sj.technique != SegmentTechnique.CAMERA or sj.x.is_camera:
                continue
            p = y_owner.get(id(sj.x))
            if p is None or p == int(j):
                continue
            t = sj.throughput
            cls_i_list.append(int(p))
            cls_j_list.append(int(j))
            cls_w_list.append([float(t.x), float(t.y), float(t.z)])
        if cls_i_list:
            print(f"  [classical links] {len(cls_i_list)} sub-kernel segments "
                  f"routed to their parents via exact connections")

    prop_i_list: list[int] = []
    prop_j_list: list[int] = []
    mmis_j_chunks: list[np.ndarray] = []
    mmis_t_chunks: list[np.ndarray] = []
    trivial_set: set[int] = set()

    # PRE-PASS: mark camera-origin and light-origin segments as trivial before
    # the cluster loop.  These always get mmis_weight=1.0 regardless of cluster
    # structure.  We cannot rely on the cluster loop body for this because a
    # camera-origin segment's x-cluster (the camera position) typically has no
    # y-endpoints, causing the early-continue guard to fire and skip the body —
    # leaving the segment absent from trivial_set with mmis_weight=0.
    #
    # Light-terminal segments (y.is_light) are intentionally NOT pre-marked here:
    # they should accumulate conditional PDFs from auxiliaries in their cluster
    # and receive a proper MMIS weight (1/p_sum), not be pinned to 1.0.
    for seg_idx, segment in enumerate(segs):
        if segment.x.is_camera or (
                segment.technique == SegmentTechnique.LIGHT and segment.y.is_light
            ):
            trivial_set.add(seg_idx)

    # boolean lookups over all segments — built once, reused per cluster
    trivial_mask = np.zeros(S, dtype=bool)
    if trivial_set:
        trivial_mask[np.fromiter(trivial_set, dtype=np.int32)] = True

    has_sampler = np.array(
        [samplers.get(seg.technique) is not None for seg in segs],
        dtype=bool,
    )

    for _, (start, end) in enumerate(cluster.cluster_ranges):
        flat_indices = cluster.sorted_indices[start:end]
        # A cluster needs at least one x- and one y-endpoint to form any pair;
        # below 2 endpoints nothing can happen, so skipping is a pure no-op.
        # (The old `<= 5` cutoff silently deleted ALL pairs in small clusters —
        # systematic energy loss that grew as the progressive kernel shrank.)
        if len(flat_indices) < 2:
            continue
        meta = cluster.endpoint_metadata[flat_indices]  # (K, 2): (seg_idx, which_end)

        y_segs = meta[meta[:, 1] == 1, 0]  # segment indices whose y is in this cluster
        x_segs = meta[meta[:, 1] == 0, 0]  # segment indices whose x is in this cluster

        # sub-kernel / out-of-pool segments cannot be merge sources (see docstring)
        if len(x_segs) > 0:
            x_segs = x_segs[source_ok[x_segs]]

        if len(y_segs) > 0:
            # Light-terminal segments are never receivers: their radiance is
            # pinned to Le every hop, so gathering into them is wasted work —
            # and NEE segments' y is a synthetic si (shape=None) whose
            # .bsdf() call segfaults mitsuba 3.8 scalar. As MMIS auxiliaries
            # they were already zeroed by the aux.y.is_light guard, so this
            # filter is output-identical.
            y_valid = np.array([
                not segs[i].y.is_light for i in y_segs
            ], dtype=bool)
            y_segs = y_segs[y_valid]
        # MMIS auxiliaries are restricted to the pool (sources and their
        # denominators must describe the same technique set); propagation
        # receivers (y_segs) stay unrestricted.
        y_aux = y_segs[pool_mask[y_segs]] if len(y_segs) > 0 else y_segs
        # ─────────────────────────────────────────────────────────────────────

        if len(y_segs) == 0 or len(x_segs) == 0:
            continue

        # propagation pairs
        # all (i in y_segs, j in x_segs, i≠j) as a cross-product, then vectorized check.
        I, J = np.meshgrid(y_segs, x_segs, indexing='ij')  # (|y|, |x|) each
        I, J = I.ravel(), J.ravel()
        neq = I != J
        I, J = I[neq], J[neq]

        if len(I) > 0:
            prop_i_list.extend(I.tolist())
            prop_j_list.extend(J.tolist())

        # MMIS pairs: cross-product x_segs × y_aux with filters
        if len(y_aux) == 0:
            continue
        J_grid, T_grid = np.meshgrid(x_segs, y_aux, indexing='ij')
        J_flat = J_grid.ravel()
        T_flat = T_grid.ravel()

        keep = (
            (J_flat != T_flat)        # self-exclusion
            & ~trivial_mask[J_flat]   # skip trivial j (their weight is pinned to 1)
            & has_sampler[J_flat]     # j must have a sampler (for conditional_pdf call later)
            & has_sampler[T_flat]     # t must have a sampler
        )

        if keep.any():
            mmis_j_chunks.append(J_flat[keep])
            mmis_t_chunks.append(T_flat[keep])

    # evvaluate BSDF for propagation pairs (once; reused across all prop iters) ---
    P = len(prop_i_list)
    prop_fr = np.zeros((P, 3), dtype=np.float64)
    for k, (i, j) in enumerate(zip(prop_i_list, prop_j_list)):
        seg_i, seg_j = segs[i], segs[j]

        # TODO: APPARENTLY ITS ALWAYS RADIANCE MODE
        sampler = samplers.get(SegmentTechnique.CAMERA)
        if sampler is None:
            continue
        fr = sampler.shift_invariant_bsdf(seg_i, seg_j)
        prop_fr[k] = [float(fr.x), float(fr.y), float(fr.z)]

    prop_i_arr = np.array(prop_i_list, dtype=np.int32) if P > 0 else np.empty(0, np.int32)
    prop_j_arr = np.array(prop_j_list, dtype=np.int32) if P > 0 else np.empty(0, np.int32)

    # evaluate conditional PDF for MMIS pairs (only once per propagation iter!!!!)
    mmis_j_arr = (np.concatenate(mmis_j_chunks).astype(np.int32)
              if mmis_j_chunks else np.empty(0, np.int32))
    mmis_t_arr = (np.concatenate(mmis_t_chunks).astype(np.int32)
                if mmis_t_chunks else np.empty(0, np.int32))
                
    M = len(mmis_j_arr)
    mmis_pdf = np.zeros(M, dtype=np.float64)

    # The bridge (NEE / to-light) technique is added ONCE PER MMIS PAIR,
    # independent of the auxiliary's own technique dispatch — it is a distinct
    # path-construction technique (paper §S2 / Eq. S8), not a continuation of
    # the auxiliary. BridgeSampler returns p_L(x') for a light-terminal seg_j
    # conditioned on a camera-side auxiliary, 0 otherwise. In the camera-only
    # limit this reproduces the old per-auxiliary `+ p_nee` denominator exactly.
    bridge_sampler = samplers.get(SegmentTechnique.BRIDGE)
    for k in range(M):
        j_idx = int(mmis_j_arr[k])
        t_idx = int(mmis_t_arr[k])
        seg_j = segs[j_idx]
        seg_t = segs[t_idx]
        p = 0.0
        sampler = samplers.get(seg_t.technique)   # auxiliary's technique → sequential pdf
        if sampler is not None:
            p += sampler.conditional_pdf(seg_j, seg_t)
        if bridge_sampler is not None:
            p += bridge_sampler.conditional_pdf(seg_j, seg_t)
        mmis_pdf[k] = p

    return PairCache(
        prop_i=prop_i_arr,
        prop_j=prop_j_arr,
        prop_fr=prop_fr,
        mmis_j=mmis_j_arr,
        mmis_t=mmis_t_arr,
        mmis_pdf=mmis_pdf,
        trivial_mmis=np.array(sorted(trivial_set), dtype=np.int32),
        cls_i=np.array(cls_i_list, dtype=np.int32) if cls_i_list else np.empty(0, np.int32),
        cls_j=np.array(cls_j_list, dtype=np.int32) if cls_j_list else np.empty(0, np.int32),
        cls_w=(np.array(cls_w_list, dtype=np.float64)
               if cls_w_list else np.empty((0, 3), np.float64)),
    )


def build_pair_cache_exact(cluster, samplers, r: float,
                           block_size: int = 256,
                           merge_min_len_factor: float | None = 1.0) -> PairCache:
    """
    DIAGNOSTIC: like build_pair_cache, but uses an EXACT Euclidean-ball
    neighborhood of radius r instead of voxel-bucket clustering.

    A propagation pair (i, j) is included iff ||y_i - x_j|| < r.
    A MMIS pair       (j, t) is included iff ||x_j - y_t|| < r.

    This removes two confounds from the voxel-hash version:
      1. The geometric mismatch — an axis-aligned voxel of side r is neither
         a ball of radius r nor centered on the query point. Two endpoints
         0.1·r apart can land in different voxels and never form a pair;
         two endpoints 1.7·r apart in opposite voxel corners CAN.
      2. The "skip clusters with ≤5 endpoints" early-continue in the
         voxel-hash builder — exact-radius has no notion of cluster size.

    O(S²) in distance checks (blocked for memory). Slow but exact.
    Use for diagnostic purposes only, with small images (~50×50 fine; 200×200
    will take minutes per iteration).

    Parameters
    ----------
    cluster   : Cluster (only segments are read; voxel ranges are unused)
    samplers  : dict[SegmentTechnique, Sampler]
    r         : float — radius of the exact ball neighborhood
    block_size: rows of the distance matrix to materialize at a time
    """
    from segments.sampler_types import SequentialSampler

    segs = cluster.segments
    S = len(segs)
    if S == 0:
        return PairCache()

    # ── Extract endpoint positions as (S, 3) numpy arrays ────────────────────
    x_pos = np.array(
        [[float(s.x.p.x), float(s.x.p.y), float(s.x.p.z)] for s in segs],
        dtype=np.float64,
    )
    y_pos = np.array(
        [[float(s.y.p.x), float(s.y.p.y), float(s.y.p.z)] for s in segs],
        dtype=np.float64,
    )

    # ── Trivial-segment marking — exact mirror of build_pair_cache pre-pass ──
    trivial_set: set[int] = set()
    for seg_idx, segment in enumerate(segs):
        if segment.x.is_camera or (
            segment.technique == SegmentTechnique.LIGHT and segment.y.is_light
        ):
            trivial_set.add(seg_idx)
    trivial_mask = np.zeros(S, dtype=bool)
    if trivial_set:
        trivial_mask[np.fromiter(trivial_set, dtype=np.int32)] = True

    has_sampler = np.array(
        [samplers.get(seg.technique) is not None for seg in segs],
        dtype=bool,
    )

    # i's y-endpoint can only be a propagation SOURCE if it's not a light-
    # terminal segment (same filter the voxel version applies to y_segs).
    valid_y_source = np.array(
        [
            not (segs[i].technique == SegmentTechnique.LIGHT and segs[i].y.is_light)
            for i in range(S)
        ],
        dtype=bool,
    )

    r2 = float(r) * float(r)

    # sub-kernel exclusion — here the kernel scale is the exact ball radius r
    if merge_min_len_factor is not None:
        min_len = float(merge_min_len_factor) * float(r)
        mergeable = np.array([float(s.len) >= min_len for s in segs], dtype=bool)
    else:
        mergeable = np.ones(S, dtype=bool)

    # ── Propagation pairs: (i, j) iff ||y_i - x_j|| < r ──────────────────────
    prop_i_list: list[int] = []
    prop_j_list: list[int] = []

    valid_i_idx = np.where(valid_y_source)[0]
    for start in range(0, len(valid_i_idx), block_size):
        end = min(start + block_size, len(valid_i_idx))
        i_block = valid_i_idx[start:end]                       # (b,)
        y_block = y_pos[i_block]                               # (b, 3)
        diff = y_block[:, None, :] - x_pos[None, :, :]         # (b, S, 3)
        d2 = (diff * diff).sum(-1)                             # (b, S)
        mask = (d2 < r2) & mergeable[None, :]                  # (b, S)
        # exclude i == j
        for local_k, i_global in enumerate(i_block):
            mask[local_k, i_global] = False
        rows, cols = np.where(mask)
        for local_k, j in zip(rows.tolist(), cols.tolist()):
            prop_i_list.append(int(i_block[local_k]))
            prop_j_list.append(int(j))

    # ── MMIS pairs: (j, t) iff ||x_j - y_t|| < r ─────────────────────────────
    mmis_j_list: list[int] = []
    mmis_t_list: list[int] = []

    # j is a valid receiver if it has a sampler, isn't trivial, and is a
    # mergeable source (sub-kernel segments never act as sources, so their
    # MMIS weight is unused — skip computing it, mirroring the voxel builder).
    valid_j_idx = np.where(has_sampler & ~trivial_mask & mergeable)[0]
    for start in range(0, len(valid_j_idx), block_size):
        end = min(start + block_size, len(valid_j_idx))
        j_block = valid_j_idx[start:end]
        x_block = x_pos[j_block]
        diff = x_block[:, None, :] - y_pos[None, :, :]
        d2 = (diff * diff).sum(-1)
        mask = (d2 < r2) & has_sampler[None, :]
        for local_k, j_global in enumerate(j_block):
            mask[local_k, j_global] = False
        rows, cols = np.where(mask)
        for local_k, t in zip(rows.tolist(), cols.tolist()):
            mmis_j_list.append(int(j_block[local_k]))
            mmis_t_list.append(int(t))

    # ── BSDF cache for propagation pairs (same shape as build_pair_cache) ────
    P = len(prop_i_list)
    prop_fr = np.zeros((P, 3), dtype=np.float64)
    if P > 0:
        sampler = samplers.get(SegmentTechnique.CAMERA)
        if sampler is not None:
            for k, (i, j) in enumerate(zip(prop_i_list, prop_j_list)):
                fr = sampler.shift_invariant_bsdf(segs[i], segs[j])
                prop_fr[k] = [float(fr.x), float(fr.y), float(fr.z)]

    # ── Conditional-PDF cache for MMIS pairs ─────────────────────────────────
    M = len(mmis_j_list)
    mmis_pdf = np.zeros(M, dtype=np.float64)
    # bridge (NEE / to-light) technique added per pair — see build_pair_cache.
    bridge_sampler = samplers.get(SegmentTechnique.BRIDGE)
    for k in range(M):
        j_idx = mmis_j_list[k]
        t_idx = mmis_t_list[k]
        seg_j = segs[j_idx]
        seg_t = segs[t_idx]
        p = 0.0
        sampler = samplers.get(seg_t.technique)
        if sampler is not None:
            p += sampler.conditional_pdf(seg_j, seg_t)
        if bridge_sampler is not None:
            p += bridge_sampler.conditional_pdf(seg_j, seg_t)
        mmis_pdf[k] = p

    return PairCache(
        prop_i=np.array(prop_i_list, dtype=np.int32) if P > 0 else np.empty(0, np.int32),
        prop_j=np.array(prop_j_list, dtype=np.int32) if P > 0 else np.empty(0, np.int32),
        prop_fr=prop_fr,
        mmis_j=np.array(mmis_j_list, dtype=np.int32) if M > 0 else np.empty(0, np.int32),
        mmis_t=np.array(mmis_t_list, dtype=np.int32) if M > 0 else np.empty(0, np.int32),
        mmis_pdf=mmis_pdf,
        trivial_mmis=np.array(sorted(trivial_set), dtype=np.int32),
    )
