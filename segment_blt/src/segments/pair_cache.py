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

    # unconditional ray-tracing technique: one pdf term per segment, p_RT(s)=G(s)/(pi|M|)
    p_rt: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))

    # per-segment merge-weight scale: |M| (area-sample Jacobian 1/p(x)) for RT
    # segments, 1.0 otherwise. Confirmed by the white-furnace |M| sweep.
    rt_scale: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))

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
            # Bridge segments are technique=BRIDGE, already excluded by the
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
            # and bridge segments' y is a synthetic si (shape=None) whose
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

    # The bridge (to-light) technique is added ONCE PER MMIS PAIR,
    # independent of the auxiliary's own technique dispatch — it is a distinct
    # path-construction technique (paper §S2 / Eq. S8), not a continuation of
    # the auxiliary. BridgeSampler returns p_L(x') for a light-terminal seg_j
    # conditioned on a camera-side auxiliary, 0 otherwise. In the camera-only
    # limit this reproduces the old per-auxiliary `+ p_bridge` denominator exactly.
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

    # RT is counted per-auxiliary inside mmis_pdf above (an RT auxiliary
    # dispatches to RayTracingSampler), identical to every other technique. Its
    # pdf is the LOCAL directional density G/π — the global 1/|M| does NOT belong
    # in the per-bounce merge operator (it inflated the gain and compounded
    # through indirect bounces; see the cbox prop-iters sweep).
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


