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


def build_pair_cache(cluster, samplers) -> PairCache:
    """
    Build PairCache for the current clustering.

    Parameters
    ----------
    cluster       : Cluster (already clustered, reverse map built)
    samplers      : dict[SegmentTechnique, Sampler]
    """
    from segments.sampler_types import SequentialSampler

    segs = cluster.segments
    S = len(segs)
    if S == 0:
        return PairCache()

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
        # NOTE: no occupancy gate here. Skipping clusters with few endpoints
        # zeroes out both propagation AND mmis pairs for those segments, which
        # turns progressive radius shrinkage into coverage collapse (sparse
        # cluster = biased dark instead of merely noisy) and builds a bias floor
        # the iteration average can't remove. The len(y_segs)==0 / len(x_segs)==0
        # guard below is the only degenerate case we actually need to drop.
        meta = cluster.endpoint_metadata[flat_indices]  # (K, 2): (seg_idx, which_end)

        y_segs = meta[meta[:, 1] == 1, 0]  # segment indices whose y is in this cluster
        x_segs = meta[meta[:, 1] == 0, 0]  # segment indices whose x is in this cluster

        if len(y_segs) > 0:
            y_valid = np.array([
                not (segs[i].technique == SegmentTechnique.LIGHT and segs[i].y.is_light)
                for i in y_segs
            ], dtype=bool)
            y_segs = y_segs[y_valid]
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

        # MMIS pairs: cross-product x_segs × y_segs with filters
        J_grid, T_grid = np.meshgrid(x_segs, y_segs, indexing='ij')
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

    for k in range(M):
        j_idx = int(mmis_j_arr[k])
        t_idx = int(mmis_t_arr[k])
        seg_j = segs[j_idx]
        seg_t = segs[t_idx]
        sampler = samplers.get(seg_t.technique)   # auxiliary's technique
        if sampler is None:
            continue
        mmis_pdf[k] = sampler.conditional_pdf(seg_j, seg_t)    

    return PairCache(
        prop_i=prop_i_arr,
        prop_j=prop_j_arr,
        prop_fr=prop_fr,
        mmis_j=mmis_j_arr,
        mmis_t=mmis_t_arr,
        mmis_pdf=mmis_pdf,
        trivial_mmis=np.array(sorted(trivial_set), dtype=np.int32),
    )
