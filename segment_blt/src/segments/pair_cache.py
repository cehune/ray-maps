"""
PairCache: precompute all valid segment pairs once per render iteration.

After clustering, the set of (i, j) pairs is stable — same cluster assignment,
same kernel disc membership — for the entire iteration. We:

  1. Build all candidate (i, j) pairs via cross-product within each cluster
     (i has y-endpoint in cluster, j has x-endpoint in cluster, i ≠ j).
  2. Vectorize the kernel disc check in numpy to filter propagation pairs.
  3. For the surviving pairs, call shift_invariant_bsdf / conditional_pdf once
     and cache the results.

Propagation then loops over num_prop_iterations as pure numpy (no BSDF calls).
MMIS is a single np.add.at + inversion.

Total savings: O(num_prop_iterations) factor on BSDF calls for propagation,
plus elimination of Python loop overhead from the inner propagation pass.


"""

from dataclasses import dataclass, field
import numpy as np
from segments.primitives import SegmentTechnique


@dataclass
class PairCache:
    # propagation pairs: segment i receives from segment j
    # (i's y-endpoint is near j's x-endpoint in the same cluster, kernel disc passed)
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
    kernel_radius : current kernel radius — used for disc check
    samplers      : dict[SegmentTechnique, Sampler]
    """
    from segments.sampler_types import SequentialSampler

    segs = cluster.segments
    S = len(segs)
    if S == 0:
        return PairCache()

    # extract per-segment numpy arrays for vectorized disc check 
    # so it's flattened like [x, y, z] [x, y, z]
    x_pos = np.array([[float(s.x.p.x), float(s.x.p.y), float(s.x.p.z)] for s in segs],
                     dtype=np.float64)  # (S, 3)
    y_pos = np.array([[float(s.y.p.x), float(s.y.p.y), float(s.y.p.z)] for s in segs],
                     dtype=np.float64)  # (S, 3)
    y_nor = np.array([[float(s.y.n.x), float(s.y.n.y), float(s.y.n.z)] for s in segs],
                     dtype=np.float64)  # (S, 3)

    prop_i_list: list[int] = []
    prop_j_list: list[int] = []
    mmis_j_list: list[int] = []
    mmis_t_list: list[int] = []
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
    for seg_idx, seg in enumerate(segs):
        if seg.x.is_camera or seg.x.is_light:
            trivial_set.add(seg_idx)

    for _, (start, end) in enumerate(cluster.cluster_ranges):
        flat_indices = cluster.sorted_indices[start:end]
        meta = cluster.endpoint_metadata[flat_indices]  # (K, 2): (seg_idx, which_end)

        y_segs = meta[meta[:, 1] == 1, 0]  # segment indices whose y is in this cluster
        x_segs = meta[meta[:, 1] == 0, 0]  # segment indices whose x is in this cluster

        if len(y_segs) == 0 or len(x_segs) == 0:
            continue

        # propagation pairs (disc-checked)
        # all (i in y_segs, j in x_segs, i≠j) as a cross-product, then vectorized check.
        I, J = np.meshgrid(y_segs, x_segs, indexing='ij')  # (|y|, |x|) each
        I, J = I.ravel(), J.ravel()
        neq = I != J
        I, J = I[neq], J[neq]

        # TODO: potentially remove if disc doesn't work nice
        if len(I) > 0:
            prop_i_list.extend(I.tolist())
            prop_j_list.extend(J.tolist())

        # mmis pairs (no disc check — just cluster membership)
        # for each j (x in cluster), t (y in cluster) is a potential auxiliary.
        for j_idx in x_segs:
            seg_j = segs[j_idx]
            # trivial segments were already added in the pre-pass; skip them here.
            if int(j_idx) in trivial_set:
                continue
            sampler_j = samplers.get(seg_j.technique)
            if sampler_j is None:
                continue
            for t_idx in y_segs:
                if t_idx == j_idx:
                    continue
                seg_t = segs[t_idx]
                sampler_t = samplers.get(seg_t.technique)
                if sampler_t is None or sampler_t.technique_type != SegmentTechnique.CAMERA:
                    continue
                mmis_j_list.append(int(j_idx))
                mmis_t_list.append(int(t_idx))

    # evvaluate BSDF for propagation pairs (once; reused across all prop iters) ---
    P = len(prop_i_list)
    prop_fr = np.zeros((P, 3), dtype=np.float64)
    valid_prop = np.ones(P, dtype=bool)

    for k, (i, j) in enumerate(zip(prop_i_list, prop_j_list)):
        seg_i, seg_j = segs[i], segs[j]
        sampler = samplers.get(seg_j.technique)
        if sampler is None or sampler.technique_type != SegmentTechnique.CAMERA:
            valid_prop[k] = False
            continue
        fr = SequentialSampler.shift_invariant_bsdf(seg_i, seg_j)
        prop_fr[k] = [float(fr.x), float(fr.y), float(fr.z)]

    # filter out any zero-technique pairs ( should be none)
    if not np.all(valid_prop):
        prop_i_arr  = np.array(prop_i_list, dtype=np.int32)[valid_prop]
        prop_j_arr  = np.array(prop_j_list, dtype=np.int32)[valid_prop]
        prop_fr     = prop_fr[valid_prop]
    else:
        prop_i_arr = np.array(prop_i_list, dtype=np.int32) if P > 0 else np.empty(0, np.int32)
        prop_j_arr = np.array(prop_j_list, dtype=np.int32) if P > 0 else np.empty(0, np.int32)

    # evaluate conditional PDF for MMIS pairs (only once per propagation iter!!!!)
    M = len(mmis_j_list)
    mmis_pdf = np.zeros(M, dtype=np.float64)

    for k, (j, t) in enumerate(zip(mmis_j_list, mmis_t_list)):
        seg_j = segs[j]
        sampler_j = samplers.get(seg_j.technique)
        if sampler_j is None:
            continue
        mmis_pdf[k] = sampler_j.conditional_pdf(seg_j, segs[t])

    return PairCache(
        prop_i=prop_i_arr,
        prop_j=prop_j_arr,
        prop_fr=prop_fr,
        mmis_j=np.array(mmis_j_list, dtype=np.int32) if M > 0 else np.empty(0, np.int32),
        mmis_t=np.array(mmis_t_list, dtype=np.int32) if M > 0 else np.empty(0, np.int32),
        mmis_pdf=mmis_pdf,
        trivial_mmis=np.array(sorted(trivial_set), dtype=np.int32),
    )
