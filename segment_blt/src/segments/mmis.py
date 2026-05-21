from segments.cluster import Cluster
from segments.primitives import SegmentTechnique
import numpy as np

class MMIS:
    """
    What are technique spaces? we have 2, fromt he camera and 
    from the lights. With multiple techniques that can generate the same
    segment, we need to evaluate the probability and weighting for all techniques
    for the likelihood of generating that segment. 

    So we give proportional credit based on how likely each technique made a 
    sample, and these must sum to 1. We find this with a PDF for each. This pdf is 
    reliant both on the segment and an auxilliary (pre-ceding segment)

    over all segments currently in this cluster that could plausibly have been the 
    preceding segment, how much total probability mass do they collectively assign to s′?
    But what does it mean to be a preceding segment? Both has to be in kernel support within 
    the cluster

    directionality is controlled byt he PDF. based on sequential model, we always start at 
    y, then check for x points within the cluster.class MMIS:
    
    Technique spaces: for unidirectional, one technique (camera sequential sampler).
    For each segment s, we ask: over all auxiliaries t in the cluster at x_s whose
    y_t is in that cluster, what is the total conditional probability mass on s? So basically
    if it is likely to have a segment as a continuiation from another segment, then 

    Weight = 1 / sum of conditional PDFs over all valid auxiliaries.
    Auxiliary validity: y_t must be in the same cluster as x_s (spatial),
    directionality handled naturally by the BSDF PDF being near zero for bad directions.
    """


    """
    should own
    the summation loop, the 1/p_sum inversion, the cluster lookup — the algorithm structure that is sampler-agnostic"""

    def compute_mmis_weight(self, segment, main_seg_idx, cluster: Cluster, samplers) -> float:
        """
        For unidirectional: sum conditional PDFs over all segments in cluster,
        treating each as a potential auxiliary (y_t -> x_s relationship).
        Returns 1/p_sum.
        """
        p_sum = 0.0

        # --- Technique space 1: CAMERA sampler ---
        # s' can be a continuation of a camera segment t where y_t ≈ x_{s'}
        x_cluster_idx = cluster.endpoint_to_cluster[main_seg_idx * 2 + 0]
        start, end = cluster.cluster_ranges[x_cluster_idx]
        for flat_idx in cluster.sorted_indices[start:end]:
            aux_seg_idx, which_end = cluster.endpoint_metadata[flat_idx]
            if which_end == 1:  # y-endpoint of t is in this cluster
                t = cluster.segments[aux_seg_idx]
                sampler = samplers[t.technique]
                if sampler.technique_type == SegmentTechnique.CAMERA:
                    p_sum += sampler.conditional_pdf(segment, t)

        # TODO: light sampling

        return 1.0 / p_sum if p_sum > 0.0 else 0.0

    def compute_all_mmis_weights(self, cluster, samplers) -> None:
        for seg_idx, segment in enumerate(cluster.segments):
            if segment.x.is_camera or segment.x.is_light:
                segment.mmis_weight = 1.0
                continue
            segment.mmis_weight = self.compute_mmis_weight(segment, seg_idx, cluster, samplers)

    def compute_all_mmis_weights_vec(self, cluster, pair_cache) -> None:
        """
        Vectorized MMIS weight computation using a prebuilt PairCache.

        All conditional PDF values are already cached in pair_cache.mmis_pdf.
        This method just sums them per segment and inverts — pure numpy, no BSDF calls.
        """
        S = len(cluster.segments)
        p_sum = np.zeros(S, dtype=np.float64)

        # trivial segments (camera-origin or light-origin x) get weight 1.0 directly
        p_sum[pair_cache.trivial_mmis] = 1.0   # sentinel: will be inverted to 1.0

        # accumulate conditional PDFs for non-trivial segments
        if len(pair_cache.mmis_j) > 0:
            np.add.at(p_sum, pair_cache.mmis_j, pair_cache.mmis_pdf)

        # invert: w = 1/p_sum (0 where no PDF mass — segment unreachable)
        nz = p_sum > 0.0
        weights = np.where(nz, 1.0 / p_sum, 0.0)

        # Trivial segments: p_sum was set to 1.0 → inverts to 1.0 
        for seg_idx, seg in enumerate(cluster.segments):
            seg.mmis_weight = float(weights[seg_idx])
        