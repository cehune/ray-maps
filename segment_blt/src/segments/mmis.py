from segments.cluster import Cluster
from segments.primitives import SegmentTechnique
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