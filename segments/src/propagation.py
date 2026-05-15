from segments.src.cluster import Cluster
from segments.src.primitives import SurfacePoint, Segment
from segments.src.primitives import SegmentTechnique
from segments.src.sampler_types import Sampler
import drjit as dr
import mitsuba as mi
import math

class Propagation:
    def __init__(self, kernel_radius, kernel_weight):
        self.kernel_radius = kernel_radius
        self.kernel_weight = kernel_weight
        self.update_kernel_val()

    def _update_kernel(self):
        self.kernel_radius *= self.kernel_weight
        self.update_kernel_val()

    def kernel_check_disc(self, seg_endpoint: SurfacePoint, next_endpoint: SurfacePoint):
        """
        More of a cylinder check than a disc, basically infinite cylinder
        ||d||^2 - (d dot n)^2 < r^2
        only measuring tangential distance, anywhere above or below though
        check with Wenyou
        """
        
        # Vector from center endpoint to candidate endpoint
        offset = next_endpoint.p - seg_endpoint.p

        # Component along the surface normal
        normal_dist = dr.dot(offset, seg_endpoint.n)

        # Remove the normal component to get tangent-plane projection
        tangent_offset = offset - normal_dist * seg_endpoint.n

        # Squared distance in tangent plane
        tangent_dist_sq = dr.squared_norm(tangent_offset)

        return tangent_dist_sq <= self.kernel_radius ** 2
    
    def update_kernel_val(self) -> float:
        """K = 1 / (pi * r^2) for the uniform disc kernel."""
        self.kernel_val = 1.0 / (math.pi * self.kernel_radius ** 2)

    def propagate_segment(self, segment: Segment, seg_idx: int, cluster: Cluster, samplers: dict[SegmentTechnique, 'Sampler']):
        out = mi.Color3f(0.0)

        # --- Technique space 1: CAMERA sampler ---
        y_cluster_idx = cluster.endpoint_to_cluster[seg_idx * 2 + 1]
        start, end = cluster.cluster_ranges[y_cluster_idx]
        for flat_idx in cluster.sorted_indices[start:end]:
            next_idx, which_end = cluster.endpoint_metadata[flat_idx]
            if next_idx == seg_idx:
                continue
            if which_end == 0:  # y-endpoint of seg is near the x of the next
                next_seg: Segment = cluster.segments[next_idx]
                if self.kernel_check_disc(segment.y, next_seg.x):
                    sampler = samplers[next_seg.technique]
                    if sampler.technique_type == SegmentTechnique.CAMERA:
                        f = sampler.shift_invariant_bsdf(segment, next_seg)
                        out += next_seg.mmis_weight * self.kernel_val * f * next_seg.geom_term * next_seg.radiance_in
        # TODO: Light

        return out
        
    def propagate_all_segments(self, cluster: Cluster, samplers):
        # should be called once per iteration
        for seg_idx, segment in enumerate(cluster.segments):
            if segment.x.is_camera or segment.x.is_light:
                continue
                
            segment.radiance_out = self.propagate_segment(segment, seg_idx, cluster, samplers)
        
        self._update_kernel()
        