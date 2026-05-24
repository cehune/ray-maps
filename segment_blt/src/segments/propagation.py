from segments.cluster import Cluster
from segments.primitives import SurfacePoint, Segment
from segments.primitives import SegmentTechnique
from segments.sampler_types import Sampler
import drjit as dr
import mitsuba as mi
import math
from segments.mmis import MMIS

class Propagation:
    def __init__(self, kernel_radius, kernel_weight, num_prop_iterations = 12):
        self.initial_radius = kernel_radius
        self.kernel_radius = kernel_radius
        self.kernel_weight = kernel_weight   # alpha in Knaus-Zwicker: 0.67
        self.num_prop_iterations = num_prop_iterations
        self._iteration = 1                  # current render iteration (1-indexed)
        self.update_kernel_val()

    def _update_kernel(self):
        # Power-law decay: r_n = r_0 * n^(-alpha/2)  (Knaus-Zwicker 2011, Eq. 5)
        # Exponential decay (r *= weight) collapses the kernel to zero by ~64 iterations.
        self._iteration += 1
        self.kernel_radius = self.initial_radius * (self._iteration ** (-self.kernel_weight / 2.0))
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
                if next_seg.mmis_weight == 0.0:  # guard before kernel check
                    continue
                if not self.kernel_check_disc(segment.y, next_seg.x):
                    continue
                sampler = samplers.get(next_seg.technique)
                if sampler is None:
                    continue
                if sampler.technique_type == SegmentTechnique.CAMERA:
                    f = sampler.shift_invariant_bsdf(segment, next_seg)
                    contrib = next_seg.mmis_weight * f * next_seg.geom_term * next_seg.radiance_in
                    out += contrib
        # TODO: Light
        return out
        
    def propagate_all_segments(self, cluster: Cluster, samplers):
        # should be called once per iteration
        for seg_idx, segment in enumerate(cluster.segments):                
            segment.radiance_out = self.propagate_segment(segment, seg_idx, cluster, samplers)
        
    def swap_segment_radiance(self, cluster: Cluster):
        for segment in cluster.segments:
            if segment.y.is_light:
                segment.radiance_in = segment.Le  # pin — never overwrite with radiance_out
            else:
                segment.radiance_in = segment.radiance_out
            segment.radiance_out = mi.Color3f(0.0)

    def iterate_propogation(self, cluster: Cluster, samplers: dict[SegmentTechnique, 'Sampler']):
        for segment in cluster.segments:
            segment.radiance_in = segment.Le

        for _ in range(self.num_prop_iterations):
            self.propagate_all_segments(cluster, samplers)
            self.swap_segment_radiance(cluster)

        self._update_kernel()

    def iterate_propogation_vec(self, cluster: Cluster, pair_cache):
        """
        Vectorized propagation using a prebuilt PairCache.

        BSDF values (prop_fr) are already cached.  Each of the num_prop_iterations
        passes is pure numpy: one gather of radiance_in + np.add.at accumulation.
        No Mitsuba/drjit calls happen inside this method.
        """
        import numpy as np

        segs = cluster.segments
        S = len(segs)

        # Initial radiance_in = Le for all segments
        Le = np.array([[float(s.Le.x), float(s.Le.y), float(s.Le.z)] for s in segs],
                      dtype=np.float64)  # (S, 3)
        is_light = np.array([s.y.is_light for s in segs], dtype=bool)  # (S,)

        rad_in = Le.copy()

        has_pairs = len(pair_cache.prop_i) > 0

        if has_pairs:
            j = pair_cache.prop_j  # (P,) source segment indices

            # Extract per-segment scalars once — static across all propagation iterations
            mmis_w = np.array([float(s.mmis_weight) for s in segs], dtype=np.float64)
            geom   = np.array([float(s.geom_term)   for s in segs], dtype=np.float64)

            # Pre-multiply BSDF with static mmis_weight and geom_term for each pair:
            # weighted_fr[k] = prop_fr[k] * mmis_w[j[k]] * geom[j[k]]   shape (P, 3)
            weighted_fr = pair_cache.prop_fr * (mmis_w[j] * geom[j])[:, np.newaxis]

        for _ in range(self.num_prop_iterations):
            rad_out = np.zeros((S, 3), dtype=np.float64)

            if has_pairs:
                # contrib[k] = weighted_fr[k] * rad_in[j[k]]   shape (P, 3)
                contrib = weighted_fr * rad_in[pair_cache.prop_j]
                np.add.at(rad_out, pair_cache.prop_i, contrib)

            # Swap: light endpoints pin to Le; others advance to rad_out
            rad_in = np.where(is_light[:, np.newaxis], Le, rad_out)

        # Write final radiance_in back to segment objects
        for seg_idx, seg in enumerate(segs):
            seg.radiance_in = mi.Color3f(
                float(rad_in[seg_idx, 0]),
                float(rad_in[seg_idx, 1]),
                float(rad_in[seg_idx, 2]),
            )

        self._update_kernel()
        # how much radiance is contributed per hop
        print(f"expected gain per hop (pairs * weighted_fr): {(len(pair_cache.prop_i) / len(segs)) * weighted_fr.mean():.4f}")
        