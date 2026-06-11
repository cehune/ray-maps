from segments.cluster import Cluster
from segments.primitives import SurfacePoint, Segment
from segments.primitives import SegmentTechnique
from segments.sampler_types import Sampler
import drjit as dr
import mitsuba as mi
import math
from segments.mmis import MMIS

class Propagation:
    def __init__(self, num_prop_iterations = 12, apply_kernel_correction: bool = False,
                 geom_clamp_factor: float | None = None, geom_clamp_mode: str = "hard",
                 geom_clamp_growth: float = 0.0):
        self.num_prop_iterations = num_prop_iterations
        self._iteration = 1                  # current render iteration (1-indexed)
        # DIAGNOSTIC toggle: multiply each pair's contribution by 1/K(cluster_of(i.y))
        self.apply_kernel_correction = apply_kernel_correction
        # Firefly suppression on geom_term. Two knobs:
        #   geom_clamp_factor: cap = factor × median(geom). None disables entirely.
        #   geom_clamp_mode: 'hard' → np.minimum(geom, cap)  (brick wall, max bias)
        #                    'soft' → geom * cap / (cap + geom)  (Reinhard-style)
        # Soft loses less energy from the legitimate tail at the same `factor`.
        self.geom_clamp_factor = geom_clamp_factor
        if geom_clamp_mode not in ("hard", "soft"):
            raise ValueError(f"geom_clamp_mode must be 'hard' or 'soft', got {geom_clamp_mode!r}")
        self.geom_clamp_mode = geom_clamp_mode
        # Progressive clamp relaxation. Effective cap at outer render iter n is
        #   cap_n = factor · n**geom_clamp_growth · median(geom).
        #   growth = 0 → constant cap (identical to the old fixed clamp; clean A/B).
        #   growth > 0 → cap grows each iteration, so the clamp eventually stops
        #     firing and its darkening bias vanishes in the running average. The
        #     fixed clamp instead leaves a permanent ~3% floor (mean_ratio ≈ 0.97).
        # _clamp_iter counts OUTER render iterations: +1 per iterate_propogation_vec
        # call (not per inner propagation pass), reset only when a new Propagation
        # is constructed (i.e. once per render config).
        self.geom_clamp_growth = geom_clamp_growth
        self._clamp_iter = 0

    def propagate_segment(self, segment: Segment, seg_idx: int, cluster: Cluster, samplers: dict[SegmentTechnique, 'Sampler']):
        out = mi.Color3f(0.0)

        y_cluster_idx = cluster.endpoint_to_cluster[seg_idx * 2 + 1]
        start, end = cluster.cluster_ranges[y_cluster_idx]
        for flat_idx in cluster.sorted_indices[start:end]:
            next_idx, which_end = cluster.endpoint_metadata[flat_idx]
            if next_idx == seg_idx:
                continue
            if which_end == 0:  # y-endpoint of seg is near the x of the next
                next_seg: Segment = cluster.segments[next_idx]
                if next_seg.mmis_weight == 0.0:  
                    continue
                
                sampler = samplers.get(next_seg.technique)
                if sampler is None:
                    continue
    
                f = sampler.shift_invariant_bsdf(segment, next_seg)
                contrib = next_seg.mmis_weight * f * next_seg.geom_term * next_seg.radiance_in
                out += contrib
        return out
        
    def propagate_all_segments(self, cluster: Cluster, samplers):
        # should be called once per iteration
        for seg_idx, segment in enumerate(cluster.segments):                
            segment.radiance_out = self.propagate_segment(segment, seg_idx, cluster, samplers)
        
    def swap_segment_radiance(self, cluster: Cluster):
        for segment in cluster.segments:
            if segment.y.is_light or segment.x.is_light:
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
        self._clamp_iter += 1   # outer render-iteration counter (drives clamp growth)

        # Initial radiance_in = Le for all segments
        Le = np.array([[float(s.Le.x), float(s.Le.y), float(s.Le.z)] for s in segs],
                      dtype=np.float64)  # (S, 3)
        is_light = np.array([s.y.is_light or s.x.is_light for s in segs], dtype=bool)

        rad_in = Le.copy()

        has_pairs = len(pair_cache.prop_i) > 0

        if has_pairs:
            j = pair_cache.prop_j  # (P,) source segment indices

            # Extract per-segment scalars once — static across all propagation iterations
            mmis_w = np.array([float(s.mmis_weight) for s in segs], dtype=np.float64)
            geom   = np.array([float(s.geom_term)   for s in segs], dtype=np.float64)

            # Firefly suppression on geom_term (1/len² explodes for short segments).
            # Median-relative so the cap scales with scene units. Bias/variance tradeoff
            # is controlled by self.geom_clamp_factor and self.geom_clamp_mode.
            if self.geom_clamp_factor is not None:
                geom_nz = geom[geom > 0]
                if len(geom_nz) > 0:
                    factor = float(self.geom_clamp_factor)
                    if self.geom_clamp_growth:
                        # cap grows as n**growth → clamp relaxes, bias → 0 in the limit
                        factor *= self._clamp_iter ** float(self.geom_clamp_growth)
                    geom_cap = factor * np.median(geom_nz)
                    if self.geom_clamp_mode == "soft":
                        # Reinhard: leaves values << cap untouched, asymptotes at cap.
                        denom = geom_cap + geom
                        geom = np.where(denom > 0, geom * geom_cap / denom, geom)
                    else:
                        geom = np.minimum(geom, geom_cap)

            # Pre-multiply BSDF with static mmis_weight and geom_term for each pair:
            # weighted_fr[k] = prop_fr[k] * mmis_w[j[k]] * geom[j[k]]   shape (P, 3)
            weighted_fr = pair_cache.prop_fr * (mmis_w[j] * geom[j])[:, np.newaxis]

            # OPTIONAL: 1/K kernel-area correction. K = tilt / voxel_size² (a density).
            # 1/K = voxel_size² / tilt has units of area. We multiply the per-pair
            # weight by 1/K for the cluster that owns the receiver's y endpoint
            # (which is where the local integration happens).
            if self.apply_kernel_correction and hasattr(cluster, "cluster_kernel_values"):
                if not hasattr(cluster, "endpoint_to_cluster"):
                    cluster.build_reverse_map()
                if len(cluster.cluster_kernel_values) == 0:
                    cluster.compute_cluster_kernel_values()
                K = cluster.cluster_kernel_values  # (C,)
                # cluster index of receiver i's y endpoint
                recv_cluster = cluster.endpoint_to_cluster[pair_cache.prop_i * 2 + 1]
                inv_K = np.where(K[recv_cluster] > 0, 1.0 / K[recv_cluster], 0.0)
                weighted_fr = weighted_fr * inv_K[:, np.newaxis]
                print(f"  [kernel-correction ON] median 1/K = {np.median(inv_K):.3e}, "
                      f"max 1/K = {inv_K.max():.3e}")

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
        # how much radiance is contributed per hop
        # print(f"expected gain per hop (pairs * weighted_fr): {(len(pair_cache.prop_i) / len(segs)) * weighted_fr.mean():.4f}")
        