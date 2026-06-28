from segments.cluster import Cluster
from segments.primitives import SurfacePoint, Segment
from segments.primitives import SegmentTechnique
from segments.sampler_types import Sampler
import drjit as dr
import mitsuba as mi
import math
from segments.mmis import MMIS

class Propagation:
    def __init__(self, num_prop_iterations = 12,
                 geom_clamp_factor: float | None = None, geom_clamp_mode: str = "hard",
                 row_gain_cap: float | None = 1.0,
                 merge_min_len_factor: float | None = 1.0):
        self.num_prop_iterations = num_prop_iterations
        self._iteration = 1                  # current render iteration (1-indexed)
        # Energy-conservation projection. The per-receiver row sum of the
        # propagation operator, Σ_j lum(w_j · f_j · G_j), estimates that
        # cluster's directional-hemispherical reflectance — physically ≤ albedo.
        # Rows exceeding the cap are scaled down to it. Unlike the median-relative
        # geom clamp this is asymptotically inactive (rows sit below albedo as the
        # estimator converges) and it bounds the operator's spectral radius below 1,
        # which makes propagation divergence (corner fireflies) impossible.
        # Set to max scene albedo (or 1.0); None disables.
        self.row_gain_cap = row_gain_cap
        # Sub-kernel exclusion for the NON-VEC path (the vec path gets this
        # from build_pair_cache): segments with len < factor * voxel_size are
        # skipped as merge sources. See build_pair_cache docstring.
        self.merge_min_len_factor = merge_min_len_factor
        # Firefly suppression on geom_term. Two knobs:
        #   geom_clamp_factor: cap = factor × median(geom). None disables entirely.
        #   geom_clamp_mode: 'hard' → np.minimum(geom, cap)  (brick wall, max bias)
        #                    'soft' → geom * cap / (cap + geom)  (Reinhard-style)
        # Soft loses less energy from the legitimate tail at the same `factor`.
        self.geom_clamp_factor = geom_clamp_factor
        if geom_clamp_mode not in ("hard", "soft"):
            raise ValueError(f"geom_clamp_mode must be 'hard' or 'soft', got {geom_clamp_mode!r}")
        self.geom_clamp_mode = geom_clamp_mode

    @staticmethod
    def _lum(c) -> float:
        return 0.2126 * float(c.x) + 0.7152 * float(c.y) + 0.0722 * float(c.z)

    def _classical_links(self, cluster: Cluster) -> dict:
        """parent_idx -> list[(child_idx, throughput)] for sub-kernel CAMERA
        segments (mirror of the cls_* arrays in build_pair_cache)."""
        links: dict[int, list] = {}
        if self.merge_min_len_factor is None or float(cluster.voxel_size) <= 0:
            return links
        min_len = float(self.merge_min_len_factor) * float(cluster.voxel_size)
        y_owner = {id(s.y): i for i, s in enumerate(cluster.segments)}
        for j, sj in enumerate(cluster.segments):
            if float(sj.len) >= min_len:
                continue
            # bridge segments are technique=BRIDGE — excluded here.
            if sj.technique != SegmentTechnique.CAMERA or sj.x.is_camera:
                continue
            p = y_owner.get(id(sj.x))
            if p is None or p == j:
                continue
            links.setdefault(p, []).append((j, sj.throughput))
        return links

    def propagate_segment(self, segment: Segment, seg_idx: int, cluster: Cluster,
                          samplers: dict[SegmentTechnique, 'Sampler'],
                          cap_budget: float | None = None):
        # light-terminal receivers are pinned to Le by the swap — gathering
        # into them is wasted (and bridge segments' synthetic y has no bsdf)
        if segment.y.is_light:
            return mi.Color3f(0.0)
        out = mi.Color3f(0.0)
        row_gain = 0.0  # luminance of Σ_j w_j·f_j·G_j — the operator row sum
        cap = self.row_gain_cap if cap_budget is None else cap_budget

        min_len = 0.0
        if self.merge_min_len_factor is not None and float(cluster.voxel_size) > 0:
            min_len = float(self.merge_min_len_factor) * float(cluster.voxel_size)

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
                if float(next_seg.len) < min_len:  # sub-kernel: not a merge source
                    continue

                sampler = samplers.get(next_seg.technique)
                if sampler is None:
                    continue

                f = sampler.shift_invariant_bsdf(segment, next_seg)
                wfg = next_seg.mmis_weight * f * next_seg.geom_term
                row_gain += (0.2126 * float(wfg.x) + 0.7152 * float(wfg.y)
                             + 0.0722 * float(wfg.z))
                out += wfg * next_seg.radiance_in

        # Energy-conservation projection — mirror of the vectorized version.
        # The row sum estimates this cluster's reflectance and cannot
        # physically exceed albedo; scale over-unity rows down uniformly.
        # (cap may be a reduced budget when classical links consume part of it.)
        if cap is not None and row_gain > cap:
            out = out * (float(cap) / row_gain) if cap > 0 else mi.Color3f(0.0)
        return out

    def propagate_all_segments(self, cluster: Cluster, samplers):
        # should be called once per iteration
        links = self._classical_links(cluster)
        for seg_idx, segment in enumerate(cluster.segments):
            # cap applies to the merged gather only; classical links are
            # unbiased one-sample MC and never scaled (see vec version).
            out = self.propagate_segment(segment, seg_idx, cluster, samplers)
            for cj, tw in links.get(seg_idx, ()):
                out = out + tw * cluster.segments[cj].radiance_in
            segment.radiance_out = out
        
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
                    geom_cap = float(self.geom_clamp_factor) * np.median(geom_nz)
                    if self.geom_clamp_mode == "soft":
                        # Reinhard: leaves values << cap untouched, asymptotes at cap.
                        denom = geom_cap + geom
                        geom = np.where(denom > 0, geom * geom_cap / denom, geom)
                    else:
                        geom = np.minimum(geom, geom_cap)

            # Pre-multiply BSDF with static mmis_weight and geom_term for each pair:
            # weighted_fr[k] = prop_fr[k] * mmis_w[j[k]] * geom[j[k]]   shape (P, 3)
            weighted_fr = pair_cache.prop_fr * (mmis_w[j] * geom[j])[:, np.newaxis]

            # Energy-conservation projection (see __init__). Row i of the
            # propagation operator has luminance gain Σ_{pairs k: prop_i[k]=i}
            # lum(weighted_fr[k]); a physical reflectance cannot exceed albedo,
            # so rows above row_gain_cap are scaled down uniformly. This bounds
            # the spectral radius of the operator at the cap, so the fixed-point
            # iteration below cannot diverge (the corner-firefly mechanism).
            # Classical links are NOT part of this projection. Their realized
            # gain (≈ albedo when the child is short) is a legitimate
            # one-sample MC fluctuation — it averages to the small near-field
            # share across realizations. Subtracting it from the row budget
            # (an earlier version did) systematically shaved ~1%/iteration of
            # legitimate long-range energy from every parent of a short child.
            # The cap applies to the MERGED operator only, which is the part
            # whose row sum estimates a reflectance and can nonphysically
            # exceed it. Classical edges are acyclic (path-parent links), so
            # they cannot form feedback loops regardless.
            if self.row_gain_cap is not None:
                cap = float(self.row_gain_cap)
                LUM = np.array([0.2126, 0.7152, 0.0722])
                lum = weighted_fr @ LUM
                row_gain = np.zeros(S, dtype=np.float64)
                np.add.at(row_gain, pair_cache.prop_i, lum)
                over = row_gain > cap
                n_capped = int(over.sum())
                if n_capped > 0:
                    scale = np.where(over, cap / np.maximum(row_gain, 1e-300), 1.0)
                    weighted_fr = weighted_fr * scale[pair_cache.prop_i][:, np.newaxis]

        has_cls = len(pair_cache.cls_i) > 0

        for _ in range(self.num_prop_iterations):
            rad_out = np.zeros((S, 3), dtype=np.float64)

            if has_pairs:
                # contrib[k] = weighted_fr[k] * rad_in[j[k]]   shape (P, 3)
                contrib = weighted_fr * rad_in[pair_cache.prop_j]
                np.add.at(rad_out, pair_cache.prop_i, contrib)

            if has_cls:
                # classical near-field links: parent ← throughput(child) · L(child)
                np.add.at(rad_out, pair_cache.cls_i,
                          pair_cache.cls_w * rad_in[pair_cache.cls_j])

            # Swap: light endpoints pin to Le; others advance to rad_out
            rad_in = np.where(is_light[:, np.newaxis], Le, rad_out)

        # ── direct-only gather for the split final readout ────────────────────
        # D(i) = Σ over pairs whose SOURCE is light-terminal of w·f·G·Le
        # (plus light-terminal classical links). The light-terminal segments
        # act as shared bridge samples: many light hits, merged with MMIS
        # weights — a smooth direct-light estimate. The depth-2 final gather
        # then carries ONLY indirect through the path's own bounce, so a
        # single lucky emitter hit never multiplies a raw 1/p path weight.
        direct = np.zeros((S, 3), dtype=np.float64)
        if has_pairs:
            lt = is_light[pair_cache.prop_j]
            if lt.any():
                np.add.at(direct, pair_cache.prop_i[lt],
                          weighted_fr[lt] * Le[pair_cache.prop_j[lt]])
        if has_cls:
            lt = is_light[pair_cache.cls_j]
            if lt.any():
                np.add.at(direct, pair_cache.cls_i[lt],
                          pair_cache.cls_w[lt] * Le[pair_cache.cls_j[lt]])

        # Write final radiance_in (and the direct split) back to segments
        for seg_idx, seg in enumerate(segs):
            seg.radiance_in = mi.Color3f(
                float(rad_in[seg_idx, 0]),
                float(rad_in[seg_idx, 1]),
                float(rad_in[seg_idx, 2]),
            )
            seg.direct_in = mi.Color3f(
                float(direct[seg_idx, 0]),
                float(direct[seg_idx, 1]),
                float(direct[seg_idx, 2]),
            )
        # how much radiance is contributed per hop
        # print(f"expected gain per hop (pairs * weighted_fr): {(len(pair_cache.prop_i) / len(segs)) * weighted_fr.mean():.4f}")
        