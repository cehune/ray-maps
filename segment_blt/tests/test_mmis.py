from tests.utils import *
from segments.primitives import SegmentTechnique
from segments.mmis import MMIS
from segments.cluster import Cluster
import math
import numpy as np

class TestMMIS:
    def test_mmis_weight_hardcoded_cluster(self):
        """
        Hardcode cluster state: two auxiliary segments (whose y-endpoints land
        in the cluster) and one main segment (whose x-endpoint lands in the cluster).
        Verify weight = 1 / (pdf_0 + pdf_1).
        """
        mmis = MMIS()

        aux_0 = MagicMock()
        aux_0.technique = SegmentTechnique.CAMERA
        aux_1 = MagicMock()
        aux_1.technique = SegmentTechnique.CAMERA
        main_seg = MagicMock()
        main_seg.technique = SegmentTechnique.CAMERA

        mock_cluster = MagicMock()
        # cluster contains 3 endpoints: aux_0.y, aux_1.y, main_seg.x
        # their flat indices: aux_0.y=1, aux_1.y=3, main_seg.x=4
        mock_cluster.cluster_ranges = [(0, 3)]
        mock_cluster.sorted_indices = np.array([1, 3, 4], dtype=np.int32)
        # main_seg is index 2 → x-endpoint flat = 2*2+0 = 4 → cluster 0
        mock_cluster.endpoint_to_cluster = np.zeros(6, dtype=np.int32)

        # endpoint_metadata[flat_idx] = (seg_idx, which_end)
        mock_cluster.endpoint_metadata = np.array([
            [0, 0],   # flat 0: aux_0 x
            [0, 1],   # flat 1: aux_0 y  ← auxiliary candidate
            [1, 0],   # flat 2: aux_1 x
            [1, 1],   # flat 3: aux_1 y  ← auxiliary candidate
            [2, 0],   # flat 4: main_seg x  ← main's x lives here
            [2, 1],   # flat 5: main_seg y
        ], dtype=np.int32)
        mock_cluster.segments = [aux_0, aux_1, main_seg]

        pdf_0, pdf_1 = 0.3, 0.5
        mock_sampler = MagicMock()
        mock_sampler.technique_type = SegmentTechnique.CAMERA
        mock_sampler.conditional_pdf.side_effect = [pdf_0, pdf_1]
        samplers = {SegmentTechnique.CAMERA: mock_sampler}

        weight = mmis.compute_mmis_weight(main_seg, 2, mock_cluster, samplers)
        expected = 1.0 / (pdf_0 + pdf_1)
        assert abs(weight - expected) < 1e-6, \
            f"Expected {expected:.6f}, got {weight:.6f}"

    def test_compute_all_mmis_weights(self):
        """
        Run compute_all_mmis_weights over a real cluster with 3 segments.
        Camera-origin segments get weight=1.0.
        Interior segments get a positive finite weight.
        """
        mmis = MMIS()

        s0 = make_segment(
            make_surface_point(0.0, 0.0, 0.0, 0, 1, 0, is_camera=True),
            make_surface_point(0.3, 0.3, 0.3, 0, 1, 0),
        )
        s1 = make_segment(
            make_surface_point(0.3, 0.3, 0.3, 0, 1, 0),
            make_surface_point(0.6, 0.6, 0.6, 0, 1, 0),
        )
        s2 = make_segment(
            make_surface_point(0.31, 0.3, 0.3, 0, 1, 0),
            make_surface_point(0.7,  0.5, 0.5, 0, 1, 0),
        )

        aabb = make_aabb([0, 0, 0], [1, 1, 1])
        c = Cluster(c=5)
        c.set_scene_aabb(aabb)
        c.set_segments([s0, s1, s2])
        rng = np.random.default_rng(0)
        c.cluster(rng, voxel_size=0.2)

        mock_sampler = MagicMock()
        mock_sampler.technique_type = SegmentTechnique.CAMERA
        mock_sampler.conditional_pdf.return_value = 0.4        
        samplers = {
            SegmentTechnique.CAMERA: mock_sampler,
        }

        mmis.compute_all_mmis_weights(c, samplers)

        assert s0.mmis_weight == 1.0, \
            "Camera-origin segment should have weight=1.0"

        for seg in [s1, s2]:
            assert seg.mmis_weight > 0.0, \
                f"Interior segment must have positive weight, got {seg.mmis_weight}"
            assert math.isfinite(seg.mmis_weight), \
                f"Interior segment weight must be finite, got {seg.mmis_weight}"
            
    def test_nonvec_vec_parity(self):
        """
        Non-vec (compute_all_mmis_weights) and vec (compute_all_mmis_weights_vec
        via PairCache) must produce identical mmis_weight values for every segment.

        Uses a deterministic mock conditional_pdf that returns the same value for
        any (segment, auxiliary) pair, so both code paths sum the same number of
        contributions.
        """
        from segments.pair_cache import build_pair_cache

        # ---- build a small cluster of 4 segments ----
        # s0 is camera-origin (trivial → weight pinned to 1.0)
        # s1, s2, s3 are interior segments whose x-endpoints land near each other,
        # so they all share an x-cluster and act as mutual auxiliaries.
        s0 = make_segment(
            make_surface_point(0.0, 0.0, 0.0, 0, 1, 0, is_camera=True),
            make_surface_point(0.30, 0.30, 0.30, 0, 1, 0),
        )
        s1 = make_segment(
            make_surface_point(0.30, 0.30, 0.30, 0, 1, 0),
            make_surface_point(0.60, 0.60, 0.60, 0, 1, 0),
        )
        s2 = make_segment(
            make_surface_point(0.31, 0.30, 0.30, 0, 1, 0),
            make_surface_point(0.70, 0.50, 0.50, 0, 1, 0),
        )
        s3 = make_segment(
            make_surface_point(0.32, 0.31, 0.29, 0, 1, 0),
            make_surface_point(0.55, 0.55, 0.65, 0, 1, 0),
        )

        aabb = make_aabb([0, 0, 0], [1, 1, 1])
        c = Cluster(c=5)
        c.set_scene_aabb(aabb)
        c.set_segments([s0, s1, s2, s3])
        rng = np.random.default_rng(0)
        c.cluster(rng, voxel_size=0.2)

        # ---- deterministic sampler: conditional_pdf returns 0.4 for any pair ----
        # because both paths call conditional_pdf(j, t) with the same args, a
        # constant return is enough to test that they accumulate the same set
        # of (j, t) pairs.
        mock_sampler = MagicMock()
        mock_sampler.technique_type = SegmentTechnique.CAMERA
        mock_sampler.conditional_pdf.return_value = 0.4
        # shift_invariant_bsdf is called when building prop_fr; not relevant to
        # MMIS weight, but must not crash.
        mock_sampler.shift_invariant_bsdf.return_value = mi.Color3f(0.0)
        samplers = {SegmentTechnique.CAMERA: mock_sampler}

        mmis = MMIS()

        # ---- non-vec path ----
        mmis.compute_all_mmis_weights(c, samplers)
        nonvec = np.array([s.mmis_weight for s in c.segments])

        # reset so the vec path can't accidentally read stale values
        for s in c.segments:
            s.mmis_weight = 0.0

        # ---- vec path ----
        pair_cache = build_pair_cache(c, samplers)
        mmis.compute_all_mmis_weights_vec(c, pair_cache)
        vec = np.array([s.mmis_weight for s in c.segments])

        # ---- parity ----
        assert np.allclose(nonvec, vec, atol=1e-9), (
            f"non-vec and vec MMIS weights disagree:\n"
            f"  non-vec: {nonvec}\n"
            f"  vec:     {vec}\n"
            f"  max abs diff: {np.abs(nonvec - vec).max()}"
        )

        # sanity: trivial segment is pinned
        assert nonvec[0] == 1.0, "camera-origin segment should be trivial (weight=1.0)"

        # sanity: at least one interior segment got contributions
        assert np.any(nonvec[1:] > 0.0), (
            "expected at least one interior segment to have a positive weight; "
            "if all are zero the cluster setup didn't actually share auxiliaries"
        )