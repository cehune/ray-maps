from tests.utils import *
from segments.primitives import SegmentTechnique
from segments.mmis import MMIS
from segments.cluster import Cluster
import math
import numpy as np

class TestMMIS:
    def test_mmis_weight_hardcoded_cluster(self):
        """
        Hardcode cluster state directly — bypass actual clustering.
        Two auxiliaries in the cluster, each with a known conditional_pdf return value.
        Verify weight = 1 / (pdf_0 + pdf_1).
        """
        mmis = MMIS()

        aux_0 = MagicMock()
        aux_0.technique = SegmentTechnique.CAMERA
        aux_1 = MagicMock()
        aux_1.technique = SegmentTechnique.CAMERA

        mock_cluster = MagicMock()
        mock_cluster.cluster_ranges = [(0, 4)]
        mock_cluster.sorted_indices = np.array([0, 1, 2, 3], dtype=np.int32)
        mock_cluster.endpoint_to_cluster = np.array([0, 0], dtype=np.int32)

        # flat 0: seg 0, x-endpoint (skipped)
        # flat 1: seg 0, y-endpoint (auxiliary candidate)
        # flat 2: seg 1, x-endpoint (skipped)
        # flat 3: seg 1, y-endpoint (auxiliary candidate)
        mock_cluster.endpoint_metadata = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ], dtype=np.int32)
        # make list indexing work on the mock
        mock_cluster.segments = [aux_0, aux_1]

        pdf_0, pdf_1 = 0.3, 0.5
        mock_sampler = MagicMock()
        mock_sampler.technique_type = SegmentTechnique.CAMERA
        mock_sampler.conditional_pdf.side_effect = [pdf_0, pdf_1]
        samplers = {
            SegmentTechnique.CAMERA: mock_sampler,
        }

        segment = MagicMock()
        weight = mmis.compute_mmis_weight(segment, 0, mock_cluster, samplers)
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