import pytest
import numpy as np
import mitsuba as mi
import math

mi.set_variant('scalar_rgb')

from segments.cluster import Cluster
from segments.primitives import *
from tests.utils import *

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(seed=42)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCluster:
    def test_voxel_size_formula(self, simple_segments):
        """r = cbrt(c * Vol / N) where N = 2 * num_segments."""
        unit_aabb = make_aabb([0, 0, 0], [1, 1, 1])
        
        c_obj = Cluster(c=5)
        c_obj.set_scene_aabb(unit_aabb)
        c_obj.set_segments(simple_segments)

        r = c_obj._compute_voxel_size()

        vol = unit_aabb.volume()
        N = 2 * len(simple_segments)
        expected = (5 * vol / N) ** (1.0 / 3.0)
        assert math.isclose(r, expected, rel_tol=1e-9)

    def test_voxel_size_scales_with_c(self, simple_segments):
        """Larger c => larger voxels."""
        unit_aabb = make_aabb([0, 0, 0], [1, 1, 1])
        c_small = Cluster(c=2)
        c_large = Cluster(c=10)
        for c_obj in [c_small, c_large]:
            c_obj.set_scene_aabb(unit_aabb)
            c_obj.set_segments(simple_segments)

        assert c_small._compute_voxel_size() < c_large._compute_voxel_size()

    def test_voxel_size_scales_with_scene(self, simple_segments):
        """Larger scene => larger voxels for same c and N."""
        small_aabb = make_aabb([0, 0, 0], [1,  1,  1 ])
        large_aabb = make_aabb([0, 0, 0], [10, 10, 10])

        c_small = Cluster(c=5)
        c_large = Cluster(c=5)
        c_small.set_scene_aabb(small_aabb)
        c_large.set_scene_aabb(large_aabb)
        c_small.set_segments(simple_segments)
        c_large.set_segments(simple_segments)

        assert c_small._compute_voxel_size() < c_large._compute_voxel_size()

    def test_correct_number_of_keys_generated(self, cluster, rng):
        """One key per endpoint = 2 * num_segments."""
        cluster._compute_voxel_size()
        cluster.voxel_size = cluster._compute_voxel_size()
        positions, normals = cluster._flatten_endpoints()
        keys = cluster._generate_cluster_keys(positions, normals, rng)

        assert len(keys) == 2 * len(cluster.segments)

    def test_flatten_endpoint_count(self, cluster):
        """_flatten_endpoints produces exactly 2S rows."""
        positions, normals = cluster._flatten_endpoints()
        S = len(cluster.segments)

        assert positions.shape == (2 * S, 3)
        assert normals.shape   == (2 * S, 3)

    def test_flatten_endpoint_meta_shape(self, cluster):
        """endpoint_meta has shape (2S, 2) after flattening."""
        cluster._flatten_endpoints()
        S = len(cluster.segments)

        assert cluster.endpoint_metadata.shape == (2 * S, 2)

    def test_flatten_positions_match_segments(self, cluster, simple_segments):
        """
        Even rows = x endpoints, odd rows = y endpoints.
        Check that positions match the original segment data.
        """
        positions, _ = cluster._flatten_endpoints()

        for seg_idx, seg in enumerate(simple_segments):
            x_row = positions[seg_idx * 2]
            y_row = positions[seg_idx * 2 + 1]

            assert np.allclose(x_row, [seg.x.p.x, seg.x.p.y, seg.x.p.z])
            assert np.allclose(y_row, [seg.y.p.x, seg.y.p.y, seg.y.p.z])

    def test_flatten_normals_match_segments(self, cluster, simple_segments):
        """Normals stay correctly associated with their endpoints after flattening."""
        _, normals = cluster._flatten_endpoints()

        for seg_idx, seg in enumerate(simple_segments):
            x_row = normals[seg_idx * 2]
            y_row = normals[seg_idx * 2 + 1]

            assert np.allclose(x_row, [seg.x.n.x, seg.x.n.y, seg.x.n.z])
            assert np.allclose(y_row, [seg.y.n.x, seg.y.n.y, seg.y.n.z])

    def test_endpoint_meta_correct(self, cluster, simple_segments):
        """
        endpoint_meta[flat_idx] = (seg_idx, which_end).
        Check every entry is consistent with flattening order.
        """
        cluster._flatten_endpoints()

        for seg_idx in range(len(simple_segments)):
            x_meta = cluster.endpoint_metadata[seg_idx * 2]
            y_meta = cluster.endpoint_metadata[seg_idx * 2 + 1]

            assert x_meta[0] == seg_idx and x_meta[1] == 0  # x endpoint
            assert y_meta[0] == seg_idx and y_meta[1] == 1  # y endpoint

    def test_cluster_ranges_cover_all_endpoints(self, cluster, rng):
        """All 2S endpoints must appear in exactly one cluster."""
        cluster.cluster(rng)

        covered = sum(end - start for start, end in cluster.cluster_ranges)
        assert covered == 2 * len(cluster.segments)

    def test_cluster_ranges_non_overlapping(self, cluster, rng):
        """Cluster ranges must be non-overlapping and contiguous."""
        cluster.cluster(rng)

        prev_end = 0
        for start, end in cluster.cluster_ranges:
            assert start == prev_end, "Gap or overlap between cluster ranges"
            assert end > start,       "Empty cluster range"
            prev_end = end

    def test_sorted_indices_are_permutation(self, cluster, rng):
        """sorted_indices must be a permutation of [0, 2S)."""
        cluster.cluster(rng)
        N = 2 * len(cluster.segments)

        assert sorted(cluster.sorted_indices.tolist()) == list(range(N))

    def test_get_cluster_segments_retrieves_correct_segment(self, cluster, rng):
        """
        get_cluster_segments must return the actual Segment objects,
        not copies
        """
        cluster.cluster(rng)

        for c_idx in range(len(cluster.cluster_ranges)):
            for seg, _ in cluster.get_cluster_segments(c_idx):
                assert any(seg is s for s in cluster.segments), \
                    f"Segment returned by cluster {c_idx} is not in segments list"


    def test_nearby_points_same_normal_cluster_together(self):
        class ZeroRng:
            def uniform(self, low, high, size):
                return np.zeros(size)

        s0 = make_segment(
            x=make_surface_point(5.0, 5.0, 5.0,  0, 1, 0),
            y=make_surface_point(1.0, 1.0, 1.0,  0, 1, 0),
        )
        s1 = make_segment(
            x=make_surface_point(5.1, 5.0, 5.0,  0, 1, 0),
            y=make_surface_point(9.0, 9.0, 9.0,  0, 1, 0),
        )

        aabb = make_aabb([0, 0, 0], [10, 10, 10])
        c_obj = Cluster(c=50)
        c_obj.set_scene_aabb(aabb)
        c_obj.set_segments([s0, s1])

        r = c_obj._compute_voxel_size()
        c_obj.voxel_size = r
        c_obj.cluster(ZeroRng())
        target_cluster = None
        for c_idx, (start, end) in enumerate(c_obj.cluster_ranges):
            if 0 in c_obj.sorted_indices[start:end].tolist():
                target_cluster = c_idx
                break

        assert target_cluster is not None
        members_flat = c_obj.sorted_indices[
            c_obj.cluster_ranges[target_cluster][0]:
            c_obj.cluster_ranges[target_cluster][1]
        ].tolist()

        assert 2 in members_flat, \
            f"s1.x (flat_idx=2) should share cluster with s0.x. Got: {members_flat}"
        print(members_flat)
        
    def test_distant_points_do_not_cluster_together(self, rng):
        """
        Two endpoints far apart must not be in the same cluster.
        Use a small voxel size (large N relative to scene) to guarantee separation.
        """
        s0 = make_segment(
            x=make_surface_point(0.01, 0.01, 0.01,  0, 1, 0),
            y=make_surface_point(0.02, 0.02, 0.02,  0, 1, 0),
        )
        s1 = make_segment(
            x=make_surface_point(0.99, 0.99, 0.99,  0, 1, 0),
            y=make_surface_point(0.98, 0.98, 0.98,  0, 1, 0),
        )

        aabb = make_aabb([0, 0, 0], [1, 1, 1])
        c_obj = Cluster(c=1)  # small c => small voxels => separation guaranteed
        c_obj.set_scene_aabb(aabb)
        c_obj.set_segments([s0, s1])
        c_obj.cluster(rng)

        def find_cluster_of(flat_idx):
            for c_idx, (start, end) in enumerate(c_obj.cluster_ranges):
                if flat_idx in c_obj.sorted_indices[start:end]:
                    return c_idx

        assert find_cluster_of(0) != find_cluster_of(2), \
            "Distant endpoints should not share a cluster"

    def test_different_normals_do_not_cluster(self, rng):
        """
        Two endpoints at the same position but with different dominant normal
        directions must not cluster together.
        """
        s0 = make_segment(
            x=make_surface_point(0.5, 0.5, 0.5,  0, 1, 0),  # normal up   → octant 3
            y=make_surface_point(0.1, 0.1, 0.1,  0, 1, 0),
        )
        s1 = make_segment(
            x=make_surface_point(0.5, 0.5, 0.5,  1, 0, 0),  # normal right → octant 1
            y=make_surface_point(0.9, 0.9, 0.9,  1, 0, 0),
        )

        aabb = make_aabb([0, 0, 0], [1, 1, 1])
        c_obj = Cluster(c=50)  # large voxels so position alone wouldn't separate them
        c_obj.set_scene_aabb(aabb)
        c_obj.set_segments([s0, s1])
        c_obj.cluster(rng)

        def find_cluster_of(flat_idx):
            for c_idx, (start, end) in enumerate(c_obj.cluster_ranges):
                if flat_idx in c_obj.sorted_indices[start:end]:
                    return c_idx

        assert find_cluster_of(0) != find_cluster_of(2), \
            "Same position but different normal octant should not cluster"