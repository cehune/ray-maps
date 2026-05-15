import pytest
import math
import numpy as np
import mitsuba as mi
import drjit as dr
from unittest.mock import MagicMock
from segments.src.propagation import Propagation
from segments.src.cluster import Cluster
from segments.src.primitives import Segment, SegmentTechnique, SurfacePoint
from segments.src.sampler_types import Sampler
from tests.utils import (
    make_surface_point,
    make_surface_point_with_mock_bsdf,
    make_segment,
    make_aabb,
    _make_chain,
    ZeroRng
)

mi.set_variant("scalar_rgb")

# ── shared helpers ────────────────────────────────────────────────────────────

def make_mock_sampler(technique: SegmentTechnique, bsdf_color=(0.8, 0.8, 0.8)):
    s = MagicMock()  # no spec
    s.technique_type = technique
    s.shift_invariant_bsdf.return_value = mi.Color3f(*bsdf_color)
    return s

def make_clustered(segments, voxel_size=0.5):
    aabb = make_aabb([0, 0, 0], [1, 1, 1])
    c = Cluster(c=5)
    c.set_scene_aabb(aabb)
    c.set_segments(segments)
    c.cluster(ZeroRng(), voxel_size=voxel_size)
    return c


def make_prop(kernel_radius=0.5, kernel_weight=0.7):
    return Propagation(kernel_radius=kernel_radius, kernel_weight=kernel_weight)


def camera_samplers(bsdf_color=(0.8, 0.8, 0.8)):
    return {SegmentTechnique.CAMERA: make_mock_sampler(SegmentTechnique.CAMERA, bsdf_color)}


# # ── kernel_check_disc ─────────────────────────────────────────────────────────

class TestKernelCheckDisc:

    def test_same_point_passes(self):
        prop = make_prop(kernel_radius=0.1)
        sp = make_surface_point(0, 0, 0, 0, 0, 1)
        assert prop.kernel_check_disc(sp, sp)

    def test_point_within_radius_passes(self):
        prop = make_prop(kernel_radius=0.5)
        center = make_surface_point(0, 0, 0, 0, 0, 1)
        nearby = make_surface_point(0.3, 0.0, 0, 0, 0, 1)  # tangential dist 0.3
        assert prop.kernel_check_disc(center, nearby)

    def test_point_outside_radius_fails(self):
        prop = make_prop(kernel_radius=0.2)
        center = make_surface_point(0, 0, 0, 0, 0, 1)
        far    = make_surface_point(0.5, 0.0, 0, 0, 0, 1)  # tangential dist 0.5
        assert not prop.kernel_check_disc(center, far)

    def test_tangential_only_normal_offset_passes(self):
        """
        A point directly above the center (along the normal) has zero
        tangential distance and must always pass the disc check.
        """
        prop = make_prop(kernel_radius=0.1)
        center = make_surface_point(0, 0, 0, 0, 0, 1)
        above  = make_surface_point(0, 0, 5.0, 0, 0, 1)  # along normal, far up
        assert prop.kernel_check_disc(center, above)

    def test_boundary_point_passes(self):
        """Point exactly at radius must pass (<=)."""
        prop = make_prop(kernel_radius=0.5)
        center = make_surface_point(0, 0, 0, 0, 0, 1)
        edge   = make_surface_point(0.5, 0, 0, 0, 0, 1)
        assert prop.kernel_check_disc(center, edge)

    def test_tilted_normal_measures_tangentially(self):
        prop = make_prop(kernel_radius=0.1)
        # normal along y — point far along normal has zero tangential distance
        center  = make_surface_point(0, 0, 0,  0, 1, 0)
        along_n = make_surface_point(0, 5, 0,  0, 1, 0)
        assert prop.kernel_check_disc(center, along_n)
        # point offset tangentially by 0.5 > radius 0.1 must fail
        tangential = make_surface_point(0.5, 0, 0,  0, 1, 0)
        assert not prop.kernel_check_disc(center, tangential)


# ── kernel_val / update ───────────────────────────────────────────────────────

class TestKernelVal:

    def test_kernel_val_formula(self):
        """K = 1 / (pi * r^2)"""
        r = 0.3
        prop = make_prop(kernel_radius=r)
        expected = 1.0 / (math.pi * r * r)
        assert prop.kernel_val == pytest.approx(expected, rel=1e-6)

    def test_update_shrinks_radius(self):
        prop = make_prop(kernel_radius=1.0, kernel_weight=0.5)
        prop._update_kernel()
        assert prop.kernel_radius == pytest.approx(0.5, rel=1e-6)

    def test_update_refreshes_kernel_val(self):
        """After update, kernel_val must reflect the new radius."""
        prop = make_prop(kernel_radius=1.0, kernel_weight=0.5)
        prop._update_kernel()
        expected = 1.0 / (math.pi * 0.5 * 0.5)
        assert prop.kernel_val == pytest.approx(expected, rel=1e-6)

    def test_repeated_updates_multiply(self):
        prop = make_prop(kernel_radius=1.0, kernel_weight=0.5)
        prop._update_kernel()
        prop._update_kernel()
        assert prop.kernel_radius == pytest.approx(0.25, rel=1e-6)


# # ── propagate_segment ─────────────────────────────────────────────────────────

class TestPropagateSegment:

    def _chain(self):
        """
        s0 (camera) -> s1 (interior) -> s2 (interior)
        s1.y cluster overlaps s2.x so they should interact.
        """
        s0 = make_segment(
            x=make_surface_point(0, 0, 0, 0, 0, 1, is_camera=True),
            y=make_surface_point_with_mock_bsdf(0.1, 0, 0, 0, 0, 1),
            technique=SegmentTechnique.CAMERA,
        )
        s1 = make_segment(
            x=make_surface_point_with_mock_bsdf(0.1, 0, 0, 0, 0, 1),
            y=make_surface_point_with_mock_bsdf(0.2, 0, 0, 0, 0, 1),
            technique=SegmentTechnique.CAMERA,
        )
        s2 = make_segment(
            x=make_surface_point_with_mock_bsdf(0.21, 0, 0, 0, 0, 1),
            y=make_surface_point_with_mock_bsdf(0.4, 0, 0, 0, 0, 1),
            technique=SegmentTechnique.CAMERA,
        )
        s1.mmis_weight = 1.0
        s2.mmis_weight = 1.0
        s1.radiance_in = mi.Color3f(1.0)
        s2.radiance_in = mi.Color3f(1.0)
        return s0, s1, s2

    def test_output_non_negative_finite(self):
        s0, s1, s2 = self._chain()
        cluster = make_clustered([s0, s1, s2], voxel_size=0.3)
        prop = make_prop(kernel_radius=0.5)
        result = prop.propagate_segment(s1, 1, cluster, camera_samplers())
        for ch in [result.x, result.y, result.z]:
            assert float(ch) >= 0.0
            assert math.isfinite(float(ch))

    def test_does_not_self_scatter(self):
        """
        A segment must not contribute to its own propagation output.
        Even if it's in its own cluster, the self-intersection guard fires.
        """
        s0, s1, s2 = self._chain()
        cluster = make_clustered([s0, s1, s2], voxel_size=0.3)
        prop = make_prop(kernel_radius=0.5)
        samplers = camera_samplers()

        # inject a canary: if self is picked up, bsdf call count goes up per self
        call_count_before = samplers[SegmentTechnique.CAMERA].shift_invariant_bsdf.call_count
        prop.propagate_segment(s1, 1, cluster, samplers)
        calls = samplers[SegmentTechnique.CAMERA].shift_invariant_bsdf.call_args_list

        for call in calls:
            seg_arg, next_arg = call[0][0], call[0][1]
            assert seg_arg is not next_arg, "Segment must not scatter with itself"

    def test_zero_mmis_weight_segments_excluded(self):
        """
        Segments with mmis_weight=0 must not contribute — they're
        either camera-origin or uninitialized.
        """
        s0, s1, s2 = self._chain()
        s2.mmis_weight = 0.0
        cluster = make_clustered([s0, s1, s2], voxel_size=0.3)
        prop = make_prop(kernel_radius=0.5)
        samplers = camera_samplers()

        prop.propagate_segment(s1, 1, cluster, samplers)

        for call in samplers[SegmentTechnique.CAMERA].shift_invariant_bsdf.call_args_list:
            assert call[0][1] is not s2, "Zero-weight segment must not be evaluated"

    def test_outside_kernel_excluded(self):
        """
        A segment spatially outside the kernel radius must not contribute
        even if it's in the same cluster.
        """
        s0, s1, s2 = self._chain()
        cluster = make_clustered([s0, s1, s2], voxel_size=0.3)
        # tiny kernel: nothing nearby should pass
        prop = make_prop(kernel_radius=1e-6)
        samplers = camera_samplers()

        result = prop.propagate_segment(s1, 1, cluster, samplers)
        samplers[SegmentTechnique.CAMERA].shift_invariant_bsdf.assert_not_called()
        assert float(result.x) == pytest.approx(0.0, abs=1e-8)

    def test_scales_with_radiance_in(self):
        """
        Doubling radiance_in on all neighbours must double the output,
        since the estimator is linear in s'.radiance_in.
        """
        s0, s1, s2 = self._chain()
        cluster = make_clustered([s0, s1, s2], voxel_size=0.3)
        prop = make_prop(kernel_radius=0.5)
        samplers = camera_samplers()

        s2.radiance_in = mi.Color3f(1.0)
        r_single = prop.propagate_segment(s1, 1, cluster, samplers)

        s2.radiance_in = mi.Color3f(2.0)
        r_double = prop.propagate_segment(s1, 1, cluster, samplers)

        assert float(r_double.x) == pytest.approx(2.0 * float(r_single.x), rel=1e-4)

    def test_scales_with_mmis_weight(self):
        """
        Doubling s'.mmis_weight must double the output.
        """
        s0, s1, s2 = self._chain()
        cluster = make_clustered([s0, s1, s2], voxel_size=0.3)
        prop = make_prop(kernel_radius=0.5)

        s2.mmis_weight = 1.0
        r_single = prop.propagate_segment(s1, 1, cluster, camera_samplers())

        s2.mmis_weight = 2.0
        r_double = prop.propagate_segment(s1, 1, cluster, camera_samplers())

        assert float(r_double.x) == pytest.approx(2.0 * float(r_single.x), rel=1e-4)

    def test_scales_with_geom_term(self):
        """
        Doubling s'.geom_term must double the output.
        """
        s0, s1, s2 = self._chain()
        cluster = make_clustered([s0, s1, s2], voxel_size=0.3)
        prop = make_prop(kernel_radius=0.5)

        s2.geom_term = 0.5
        r_single = prop.propagate_segment(s1, 1, cluster, camera_samplers())

        s2.geom_term = 1.0
        r_double = prop.propagate_segment(s1, 1, cluster, camera_samplers())

        assert float(r_double.x) == pytest.approx(2.0 * float(r_single.x), rel=1e-4)

    def test_light_technique_segments_excluded_in_camera_pass(self):
        """
        LIGHT-technique segments must not be picked up in the CAMERA
        propagation pass — technique filtering is mandatory.
        """
        s0, s1, s2 = self._chain()
        s2.technique = SegmentTechnique.LIGHT
        cluster = make_clustered([s0, s1, s2], voxel_size=0.3)
        prop = make_prop(kernel_radius=0.5)
        samplers = camera_samplers()

        prop.propagate_segment(s1, 1, cluster, samplers)

        for call in samplers[SegmentTechnique.CAMERA].shift_invariant_bsdf.call_args_list:
            assert call[0][1] is not s2, "LIGHT segment must not appear in CAMERA pass"

    def test_uses_next_seg_mmis_weight_not_current(self):
        """
        The weight in the sum is s'.mmis_weight, not s.mmis_weight.
        Setting s.mmis_weight to something wild must not change the output.
        """
        s0, s1, s2 = self._chain()
        cluster = make_clustered([s0, s1, s2], voxel_size=0.3)
        prop = make_prop(kernel_radius=0.5)

        s1.mmis_weight = 1.0
        r_normal = prop.propagate_segment(s1, 1, cluster, camera_samplers())

        s1.mmis_weight = 999.0
        r_wild = prop.propagate_segment(s1, 1, cluster, camera_samplers())

        assert float(r_normal.x) == pytest.approx(float(r_wild.x), rel=1e-5)


# ── propagate_all_segments ────────────────────────────────────────────────────

class TestPropagateAllSegments:

    def test_camera_origin_segments_skipped(self):
        """
        Segments with x.is_camera=True must not have radiance_out written.
        """
        s0 = make_segment(
            x=make_surface_point(0, 0, 0, 0, 0, 1, is_camera=True),
            y=make_surface_point_with_mock_bsdf(0.1, 0, 0, 0, 0, 1),
        )
        s1 = make_segment(
            x=make_surface_point_with_mock_bsdf(0.1, 0, 0, 0, 0, 1),
            y=make_surface_point_with_mock_bsdf(0.2, 0, 0, 0, 0, 1),
        )
        s0.radiance_out = mi.Color3f(0.0)
        s1.mmis_weight = 1.0
        s1.radiance_in = mi.Color3f(1.0)

        cluster = make_clustered([s0, s1], voxel_size=0.3)
        prop = make_prop(kernel_radius=0.5)
        prop.propagate_all_segments(cluster, camera_samplers())

        # s0 must be untouched
        assert float(s0.radiance_out.x) == 0.0

    def test_light_origin_segments_skipped(self):
        s0 = make_segment(
            x=make_surface_point(0, 0, 0, 0, 0, 1, is_light=True),
            y=make_surface_point_with_mock_bsdf(0.1, 0, 0, 0, 0, 1),
        )
        s0.radiance_out = mi.Color3f(0.0)
        cluster = make_clustered([s0], voxel_size=0.3)
        prop = make_prop(kernel_radius=0.5)
        prop.propagate_all_segments(cluster, camera_samplers())
        assert float(s0.radiance_out.x) == 0.0

    def test_zero_radiance_in_gives_zero_out(self):
        """
        If all radiance_in are zero, all radiance_out must be zero —
        the estimator is linear in s'.radiance_in.
        """
        s0 = make_segment(
            x=make_surface_point(0, 0, 0, 0, 0, 1, is_camera=True),
            y=make_surface_point_with_mock_bsdf(0.1, 0, 0, 0, 0, 1),
        )
        s1 = make_segment(
            x=make_surface_point_with_mock_bsdf(0.1, 0, 0, 0, 0, 1),
            y=make_surface_point_with_mock_bsdf(0.2, 0, 0, 0, 0, 1),
            technique=SegmentTechnique.CAMERA,
        )
        s1.mmis_weight = 1.0
        s1.radiance_in = mi.Color3f(0.0)

        cluster = make_clustered([s0, s1], voxel_size=0.3)
        prop = make_prop(kernel_radius=0.5)
        prop.propagate_all_segments(cluster, camera_samplers())

        assert float(s1.radiance_out.x) == pytest.approx(0.0, abs=1e-8)

class TestIteratePropagation:

    def test_radiance_in_initialised_from_le(self):
        """
        Before any propagation pass, radiance_in must equal Le for all segments.
        iterate_propagation initialises this on line 9-10 of Algorithm 1.
        """
        s0, s1, s2= _make_chain()
        cluster = make_clustered([s0, s1, s2], voxel_size=0.1)
        prop = Propagation(kernel_radius=0.5, kernel_weight=0.7, num_prop_iterations=0)
        prop.iterate_propogation(cluster, camera_samplers())

        assert float(s2.radiance_in.x) == pytest.approx(1.0, abs=1e-6), \
            "s2.radiance_in must equal Le after init"
        assert float(s1.radiance_in.x) == pytest.approx(0.0, abs=1e-6)
        assert float(s0.radiance_in.x) == pytest.approx(0.0, abs=1e-6)

    def test_single_iteration_propagates_one_hop(self):
        """
        After k=1: s1 has Le=1, so segments in s1's y-cluster receive it.
        s2.x is in the same cluster as s1.y → s2 should have non-zero radiance_in.
        s3 is one hop further — it should NOT receive anything yet
        (its radiance_in comes from s2, which was 0 at the start of k=1).
        """
        s0, s1, s2 = _make_chain()
        cluster = make_clustered([s0, s1, s2], voxel_size=0.1)
        for i, s in enumerate([s0, s1, s2]):
            print(f"s{i}.x={float(s.x.p.x):.3f} cluster={cluster.endpoint_to_cluster[i*2+0]}")
            print(f"s{i}.y={float(s.y.p.x):.3f} cluster={cluster.endpoint_to_cluster[i*2+1]}")
        prop = Propagation(kernel_radius=0.5, kernel_weight=0.7, num_prop_iterations=1)
        prop.iterate_propogation(cluster, camera_samplers())

        assert float(s1.radiance_in.x) > 0.0
        # s0 has not received yet
        assert float(s0.radiance_in.x) == pytest.approx(0.0, abs=1e-6)

    def test_two_iterations_propagates_two_hops(self):
        s0, s1, s2 = _make_chain()
        cluster = make_clustered([s0, s1, s2], voxel_size=0.1)
        prop = Propagation(kernel_radius=0.5, kernel_weight=0.7, num_prop_iterations=2)
        prop.iterate_propogation(cluster, camera_samplers())
        print("FINAL s0.radiance_out:", s0.radiance_out)
        print("FINAL s1.radiance_out:", s1.radiance_out)
        print("FINAL s2.radiance_out:", s2.radiance_out)
        # k=1: s1 gets from s2; k=2: s0 gets from s1
        assert float(s0.radiance_out.x) > 0.0

    def test_more_iterations_means_more_radiance_at_far_segment(self):
        """
        With k=2, s3 receives radiance. With k=3, s3 also receives the
        contribution that bounced through an extra hop, so its radiance_in
        must be >= the k=2 case.
        """
        s0, s1, s2= _make_chain()
        cluster_2 = make_clustered([s0, s1, s2], voxel_size=0.1)
        prop_2 = Propagation(kernel_radius=0.5, kernel_weight=0.7, num_prop_iterations=2)
        prop_2.iterate_propogation(cluster_2, camera_samplers())
        r_k2 = float(s2.radiance_in.x)

        s0, s1, s2= _make_chain()
        cluster_3 = make_clustered([s0, s1, s2], voxel_size=0.1)
        prop_3 = Propagation(kernel_radius=0.5, kernel_weight=0.7, num_prop_iterations=3)
        prop_3.iterate_propogation(cluster_3, camera_samplers())
        r_k3 = float(s0.radiance_in.x)

        assert r_k3 >= r_k2, \
            f"More iterations must not reduce far radiance: k=2 gave {r_k2}, k=3 gave {r_k3}"

    def test_zero_le_gives_zero_radiance_everywhere(self):
        """
        If no segment has Le > 0, no radiance should propagate regardless of k.
        """
        s0, s1, s2 = _make_chain()
        s1.Le = mi.Color3f(0.0)  # override the emission
        cluster = make_clustered([s0, s1, s2], voxel_size=0.1)
        prop = Propagation(kernel_radius=0.5, kernel_weight=0.7, num_prop_iterations=4)
        prop.iterate_propogation(cluster, camera_samplers())

        for seg, name in [(s1, "s1"), (s2, "s2")]:
            assert float(seg.radiance_in.x) == pytest.approx(0.0, abs=1e-8), \
                f"{name}.radiance_in must be zero with no emission"

    def test_kernel_shrinks_once_per_iterate_call(self):
        """
        _update_kernel is called once at the end of iterate_propagation,
        regardless of num_prop_iterations. Kernel must shrink exactly once.
        """
        prop = Propagation(kernel_radius=1.0, kernel_weight=0.5, num_prop_iterations=4)
        s0 = make_segment(
            x=make_surface_point(0, 0, 0, 0, 0, 1, is_camera=True),
            y=make_surface_point_with_mock_bsdf(0.1, 0, 0, 0, 0, 1),
        )
        cluster = make_clustered([s0], voxel_size=0.1)
        prop.iterate_propogation(cluster, camera_samplers())

        assert prop.kernel_radius == pytest.approx(0.5, rel=1e-6), \
            "Kernel radius must shrink by kernel_weight exactly once per iterate call"
        expected_val = 1.0 / (math.pi * 0.5 ** 2)
        assert prop.kernel_val == pytest.approx(expected_val, rel=1e-6), \
            "kernel_val must reflect updated radius"

    def test_swap_resets_radiance_out_to_zero(self):
        """
        After swap, radiance_out must be zero for all interior segments —
        ready for the next propagation pass to write fresh values.
        """
        s0, s1, s2 = _make_chain()
        cluster = make_clustered([s0, s1, s2], voxel_size=0.1)
        prop = Propagation(kernel_radius=0.5, kernel_weight=0.7, num_prop_iterations=2)
        prop.iterate_propogation(cluster, camera_samplers())

        for seg, name in [(s1, "s1"), (s2, "s2")]:
            assert float(seg.radiance_out.x) == pytest.approx(0.0, abs=1e-8), \
                f"{name}.radiance_out must be zero after final swap"

    def test_camera_le_stays_as_radiance_in_after_swap(self):
        """
        Camera-origin segments must always have radiance_in = Le after swap,
        not whatever radiance_out was written (which should be zero anyway).
        """
        s0, s1, s2 = _make_chain()
        s0.Le = mi.Color3f(0.5)
        cluster = make_clustered([s0, s1, s2], voxel_size=0.1)
        prop = Propagation(kernel_radius=0.5, kernel_weight=0.7, num_prop_iterations=2)
        prop.iterate_propogation(cluster, camera_samplers())

        assert float(s0.radiance_in.x) == pytest.approx(0.5, abs=1e-6), \
            "Camera-origin segment radiance_in must always equal Le"