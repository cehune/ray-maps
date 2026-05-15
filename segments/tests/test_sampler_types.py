from segments.src.sampler_types import SequentialSampler
from segments.tests.utils import make_segment, make_surface_point_with_mock_bsdf, straight_chain
import drjit as dr
import pytest
import mitsuba as mi
import math
from unittest.mock import MagicMock

class TestSequentialSampler:

    def test_conditional_pdf_non_negative(self):
        """p_c must be non-negative and finite for any valid geometry."""
        sampler = SequentialSampler()

        auxiliary = make_segment(
            x=make_surface_point_with_mock_bsdf(0, 0, 0,  0, 0, 1),
            y=make_surface_point_with_mock_bsdf(0, 0, 1,  0, 0, 1, bsdf_pdf_value=0.5),
        )
        segment = make_segment(
            x=make_surface_point_with_mock_bsdf(0, 0, 1,  0, 0, 1),
            y=make_surface_point_with_mock_bsdf(0.1, 0, 2, 0, 0, 1),
        )

        pdf = sampler.conditional_pdf(segment, auxiliary)

        assert pdf >= 0.0,         f"PDF must be non-negative, got {pdf}"
        assert pdf == pdf,         "PDF must not be NaN"
        assert pdf < float('inf'), f"PDF must be finite, got {pdf}"

    def test_conditional_pdf_grazing_angle_near_zero(self):
        """
        Cosine at segment.y near zero when ray arrives parallel to the surface.
        Ray goes horizontally from (0,0,1) to (1,0,1).
        segment.y.n = (0,0,1) — perpendicular to ray → cosine = 0.
        """
        sampler = SequentialSampler()

        auxiliary = make_segment(
            x=make_surface_point_with_mock_bsdf(0, 0, 0,  0, 0, 1),
            y=make_surface_point_with_mock_bsdf(0, 0, 1,  0, 0, 1, bsdf_pdf_value=1.0),
        )
        segment = make_segment(
            x=make_surface_point_with_mock_bsdf(0,   0, 1,  0, 0, 1),
            y=make_surface_point_with_mock_bsdf(1.0, 0, 1,  0, 0, 1),
        )

        pdf = sampler.conditional_pdf(segment, auxiliary)

        assert pdf < 1e-6, \
            f"Grazing arrival should give near-zero PDF, got {pdf}"

    def test_conditional_pdf_lambertian_known_value(self):
        """
        Mock bsdf_pdf = 1/pi (Lambertian, normal incidence).
        Segment goes straight up: dist=1, cosine at segment.y = 1.
        Expected: (1/pi) * 1 / 1^2 = 1/pi.
        """
        sampler = SequentialSampler()

        expected_bsdf_pdf = float(1.0 / dr.pi)

        auxiliary = make_segment(
            x=make_surface_point_with_mock_bsdf(0, 0, -1,  0, 0, 1),
            y=make_surface_point_with_mock_bsdf(0, 0,  0,  0, 0, 1, bsdf_pdf_value=expected_bsdf_pdf),
        )
        segment = make_segment(
            x=make_surface_point_with_mock_bsdf(0, 0, 0,  0, 0, 1),
            y=make_surface_point_with_mock_bsdf(0, 0, 1,  0, 0, 1),
        )

        pdf = sampler.conditional_pdf(segment, auxiliary)
        expected = expected_bsdf_pdf * 1.0 / 1.0  # cosine=1, dist_sq=1

        assert abs(pdf - expected) < 1e-5, \
            f"Expected {expected:.6f}, got {pdf:.6f}"
        

    def test_non_negative_finite(self):
        sampler = SequentialSampler()
        _, seg, next_seg = straight_chain()
        result = sampler.shift_invariant_bsdf(seg, next_seg)
        for ch in [result.x, result.y, result.z]:
            assert float(ch) >= 0.0
            assert math.isfinite(float(ch))

    def test_evaluates_at_y_of_seg_not_x(self):
        """
        Camera convention: BSDF must be evaluated at seg.y, never seg.x.
        """
        sampler = SequentialSampler()
        _, seg, next_seg = straight_chain()
        sampler.shift_invariant_bsdf(seg, next_seg)
        seg.x.si.bsdf.assert_not_called()
        seg.y.si.bsdf.assert_called()

    def test_zero_for_none_bsdf(self):
        sampler = SequentialSampler()
        _, seg, next_seg = straight_chain()
        seg.y.si.bsdf = MagicMock(return_value=None)
        result = sampler.shift_invariant_bsdf(seg, next_seg)
        assert float(result.x) == 0.0
        assert float(result.y) == 0.0
        assert float(result.z) == 0.0

    def test_wi_is_minus_seg_dir(self):
        """
        wi at seg.y must be -seg.dir (light arriving from seg.x).
        seg goes (0,0,1)->(0,0,2), dir=(0,0,1), so wi_local.z=-1.
        """
        sampler = SequentialSampler()
        _, seg, next_seg = straight_chain()

        captured = {}
        original = seg.y.si.bsdf().eval

        def capture(si, wo):
            captured['wi'] = mi.Vector3f(si.wi)
            return original(si, wo)

        seg.y.si.bsdf().eval = capture
        sampler.shift_invariant_bsdf(seg, next_seg)

        assert float(captured['wi'].z) == pytest.approx(-1.0, abs=1e-4)
        assert float(captured['wi'].x) == pytest.approx( 0.0, abs=1e-4)
        assert float(captured['wi'].y) == pytest.approx( 0.0, abs=1e-4)

    def test_wo_points_toward_next_y(self):
        """
        Shift-invariant approximation: x' ≈ y, so wo points y -> y'.
        seg.y=(0,0,2), next_seg.y=(0,0,3) → wo_local.z=+1.
        """
        sampler = SequentialSampler()
        _, seg, next_seg = straight_chain()

        captured = {}
        original = seg.y.si.bsdf().eval

        def capture(si, wo):
            captured['wo'] = mi.Vector3f(wo)
            return original(si, wo)

        seg.y.si.bsdf().eval = capture
        sampler.shift_invariant_bsdf(seg, next_seg)

        assert float(captured['wo'].z) == pytest.approx(1.0, abs=1e-4)
        assert float(captured['wo'].x) == pytest.approx(0.0, abs=1e-4)
        assert float(captured['wo'].y) == pytest.approx(0.0, abs=1e-4)