from segments.src.sampler_types import SequentialSampler
from segments.tests.utils import make_segment, make_surface_point_with_mock_bsdf
import drjit as dr

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