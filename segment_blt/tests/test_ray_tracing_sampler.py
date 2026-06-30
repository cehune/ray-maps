"""
Regression tests for the Eq.13 ray-tracing segment sampler against an analytic
sphere (closed-form ground truth). Mirrors validation/validate_eq13_sampler.py
but exercises the REAL sample_surface_point / sample_segment_from_point.

Scene = an INWARD-facing sphere (flip_normals), i.e. a spherical enclosure: x is
on the wall, the cosine-sampled direction about x's (inward) normal crosses the
interior and ray-casts to the far wall. With an outward-facing sphere every ray
escapes to infinity and no segment is ever formed.

Tolerances are loose because mitsuba's scalar_rgb variant is single precision
(float32): the invariant ratio cancels to pi only to ~1e-5, not machine eps.

Run in your venv:  pytest tests/test_ray_tracing_sampler.py -q
"""
import math
import numpy as np
import pytest

mi = pytest.importorskip("mitsuba")
mi.set_variant("scalar_rgb")

from segments.segment_gen_sampler import sample_surface_point, sample_segment_from_point
from segments.primitives import SegmentTechnique

R = 1.0

def _sphere_scene():
    # flip_normals => normals point inward; we sample the interior (enclosure).
    return mi.load_dict({
        "type": "scene",
        "shape": {"type": "sphere", "radius": R, "center": [0, 0, 0],
                  "flip_normals": True,
                  "bsdf": {"type": "diffuse"}},
    })

def _pick(scene, sampler):
    """Mirror _add_ray_tracing_segments' shape pick (single-shape sphere here)."""
    shapes = scene.shapes()
    total = float(sum(float(s.surface_area()) for s in shapes))
    return sample_surface_point(scene, sampler, total, shapes[0])


def _draw(n):
    scene = _sphere_scene()
    sampler = mi.load_dict({"type": "independent"})
    sampler.seed(0, n * 8)
    ratios, lengths = [], []
    tries = 0
    while len(lengths) < n and tries < n * 50:
        tries += 1
        res = _pick(scene, sampler)
        if res is None:
            continue
        point, p_x = res                                    # (SurfacePoint, 1/|M|)
        out = sample_segment_from_point(scene, point, sampler, SegmentTechnique.CAMERA)
        if out is None:
            continue
        seg, p_y_given_x = out
        # invariant: geom_term / p(y|x) == pi  (p(x) cancels separately)
        ratios.append(float(seg.geom_term) / p_y_given_x)
        lengths.append(float(seg.len))
    return np.array(ratios), np.array(lengths)

def test_invariant_geom_over_pdf_is_pi():
    """G(s)/p(y|x) is the constant pi for the geometry-importance sampler."""
    ratios, _ = _draw(5_000)
    assert len(ratios) > 1000, f"sampler produced too few segments ({len(ratios)})"
    # The invariant is ratio == pi. In float32 a small fraction of grazing hits
    # (cos_y ~ 0) amplify rounding in |n_y·dir|/cos_y into a heavy tail, so the
    # raw std is outlier-driven. Validate with robust statistics: the median IS
    # pi and the bulk (IQR) is tight. The shading-normal/Jacobian bug instead
    # shifts the median far from pi and spreads the bulk over O(100) (see the
    # harness), so this still cleanly separates correct from broken.
    assert abs(np.median(ratios) - math.pi) < 1e-2
    iqr = np.subtract(*np.percentile(ratios, [75, 25]))
    assert iqr < 1e-2, f"ratio not near-constant (IQR={iqr:.2e}) -> normal/Jacobian bug"
    assert abs(ratios.mean() - math.pi) < 5e-2

def test_sphere_chord_length_law():
    """Cosine entry + ray cast => E[len]=4R/3, p(l)=l/(2R^2)."""
    _, lengths = _draw(20_000)
    assert len(lengths) > 1000, f"sampler produced too few segments ({len(lengths)})"
    assert abs(lengths.mean() - 4/3 * R) < 1e-2
    bins = 30; edges = np.linspace(0, 2*R, bins + 1)
    obs, _ = np.histogram(lengths, edges)
    exp = len(lengths) * np.diff(edges**2 / (4*R**2))
    chi2 = ((obs - exp) ** 2 / exp).sum()
    assert chi2 < 3 * (bins - 1), f"chord-length pdf mismatch: chi2/dof={chi2/(bins-1):.2f}"

def test_p_x_is_inverse_total_area():
    """sample_surface_point's p(x) must equal 1/|M| = 1/(4 pi R^2) for the sphere."""
    scene = _sphere_scene()
    sampler = mi.load_dict({"type": "independent"}); sampler.seed(0, 64)
    res = _pick(scene, sampler)
    assert res is not None
    _, p_x = res
    assert abs(p_x - 1.0 / (4 * math.pi * R * R)) < 1e-4
