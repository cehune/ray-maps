import pytest
import mitsuba as mi
from segments.src.primitives import Segment, SurfacePoint, make_endpoint_si
from unittest.mock import MagicMock

@pytest.fixture
def simple_scene():
    """Cornell-box-like scene with a visible area light."""
    return mi.load_dict({
        'type': 'scene',
        'integrator': {'type': 'path'},
        'sensor': {
            'type': 'perspective',
            'fov': 45,
            'to_world': mi.ScalarTransform4f.look_at(
                origin=[0, 0, 4],
                target=[0, 0, 0],
                up=[0, 1, 0]
            ),
            'film': {
                'type': 'hdrfilm',
                'width': 32,
                'height': 32,
                'pixel_format': 'rgb',
            },
            'sampler': {'type': 'independent', 'sample_count': 1}
        },
        # floor
        'floor': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f.translate([0, -1, 0]) @
                        mi.ScalarTransform4f.rotate([1,0,0], 90),
            'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0.8, 0.8, 0.8]}}
        },
        # area light facing camera — easy to hit
        'light': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f.translate([0, 0, -1]) @
                        mi.ScalarTransform4f.scale([0.5, 0.5, 1]),
            'emitter': {
                'type': 'area',
                'radiance': {'type': 'rgb', 'value': [10.0, 10.0, 10.0]}
            }
        }
    })

@pytest.fixture
def scene_sensor_sampler(simple_scene):
    sensor  = simple_scene.sensors()[0]
    sampler = mi.load_dict({'type': 'independent', 'sample_count': 1})
    sampler.seed(0, 32 * 32)
    return simple_scene, sensor, sampler

def make_surface_point(px, py, pz, nx, ny, nz) -> SurfacePoint:
    si = make_endpoint_si(mi.Point3f(px, py, pz), mi.Vector3f(nx, ny, nz))
    return SurfacePoint(si = si)

def make_surface_point_with_mock_bsdf(px, py, pz, nx, ny, nz, bsdf_pdf_value=1.0):
    """
    SurfacePoint with a fully mocked si so bsdf().eval_pdf() returns bsdf_pdf_value.
    Position and normal are real so all the math in conditional_pdf works correctly.
    """
    mock_bsdf = MagicMock()
    mock_bsdf.eval_pdf.return_value = bsdf_pdf_value

    mock_si = MagicMock(spec=mi.SurfaceInteraction3f)
    mock_si.p = mi.Point3f(px, py, pz)
    mock_si.n = mi.Normal3f(nx, ny, nz)
    mock_si.bsdf.return_value = mock_bsdf  # mock_si.bsdf() → mock_bsdf

    sp = SurfacePoint(si=mock_si)
    return sp


def make_segment(x: SurfacePoint, y: SurfacePoint, **kwargs) -> Segment:
    return Segment(x=x, y=y, **kwargs)