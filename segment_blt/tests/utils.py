import pytest
import mitsuba as mi
from segments.primitives import Segment, SurfacePoint, make_endpoint_si, SegmentTechnique
from unittest.mock import MagicMock
from segments.cluster import Cluster
import numpy as np

class ZeroRng:
    def uniform(self, low, high, size):
        return np.zeros(size)

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

def make_surface_point(px, py, pz, nx, ny, nz, is_camera=False, is_light=False) -> SurfacePoint:
    si = make_endpoint_si(mi.Point3f(px, py, pz), mi.Vector3f(nx, ny, nz))
    return SurfacePoint(si = si, is_camera = is_camera, is_light = is_light)

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

@pytest.fixture
def simple_segments():
    """
    3 segments with known positions and normals.
    Segments 0 and 2 have x-endpoints that are spatially close
    and share the same normal — they should cluster together.
    Segment 1 is far away.
    """
    # seg 0: x endpoint near [0.1, 0.1, 0.1], normal up
    s0 = make_segment(
        x=make_surface_point(0.10, 0.10, 0.10,  0, 1, 0),
        y=make_surface_point(0.50, 0.50, 0.50,  0, 1, 0),
    )
    # seg 1: far away
    s1 = make_segment(
        x=make_surface_point(0.80, 0.80, 0.80,  1, 0, 0),
        y=make_surface_point(0.85, 0.85, 0.85,  1, 0, 0),
    )
    # seg 2: x endpoint near seg 0's x endpoint, same normal
    s2 = make_segment(
        x=make_surface_point(0.12, 0.11, 0.09,  0, 1, 0),
        y=make_surface_point(0.55, 0.55, 0.55,  0, 1, 0),
    )
    return [s0, s1, s2]


@pytest.fixture
def cluster(simple_segments):
    unit_aabb = make_aabb([0, 0, 0], [1, 1, 1])
    c = Cluster(c=5)
    c.set_scene_aabb(unit_aabb)
    c.set_segments(simple_segments)
    return c

def make_aabb(min_pt, max_pt) -> mi.BoundingBox3f:
    aabb = mi.BoundingBox3f()
    aabb.expand(mi.Point3f(*min_pt))
    aabb.expand(mi.Point3f(*max_pt))
    return aabb

def straight_chain():
    """
    Three collinear segments along z-axis, all normals pointing +z.
      s0: (0,0,0)->(0,0,1)  camera origin
      s1: (0,0,1)->(0,0,2)  interior
      s2: (0,0,2)->(0,0,3)  interior
    """
    s0 = make_segment(
        x=make_surface_point(0, 0, 0, 0, 0, 1, is_camera=True),
        y=make_surface_point_with_mock_bsdf(0, 0, 1, 0, 0, 1),
    )
    s1 = make_segment(
        x=make_surface_point_with_mock_bsdf(0, 0, 1, 0, 0, 1),
        y=make_surface_point_with_mock_bsdf(0, 0, 2, 0, 0, 1),
    )
    s2 = make_segment(
        x=make_surface_point_with_mock_bsdf(0, 0, 2, 0, 0, 1),
        y=make_surface_point_with_mock_bsdf(0, 0, 3, 0, 0, 1),
    )
    return s0, s1, s2

def _make_chain():
    # segments travel along x-axis, so normals must have x-component
    # use (1,0,0) normals so dot(n, dir) = 1
    s0 = make_segment(
        x=make_surface_point(0.0,  0, 0, 1, 0, 0, is_camera=True),
        y=make_surface_point_with_mock_bsdf(0.32, 0, 0, 1, 0, 0),
        technique=SegmentTechnique.CAMERA,
    )
    s1 = make_segment(
        x=make_surface_point_with_mock_bsdf(0.33, 0, 0, 1, 0, 0),  # near s0.y
        y=make_surface_point_with_mock_bsdf(0.65, 0, 0, 1, 0, 0),
        technique=SegmentTechnique.CAMERA,
    )
    s2 = make_segment(
        x=make_surface_point_with_mock_bsdf(0.66, 0, 0, 1, 0, 0),  # near s1.y
        y=make_surface_point_with_mock_bsdf(0.9,  0, 0, 1, 0, 0),
        technique=SegmentTechnique.CAMERA,
    )
    # s2 emits — after init s2.radiance_in=1
    # k=1: s1 gathers from s2 → s1.radiance_out > 0
    # k=2: s0 gathers from s1 (now has radiance) → s0.radiance_out > 0
    s2.Le = mi.Color3f(1.0)
    for s in [s0, s1, s2]:
        s.mmis_weight = 1.0
    return s0, s1, s2