import pytest
import mitsuba as mi

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
