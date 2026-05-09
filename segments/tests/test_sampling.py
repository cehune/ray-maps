import mitsuba as mi
import drjit as dr
from segments.src.sampler import * 

def test_light_path_returns_segments():
    """sample_light_path returns a non-empty list of Segment objects"""
    scene = mi.load_file('../samples/cbox/cbox.xml')
    sampler = mi.load_dict({'type': 'independent'})
    sampler.seed(0, 1)

    found_valid_path = False
    for _ in range(20):
        path = sample_light_path(scene, sampler)
        if len(path) > 0:
            found_valid_path = True
            break

    assert found_valid_path, "sample_light_path never returned a non-empty path"


def test_light_path_first_segment_is_from_emitter():
    """First segment must start from a light surface point"""
    scene = mi.load_file('../samples/cbox/cbox.xml')
    sampler = mi.load_dict({'type': 'independent'})
    sampler.seed(1, 1)

    for _ in range(20):
        path = sample_light_path(scene, sampler)
        if len(path) > 0:
            assert path[0].x.is_light, \
                "First segment x must be a light surface point"
            return

    assert False, "No valid path found"


def test_light_path_segment_geometry():
    """All segments have valid length, normalised direction, and positive geometry term"""
    scene = mi.load_file('../samples/cbox/cbox.xml')
    sampler = mi.load_dict({'type': 'independent'})
    sampler.seed(2, 1)

    for _ in range(20):
        path = sample_light_path(scene, sampler)
        if len(path) == 0:
            continue

        for seg in path:
            manual_dist = dr.norm(seg.y.p - seg.x.p)

            assert seg.len > 0.0
            assert dr.abs(seg.len - manual_dist) < 1e-5, \
                f"Segment length mismatch: {seg.len} vs {manual_dist}"
            assert dr.abs(dr.norm(seg.dir) - 1.0) < 1e-5, \
                "Segment direction is not normalised"
            assert seg.geom_term > 0.0, \
                "Geometry term must be positive"
        return

    assert False, "No valid path found"


def test_light_path_weight_is_finite_and_positive():
    # TODO: Remove, will break with MIS
    """All segment throughputs must be finite and positive"""
    scene = mi.load_file('../samples/cbox/cbox.xml')
    sampler = mi.load_dict({'type': 'independent'})
    sampler.seed(3, 1)

    for _ in range(20):
        path = sample_light_path(scene, sampler)
        if len(path) == 0:
            continue

        for seg in path:
            w = seg.throughput
            assert dr.all(dr.isfinite(w)), \
                f"Throughput contains non-finite values: {w}"
            assert dr.all(w >= 0.0), \
                f"Throughput contains negative values: {w}"
        return

    assert False, "No valid path found"


def test_light_path_segments_are_connected():
    """Consecutive segments must share an endpoint"""
    scene = mi.load_file('../samples/cbox/cbox.xml')
    sampler = mi.load_dict({'type': 'independent'})
    sampler.seed(4, 1)

    for _ in range(20):
        path = sample_light_path(scene, sampler)
        if len(path) < 2:
            continue

        for i in range(len(path) - 1):
            dist = dr.norm(path[i].y.p - path[i+1].x.p)
            assert dist < 1e-4, \
                f"Segments {i} and {i+1} are not connected: gap {dist}"
        return

    assert False, "No path with 2+ segments found"


def test_light_path_throughput_decreases_monotonically():
    # TODO: Remove will break with MIS
    """Throughput magnitude should not increase along the path (energy conservation)"""
    scene = mi.load_file('../samples/cbox/cbox.xml')
    sampler = mi.load_dict({'type': 'independent'})
    sampler.seed(5, 1)

    for _ in range(20):
        path = sample_light_path(scene, sampler)
        if len(path) < 2:
            continue

        for i in range(len(path) - 1):
            curr_lum = luminance_rgb(path[i].throughput)
            next_lum = luminance_rgb(path[i+1].throughput)
            assert next_lum <= curr_lum + 1e-4, \
                f"Throughput increased at segment {i}: {curr_lum} -> {next_lum}"
        return

    assert False, "No path with 2+ segments found"


def test_light_path_last_segment_hits_light_or_surface():
    # TODO: Remove, will break with MIS
    """Last segment endpoint must be either a light or a valid non-degenerate surface"""
    scene = mi.load_file('../samples/cbox/cbox.xml')
    sampler = mi.load_dict({'type': 'independent'})
    sampler.seed(6, 1)

    for _ in range(20):
        path = sample_light_path(scene, sampler)
        if len(path) == 0:
            continue

        last_seg = path[-1]
        # endpoint is either a light or a surface with a valid normal
        normal_len = dr.norm(last_seg.y.n)
        assert dr.abs(normal_len - 1.0) < 1e-4, \
            f"Last segment endpoint has invalid normal: {last_seg.y.n}"
        return

    assert False, "No valid path found"


def test_camera_path_returns_segments():
    """sample_camera_path returns a non-empty list of Segment objects"""
    scene = mi.load_file('../samples/cbox/cbox.xml')
    sensor = scene.sensors()[0]
    film = sensor.film()
    film_size = film.crop_size()

    sampler = mi.load_dict({'type': 'independent'})
    sampler.seed(7, 1)

    for _ in range(20):
        pixel = mi.Point2f(
            sampler.next_1d() * film_size.x,
            sampler.next_1d() * film_size.y
        )
        ray, ray_weight = sensor.sample_ray(
            time=0.0,
            sample1=sampler.next_1d(),
            sample2=pixel / film_size,
            sample3=mi.Point2f(sampler.next_1d(), sampler.next_1d())
        )
        si = scene.ray_intersect(ray)

        path = sample_camera_path(scene, sampler, ray, ray_weight, si)
        if len(path) > 0:
            return

    assert False, "sample_camera_path never returned a non-empty path"


def test_camera_path_first_segment_starts_at_camera():
    """First segment must start from the camera surface point"""
    scene = mi.load_file('../samples/cbox/cbox.xml')
    sensor = scene.sensors()[0]
    film = sensor.film()
    film_size = film.crop_size()

    sampler = mi.load_dict({'type': 'independent'})
    sampler.seed(8, 1)

    for _ in range(20):
        pixel = mi.Point2f(
            sampler.next_1d() * film_size.x,
            sampler.next_1d() * film_size.y
        )
        ray, ray_weight = sensor.sample_ray(
            time=0.0,
            sample1=sampler.next_1d(),
            sample2=pixel / film_size,
            sample3=mi.Point2f(sampler.next_1d(), sampler.next_1d())
        )
        si = scene.ray_intersect(ray)
        path = sample_camera_path(scene, sampler, ray, ray_weight, si)
        if len(path) > 0:
            assert path[0].x.is_camera, \
                "First segment x must be the camera"
            return

    assert False, "No valid camera path found"