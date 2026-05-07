import mitsuba as mi
import drjit as dr

from src.segments import SurfacePoint, Segment

def test_two_bounce_segment():
    scene = mi.load_file('../samples/cbox/cbox.xml')

    sensor = scene.sensors()[0]
    film = sensor.film()
    film_size = film.crop_size()

    sampler = mi.load_dict({'type': 'independent'})
    sampler.seed(0, 1)

    found_valid_path = False

    for i in range(20):
        pixel = mi.Point2f(
            sampler.next_1d() * film_size.x,
            sampler.next_1d() * film_size.y
        )

        ray, weight = sensor.sample_ray(
            time=0.0,
            sample1=sampler.next_1d(),
            sample2=pixel / film_size,
            sample3=mi.Point2f(
                sampler.next_1d(),
                sampler.next_1d()
            )
        )

        si = scene.ray_intersect(ray)

        if not si.is_valid():
            continue

        surface_point_1 = SurfacePoint(si)

        bsdf = si.bsdf()

        bsdf_sample, bsdf_weight = bsdf.sample(
            mi.BSDFContext(),
            si,
            sampler.next_1d(),
            mi.Point2f(
                sampler.next_1d(),
                sampler.next_1d()
            )
        )

        wo = si.to_world(bsdf_sample.wo)

        ray2 = si.spawn_ray(wo)
        si2 = scene.ray_intersect(ray2)

        if not si2.is_valid():
            continue

        surface_point_2 = SurfacePoint(si2)

        segment = Segment(
            surface_point_1,
            surface_point_2
        )

        manual_dist = dr.norm(
            surface_point_2.p - surface_point_1.p
        )

        #
        # Assertions
        #

        assert segment.len > 0.0

        assert dr.abs(
            segment.len - manual_dist
        ) < 1e-5

        assert dr.abs(
            dr.norm(segment.dir) - 1.0
        ) < 1e-5

        assert segment.geom_term > 0.0

        found_valid_path = True
        break

    assert found_valid_path, \
        "Failed to find valid 2-bounce path"