import mitsuba as mi
from primitives import *

def luminance_rgb(c: mi.Color3f) -> float:
    """Perceived luminance weights for RGB."""
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def sample_camera_path(scene, sampler, ray, max_depth=16, rr_start_depth=3):
    # TODO: Change to BDPT which means adding ray weight into here, not applied after 
    # in the main render loop 
    segment_path = []
    weight = mi.Color3f(1.0)

    si = scene.ray_intersect(ray)
    if not si.is_valid():
        return segment_path
    else:
        # generate first segment, which starts at the camera
        s0 = SurfacePoint()
        s0.p = ray.origin
        s0.n = ray.direction
        s0.is_camera = True
        segment_path.append(Segment(s0, si), 1.0)

    prev_sp = SurfacePoint(si)

    for i in range(max_depth):
        # sample next direction first
        bsdf = si.bsdf()
        bsdf_sample, bsdf_weight = bsdf.sample(
            mi.BSDFContext(), si,
            sampler.next_1d(),
            mi.Point2f(sampler.next_1d(), sampler.next_1d())
        )

        if bsdf_sample.pdf == 0 or dr.all(bsdf_weight == 0):
            break

        # russian roulette
        if i >= rr_start_depth:
            q = min(1.0, luminance_rgb(weight))
            if sampler.next_1d() >= q:
                break
            weight /= q

        weight *= bsdf_weight # update weight before building segment


        wo = si.to_world(bsdf_sample.wo)
        next_ray = si.spawn_ray(wo)
        si_next = scene.ray_intersect(next_ray)

        if not si_next.is_valid():
            break

        curr_sp = SurfacePoint(si_next)

        # check emitter before building segment
        if si_next.emitter(scene):
            curr_sp.is_light = True
            curr_sp.Le = si_next.emitter(scene).eval(si_next)

        try:
            seg = Segment(prev_sp, curr_sp, throughput=mi.Color3f(weight))
            segment_path.append(seg)
        except ValueError:
            break

        if curr_sp.is_light:
            break

        prev_sp = curr_sp
        si = si_next
    return segment_path

def evaluate_path_base(segments: list[Segment]) -> mi.Color3f:
    # stupid estimator that just selects last segment's emitter contribution
    if not segments:
        return mi.Color3f(0.0)

    last_seg = segments[-1]
    if not last_seg.y.is_light:
        return mi.Color3f(0.0)

    Le = last_seg.y.Le
    if dr.all(Le == mi.Color3f(0.0)):
        return mi.Color3f(0.0)

    # throughput on last segment already contains full bsdf_weight product
    return last_seg.throughput * Le

def render_pixel(scene, sampler, ray) -> mi.Color3f:
    segments = sample_camera_path(scene, sampler, ray)
    return evaluate_path_base(segments)