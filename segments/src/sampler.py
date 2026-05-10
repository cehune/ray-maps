import mitsuba as mi
from primitives import *

def luminance_rgb(c: mi.Color3f) -> float:
    """Perceived luminance weights for RGB."""
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]

def generate_path(scene, sampler, starting_weight,starting_sp, si, technique = SegmentTechnique.CAMERA, max_depth=16, rr_start_depth = 3):
    """
    starting sp is either the camera or light
    si is the first surface interaction

    returns a path of disconnected segments with independant throughput from x to y
    """
    first_sp = SurfacePoint.from_intersection(si)
    segment_path = []

    # if it's a light, then we just return after setting the first hit to a light src
    emitter = si.emitter(scene)
    if emitter:
        first_sp.is_light = True
        first_sp.Le = emitter.eval(si)

    segment_path.append(Segment(starting_sp, first_sp, throughput=mi.Color3f(starting_weight), technique=technique))

    if first_sp.is_light:
        return segment_path
    
    # then we set up the loop
    prev_sp = first_sp

    for i in range(max_depth):
        # generate bsdf at the point
        bsdf = si.bsdf()
        bs, bsdf_weight = bsdf.sample(
            mi.BSDFContext(), si,
            sampler.next_1d(),
            mi.Point2f(sampler.next_1d(), sampler.next_1d())
        )

        if bs.pdf == 0 or dr.all(bsdf_weight == 0):
            break

        # in standard path tracing we would normally just modify weight absed on bsdf weight and carry forward
        # since segments are disocnnected, only use a local weight
        local_weight = bsdf_weight

        # russian roulette
        if i >= rr_start_depth:
            q = min(0.97, luminance_rgb(local_weight))
            if sampler.next_1d() >= q:
                break
            local_weight /= q

        # bs.wo is the sample direction outgoing in local space, convert to world
        world_wo = si.to_world(bs.wo)
        si_next = scene.ray_intersect(si.spawn_ray(world_wo))
        if not si_next.is_valid():
            # TODO: get scene evnironemnt contribution
            break

        curr_sp = SurfacePoint.from_intersection(si_next)

        emitter_hit = si_next.emitter(scene)
        if emitter_hit:
            curr_sp.is_light = True
            curr_sp.Le = emitter_hit.eval(si_next)

        segment_path.append(
            Segment(prev_sp, curr_sp, throughput=mi.Color3f(local_weight), technique=technique)
        )

        if curr_sp.is_light or dr.norm(curr_sp.p - prev_sp.p) < 1e-4: # same spot basically
            break

        prev_sp = curr_sp
        si = si_next

    return segment_path

def sample_light_path(scene, sampler, max_depth=16, rr_start_depth=3):
    # just get an emitter, weighted by their power
    emitter_sample = scene.sample_emitter(sampler.next_1d())
    emitter = emitter = scene.emitters()[emitter_sample[0]]
    emitter_pdf = emitter_sample[1] # probability of having selected that emitter
    # dividing by this pdf later is importance sampling, want to sample from bright lights
    # more often

    if emitter_pdf == 0: # non valid, shouldn't mmatter for well defined scenes
        return []
    
    # sample outgoing direction from emitter (cosine weighted over hemisphere)
    # this would give local, and then we would have to rotate by the emitter normal
    # and retrieve the pdf for that direction. can just directly do emitter.sample
    # but with sample ray we just get a random one on it's surface
    emitted_ray, emitted_ray_weight = emitter.sample_ray(
        time=0.0,
        sample1=sampler.next_1d(),
        sample2=mi.Point2f(sampler.next_1d(), sampler.next_1d()),
        sample3=mi.Point2f(sampler.next_1d(), sampler.next_1d())
    )

    if dr.all(emitted_ray_weight == 0):
        return []

    light_sp = SurfacePoint(
        p = emitted_ray.o,
        n = -emitted_ray.d,
        is_light = True
    )

    si = scene.ray_intersect(emitted_ray)

    if not si.is_valid():
        return []
    
    # generate weight cost theta / pdfs
    weight = emitted_ray_weight / emitter_pdf
    
    return generate_path(scene, sampler, weight, light_sp, si, 
                         technique=SegmentTechnique.LIGHT, max_depth=max_depth, 
                         rr_start_depth=rr_start_depth)

def sample_camera_path(scene, sampler, ray, ray_weight, si, max_depth=16, rr_start_depth=3):
    if not si.is_valid():
        return []
    
    cam_sp = SurfacePoint(p=ray.o, n=mi.Vector3f(-ray.d))
    cam_sp.is_camera = True

    return generate_path(scene, sampler, ray_weight, cam_sp, si,
                          technique=SegmentTechnique.CAMERA,
                          max_depth=max_depth,
                          rr_start_depth=rr_start_depth)


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