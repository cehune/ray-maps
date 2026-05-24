import mitsuba as mi
from segments.primitives import *

def luminance_rgb(c: mi.Color3f) -> float:
    """Perceived luminance weights for RGB."""
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]

def generate_path(scene, sampler, starting_weight,starting_sp, si, technique = SegmentTechnique.CAMERA, max_depth=8):
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

        ctx = mi.BSDFContext(
            mi.TransportMode.Importance if technique == SegmentTechnique.LIGHT
            else mi.TransportMode.Radiance
        )
        bs, bsdf_weight = bsdf.sample(
            ctx, si,
            sampler.next_1d(),
            mi.Point2f(sampler.next_1d(), sampler.next_1d())
        )

        if bs.pdf == 0 or dr.all(bsdf_weight == 0):
            break

        # normally put russian roulette here, but segments don't use RR

        # bs.wo is the sample direction outgoing in local space, convert to world
        world_wo = si.to_world(bs.wo)
        si_next = scene.ray_intersect(si.spawn_ray(world_wo))
        if not si_next.is_valid():
            # TODO: get scexe evnironemnt contribution
            break

        curr_sp = SurfacePoint.from_intersection(si_next)

        emitter_hit = si_next.emitter(scene)
        if emitter_hit:
            curr_sp.is_light = True
            curr_sp.Le = emitter_hit.eval(si_next)
        try:
            seg = Segment(prev_sp, curr_sp, throughput=mi.Color3f(bsdf_weight), technique=technique)
            segment_path.append(seg)

        except ValueError:
            break # just if its too short

        if (technique == SegmentTechnique.CAMERA and curr_sp.is_light): # same spot basically
            break

        prev_sp = curr_sp
        si = si_next

    return segment_path

def sample_light_path(scene, sampler, max_depth=8):
    # just get an emitter, weighted by their power
    emitter_sample = scene.sample_emitter(sampler.next_1d())
    emitter = emitter = scene.emitters()[emitter_sample[0]]
    emitter_pdf = emitter_sample[1] # probability of having selected that emitter
    # dividing by this pdf later is importance sampling, want to sample from bright lights
    # more often

    if emitter_pdf == 0: # non valid, shouldn't mmatter for well defined scenes
        return []
    
    # Rather than use sample_ray, get direction and position directly
    # or else, weight includes the Le. We need this separate to get the normal, cant
    # just use the direction as a normal
    # ps has the normal and position
    ps: mi.PositionSample3f = emitter.sample_position(
        ref=0.0,
        ds=mi.Point2f(sampler.next_1d(), sampler.next_1d()),
    )[0]
    if ps.pdf == 0:
        return []

    # Get the direction (again, would be from sample ray otherwise, we need it seaprate)
    local_dir = mi.warp.square_to_cosine_hemisphere(
        mi.Point2f(sampler.next_1d(), sampler.next_1d())
    )
    dir_pdf = mi.warp.square_to_cosine_hemisphere_pdf(local_dir)
    world_dir = mi.Frame3f(ps.n).to_world(local_dir)

    # Build a proper SI at the light point with correct normal + wi
    light_si = make_endpoint_si(ps.p, ps.n)
    light_si.wi = local_dir   # outgoing dir in local frame
    Le_at_light = emitter.eval(light_si)

    light_sp = SurfacePoint(si=light_si, is_light=True, Le=Le_at_light)

    # Trace first segment
    emitted_ray = mi.Ray3f(ps.p, world_dir)
    si = scene.ray_intersect(emitted_ray)
    if not si.is_valid():
        return []

    # Starting throughput: cos(θ) / (emitter_pdf · pos_pdf · dir_pdf)
    cos_theta = dr.abs(dr.dot(ps.n, world_dir))
    weight = mi.Color3f(cos_theta / (emitter_pdf * ps.pdf * dir_pdf))

    return generate_path(scene, sampler, weight, light_sp, si,
                         technique=SegmentTechnique.LIGHT, max_depth=max_depth)

def sample_camera_path(scene, sampler, ray, ray_weight, si, max_depth=16, rr_start_depth=3):
    if not si.is_valid():
        return []
    
    cam_sp = SurfacePoint(
        si = make_endpoint_si(ray.o,-ray.d),
        is_camera = True)
    return generate_path(scene, sampler, ray_weight, cam_sp, si,
                          technique=SegmentTechnique.CAMERA,
                          max_depth=max_depth)


def evaluate_path_base(segments: list[Segment]) -> mi.Color3f:
    # TODO: drop once the MMIS and propogation work!
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