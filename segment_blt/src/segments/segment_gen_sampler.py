import mitsuba as mi
from segments.primitives import *

def luminance_rgb(c: mi.Color3f) -> float:
    """Perceived luminance weights for RGB."""
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]

def _area_pdf_of_light_hit(scene, ref_si, si_light):
    """Area-measure pdf of NEE-sampling the point si_light (from anywhere —
    the area-measure pdf is reference-independent: emitter pick pdf ×
    position pdf). Needed for MIS between BSDF-found light hits and NEE
    segments: both end on the emitter, so both techniques' pdfs go into the
    MMIS denominator.

    NOTE: do NOT use mi.DirectionSample3f(scene, si, ref) +
    scene.pdf_emitter_direction here — that constructor segfaults in
    mitsuba 3.8 scalar_rgb."""
    try:
        shape = si_light.shape
        if shape is None:
            return 0.0
        ps = mi.PositionSample3f(si_light)
        p_pos = float(shape.pdf_position(ps))        # 1/area for uniform shapes
        if p_pos <= 0.0:
            return 0.0
        n_emitters = max(len(scene.emitters()), 1)   # uniform emitter selection
        return p_pos / n_emitters
    except Exception:
        return 0.0


def make_nee_segment(scene, sampler, si, vertex_sp):
    """Sample the emitter from vertex_sp (NEE) and return a light-terminal
    segment carrying Le, or None (occluded / zero pdf / degenerate).

    The segment is a normal camera-technique merge source: its y is pinned
    to Le, its geom term is real, and conditional_pdf adds its stored
    nee_parea so the MMIS denominator MIS-weights it against BSDF-found
    light hits automatically. is_nee=True keeps it out of classical-link
    routing (the near-field stratum is covered by BSDF children alone) and
    out of readout path chains.
    """
    ds, spec = scene.sample_emitter_direction(
        si, mi.Point2f(sampler.next_1d(), sampler.next_1d()), True)
    if float(ds.pdf) <= 0.0 or dr.all(spec == 0):
        return None                       # occluded or unsampleable
    cos_l = float(dr.abs(dr.dot(ds.n, ds.d)))
    dist = float(ds.dist)
    if cos_l < 1e-8 or dist < 1e-6:
        return None
    light_si = make_endpoint_si(ds.p, ds.n)
    light_sp = SurfacePoint(si=light_si, is_light=True,
                            Le=mi.Color3f(spec * ds.pdf))   # spec = Le/pdf_ω
    try:
        seg = Segment(vertex_sp, light_sp, throughput=mi.Color3f(0.0),
                      technique=SegmentTechnique.CAMERA)
    except ValueError:
        return None
    seg.is_nee = True
    seg.nee_parea = float(ds.pdf) * cos_l / (dist * dist)   # area measure
    return seg


def generate_path(scene, sampler, starting_weight,starting_sp, si, technique = SegmentTechnique.CAMERA, max_depth=8, nee_out=None):
    """
    starting sp is either the camera or light
    si is the first surface interaction

    returns a path of disconnected segments with independant throughput from x to y
    nee_out: optional list — when given (camera technique), one NEE segment
    per path vertex is appended to it (kept OUT of the returned chain so
    path-structured consumers like evaluate_path_base / the depth-d readout
    are unaffected).
    """
    first_sp = SurfacePoint.from_intersection(si)
    segment_path = []

    do_nee = nee_out is not None and technique == SegmentTechnique.CAMERA

    # if it's a light, then we just return after setting the first hit to a light src
    emitter = si.emitter(scene)
    if emitter:
        first_sp.is_light = True
        first_sp.Le = emitter.eval(si)

    segment_path.append(Segment(starting_sp, first_sp, throughput=mi.Color3f(starting_weight), technique=technique))

    if first_sp.is_light:
        # BSDF-found light hit: store the NEE area pdf of this y for MIS
        segment_path[-1].nee_parea = _area_pdf_of_light_hit(
            scene, starting_sp.si, si) if do_nee else 0.0
        return segment_path

    if do_nee:
        nee = make_nee_segment(scene, sampler, si, first_sp)
        if nee is not None:
            nee_out.append(nee)

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
            if curr_sp.is_light and do_nee:
                # MIS partner pdf for this BSDF-found light hit
                seg.nee_parea = _area_pdf_of_light_hit(scene, si, si_next)
            segment_path.append(seg)

        except ValueError:
            break # just if its too short

        if (technique == SegmentTechnique.CAMERA and curr_sp.is_light): # same spot basically
            break

        if do_nee:
            nee = make_nee_segment(scene, sampler, si_next, curr_sp)
            if nee is not None:
                nee_out.append(nee)

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

def sample_camera_path(scene, sampler, ray, ray_weight, si, max_depth=16, rr_start_depth=3,
                       nee_out=None):
    if not si.is_valid():
        return []

    cam_sp = SurfacePoint(
        si = make_endpoint_si(ray.o,-ray.d),
        is_camera = True)
    return generate_path(scene, sampler, ray_weight, cam_sp, si,
                          technique=SegmentTechnique.CAMERA,
                          max_depth=max_depth, nee_out=nee_out)


def evaluate_path_base(segments: list[Segment]) -> mi.Color3f:
    # BSDF-only unidirectional path tracer (no NEE). Unbiased but high-variance
    # for small lights — fine as a convergence reference.
    if not segments:
        return mi.Color3f(0.0)

    last_seg = segments[-1]
    if not last_seg.y.is_light:
        return mi.Color3f(0.0)

    Le = last_seg.y.Le
    if dr.all(Le == mi.Color3f(0.0)):
        return mi.Color3f(0.0)

    # Each segment.throughput holds only its own bounce weight (segment 0 = ray_weight,
    # later segments = per-bounce bsdf_weight). The path integrand is the product.
    throughput = mi.Color3f(1.0)
    for seg in segments:
        throughput = throughput * seg.throughput
    return throughput * Le

def render_pixel(scene, sampler, ray) -> mi.Color3f:
    segments = sample_camera_path(scene, sampler, ray)
    return evaluate_path_base(segments)