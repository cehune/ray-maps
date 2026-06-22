import mitsuba as mi
import drjit as dr
from segments.primitives import Segment, SegmentTechnique, MIN_SEG_LEN
from abc import ABC, abstractmethod


class Sampler(ABC):
    technique_type: SegmentTechnique

    @staticmethod
    def build_registry(*samplers: 'Sampler') -> dict[SegmentTechnique, 'Sampler']:
        return {s.technique_type: s for s in samplers}

class SequentialSampler(Sampler):
    technique_type = SegmentTechnique.CAMERA

    # should just do the pdf move from mmis
    def conditional_pdf(self, segment: Segment, auxiliary: Segment) -> float:
        """
        p(segment | auxiliary): probability that auxiliary spawned segment.
        this is the equation 15
        segment is our current, auxiliary is the last

        ppaer uses ups style vertex pertubation, so we just pretend and set xi+1 = yi
        and then keep the pdf as if you sampled from the kernel. Recall via MMIS
        the pk term will cancel out with the numerator because it is the inverse of 
        the kernel, so just deal with the rest of the equation

        pk is the same for every segment in the voxel cluster and it is UNIFORM
        so if every single segment has the same pk probability of landing at the x i+1
        from yi, then its a constant we can pull out. 

        with s' as the connection between yi and xi+1
        so the dot product is with the connection and then the far next endpoint

        we have pw(s'|yi, si) |n_yi+1 dot s'| / {dist between connection y+1 and xi+1 squared}
        """

        if auxiliary.y.is_camera or auxiliary.y.is_light:
            return 0.0

        # NOTE: the NEE / to-light-bridge mass used to be folded in here as
        # `+ p_nee`. It is now its own path-construction technique
        # (BridgeSampler, paper §S2 / Eq. S8); the MMIS caller adds it once per
        # camera-side auxiliary, independent of this technique dispatch. This
        # sampler returns ONLY the sequential-continuation pdf p(segment|aux).

        # Distance from auxiliary.y to segment.y — this is the connection vector
        # used in the area-measure conversion.  For adjacent segments in the
        # sequential model auxiliary.y == segment.x so this equals segment.len,
        # but using the actual distance is correct for arbitrary auxiliaries in
        # the same cluster.
        delta  = segment.y.p - auxiliary.y.p
        len_sq = float(dr.squared_norm(delta))
        # GUARD SYMMETRY: must match the Segment admission threshold exactly.
        # For the true parent, this distance IS segment.len, and Segment
        # guarantees len >= MIN_SEG_LEN — so this guard can never delete a
        # parent term for a segment that exists.
        if len_sq < MIN_SEG_LEN * MIN_SEG_LEN:
            return 0.0

        # cant store outgoing direction local to the segment because we assume it
        # happens at the auxiliary endpoint, using auxiliary.y.n, check with Wenyou
        wo_world = dr.normalize(delta)   # unit connection vector from auxiliary.y → segment.y
        wo_local = mi.Frame3f(auxiliary.y.n).to_local(wo_world)

        # auxiliary.y.si is shared across MMIS calls — save/restore si.wi so we
        # don't leave a stale wi behind for the next caller. Cheap, no allocation.
        si = auxiliary.y.si
        saved_wi = si.wi
        si.wi = mi.Frame3f(auxiliary.y.n).to_local(-auxiliary.dir)
        try:
            bsdf = si.bsdf()
            if bsdf is None:
                return 0.0

            pw_result = bsdf.eval_pdf(mi.BSDFContext(), si, wo_local)
            try:
                pw = float(pw_result[1])  # eval_pdf returns (value, pdf) in some versions
            except (TypeError, IndexError):
                pw = float(pw_result)
        finally:
            si.wi = saved_wi

        # cos at segment.y (incoming side) — the area measure conversion
        cos_at_seg_y = float(dr.abs(dr.dot(segment.y.n, wo_world)))

        # NOTE: no grazing-cosine guard here. pw goes to zero smoothly at
        # grazing (cos/π for diffuse), so deleting mass would break the w*G
        # cancellation (numerator keeps f*G > 0 for the same configuration).

        # eval_pdf returns solid-angle PDF directly (e.g. cos/π for diffuse) — no conversion needed
        return pw * cos_at_seg_y / len_sq


    """
    KLEEP IN MIND THE LOGIC IS DIFF FROM THE PDF
    EARLIER IN MMIS WE WERE FINDING WHICH MIGHT LEAD TO OUR ENDPOINT IN THE CLUISTER

    NOW WE ARE JUST LOOKING AT THE ENDPOINT AND FINDING WHAT IT BSDFS WITH
    """
    @staticmethod
    def shift_invariant_bsdf(seg: Segment, next_seg: Segment) -> mi.Color3f:
        """
        f_r(s, s') via shift-invariant approximation, Eq (6).
        Evaluate BSDF at y of s:
          wi = -seg.dir  (incoming at y, from x)
          wo = direction from y toward y' of next_seg

        mitsuba's bsdf.eval returns f_r * cos(theta_o), but the propagation formula
        needs the pure f_r — both cosines are already carried by G(s').
        Divide by cos(theta_o) to recover f_r.
        """
        wi_world = -seg.dir
        wo_world = dr.normalize(next_seg.y.p - seg.y.p)

        frame_y = mi.Frame3f(seg.y.n)
        si = seg.y.si
        saved_wi = si.wi
        si.wi = frame_y.to_local(wi_world)
        wo_local = frame_y.to_local(wo_world)

        try:
            bsdf = si.bsdf()
            if bsdf is None:
                return mi.Color3f(0.0)

            # bsdf.eval returns f_r * cos(theta_o); divide out cos(theta_o) to get pure f_r
            cos_o = abs(float(mi.Frame3f.cos_theta(wo_local)))
            if cos_o < 1e-6:
                return mi.Color3f(0.0)
            return bsdf.eval(mi.BSDFContext(), si, wo_local) / cos_o
        finally:
            si.wi = saved_wi


class LightSampler(Sampler):
    technique_type = SegmentTechnique.LIGHT
    # TODO combine with sequential sampler, don't need to repeat so much code lol
    def conditional_pdf(self, segment: Segment, auxiliary: Segment):
        """
        wenyou says this is a mirror of the camera sampler
        but its from the persepctive of the light sampelr, so we still start from
        aux x->y and assume that seg follows from x->y (ie that seg is a light seg)
        pdf oviously treats segment y 
        so it has to be auxiliary y light toward 

        since the bsdf occurs at auxiliary x, we need to guard first on it's x
        """
        if auxiliary.x.is_camera or auxiliary.x.is_light:
            return 0.0
            # couldn't have sampled it
        delta = segment.x.p - auxiliary.x.p
        len_sq = float(dr.squared_norm(delta))
        # GUARD SYMMETRY: same constant as Segment admission (see SequentialSampler)
        if len_sq < MIN_SEG_LEN * MIN_SEG_LEN:
            return 0.0

        wo_world = dr.normalize(delta)
        wo_local = mi.Frame3f(auxiliary.x.n).to_local(wo_world)

        # incoming direction at auxiliary.x is along the segment direction
        # light path travels x→y, so incoming at x is -seg.dir

        si = auxiliary.x.si
        saved_wi = si.wi
        si.wi = mi.Frame3f(auxiliary.x.n).to_local(auxiliary.dir)
        try:
            bsdf = si.bsdf()
            if bsdf is None:
                return 0.0

            ctx = mi.BSDFContext(mode=mi.TransportMode.Importance)

            pw_result = bsdf.eval_pdf(ctx, si, wo_local)
            try:
                pw = float(pw_result[1])
            except (TypeError, IndexError):
                pw = float(pw_result)
        finally:
            si.wi = saved_wi

        cos_at_seg_x = float(dr.abs(dr.dot(segment.x.n, wo_world)))

        # no grazing-cosine guard — pw handles grazing smoothly (see SequentialSampler)
        return pw * cos_at_seg_x / len_sq

    @staticmethod
    def shift_invariant_bsdf(seg: Segment, next_seg: Segment) -> mi.Color3f:
        # same as camera version for diffuse — adjoint distinction matters for glossy
        if seg.y.is_light or seg.y.is_camera:
            return mi.Color3f(0.0)
        
        wi_world = -seg.dir
        wo_world = dr.normalize(next_seg.y.p - seg.y.p)

        frame_y = mi.Frame3f(seg.y.n)
        si = seg.y.si
        saved_wi = si.wi
        si.wi = frame_y.to_local(wi_world)
        wo_local = frame_y.to_local(wo_world)

        try:
            bsdf = si.bsdf()
            if bsdf is None:
                return mi.Color3f(0.0)

            # bsdf.eval returns f_r * cos(theta_o); divide out cos(theta_o) to get pure f_r
            cos_o = abs(float(mi.Frame3f.cos_theta(wo_local)))
            if cos_o < 1e-6:
                return mi.Color3f(0.0)
            ctx = mi.BSDFContext(mode=mi.TransportMode.Importance)
            return bsdf.eval(ctx, si, wo_local) / cos_o
        finally:
            si.wi = saved_wi


class BridgeSampler(Sampler):
    """
    Bridge-segment technique (paper §S2, Eq. S8 — the to-light case).

    A bridge segment connects an existing camera-side segment to a freshly
    sampled vertex on a light source. That is exactly next-event estimation,
    expressed as its own path-construction technique rather than smuggled into
    the sequential camera sampler.

    Generation pdf (S8):  p(s | z_{i-1}) = p_L(x) · p_K(y | z_{i-1})
      • p_L(x)            area-measure pdf of the sampled light vertex
                          (emitter-pick × position pdf, reference-independent),
                          stored on the segment as `bridge_parea` at creation.
      • p_K(y | z_{i-1})  uniform-kernel perturbation placing the other endpoint
                          in the kernel of an existing segment's endpoint. As
                          everywhere, p_K = 1/|K| is constant across the cluster
                          and cancels the numerator K in MMIS, so it is omitted
                          here (the kernel-cancellation convention).

    The MMIS caller adds this term once per camera-side auxiliary in the kernel
    (the conditioning segments the bridge could have attached to), independent
    of the auxiliary's own technique — i.e. it counts as a genuinely distinct
    technique, giving the balance heuristic between BSDF-found light hits and
    NEE. Applies to any light-terminal segment (NEE-made OR BSDF-found light
    hit); both carry `bridge_parea`.

    TODO (S6/S7): to-camera bridges and general camera↔light connections plug in
    by relaxing the `segment.y.is_light` gate (adding p_W for the sensor case,
    and a second p_K factor for the general two-kernel connection).
    """
    technique_type = SegmentTechnique.BRIDGE

    def conditional_pdf(self, segment: Segment, auxiliary: Segment) -> float:
        # to-light bridge applies only to light-terminal segments
        if not segment.y.is_light:
            return 0.0
        # the conditioning segment must be a real camera-side surface endpoint:
        # the to-light bridge attaches the light vertex to a camera vertex in
        # the kernel. Light-side / synthetic endpoints are not valid conditioners.
        if (auxiliary.technique != SegmentTechnique.CAMERA
                or auxiliary.y.is_light or auxiliary.y.is_camera):
            return 0.0
        return float(getattr(segment, "bridge_parea", 0.0))

    @staticmethod
    def shift_invariant_bsdf(seg: Segment, next_seg: Segment) -> mi.Color3f:
        # Bridge segments are never the BSDF vertex of a merge (their y is on the
        # light); propagation always evaluates f_r with the CAMERA sampler at the
        # receiver. Provided only for registry symmetry.
        return SequentialSampler.shift_invariant_bsdf(seg, next_seg)

