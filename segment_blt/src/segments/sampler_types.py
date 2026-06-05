import mitsuba as mi
import drjit as dr
from segments.primitives import Segment, SegmentTechnique
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

        # Distance from auxiliary.y to segment.y — this is the connection vector
        # used in the area-measure conversion.  For adjacent segments in the
        # sequential model auxiliary.y == segment.x so this equals segment.len,
        # but using the actual distance is correct for arbitrary auxiliaries in
        # the same cluster.
        delta  = segment.y.p - auxiliary.y.p
        len_sq = float(dr.squared_norm(delta))
        if len_sq < 1e-10:
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

        cos_at_aux_y = float(dr.abs(dr.dot(auxiliary.y.n, wo_world)))

        # cos at segment.y (incoming side) — the area measure conversion
        cos_at_seg_y = float(dr.abs(dr.dot(segment.y.n, wo_world)))

        if cos_at_aux_y < 1e-6:
            return 0.0

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
        if len_sq < 1e-10:
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

        cos_at_aux_x = float(dr.abs(dr.dot(auxiliary.x.n, wo_world)))
        cos_at_seg_x = float(dr.abs(dr.dot(segment.x.n, wo_world)))

        if cos_at_aux_x < 1e-6:
            return 0.0

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

