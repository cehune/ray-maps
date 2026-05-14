import mitsuba as mi
import drjit as dr
from primitives import Segment, SegmentTechnique

class SequentialSampler:
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
        
        segment_diff = segment.y.p - segment.x.p

        # cant store outgoing direction local to the segment because we assume it 
        # happens at the auxiliary endpoint, using auxiliary.y.n, check with Wenyou
        wo_world = segment_diff * (1.0 / segment.len)
        wo_local = mi.Frame3f(auxiliary.y.n).to_local(wo_world)

        # pw (s'|yi, si) bsdf sampling pdf to next direction (just along si+1)
        # this is along x to y
        pw = auxiliary.y.si.bsdf().eval_pdf(mi.BSDFContext(), auxiliary.y.si, wo_local)

        cosine = dr.abs(dr.dot(segment.y.n, wo_world))

        return pw * cosine / (segment.len ** 2)
