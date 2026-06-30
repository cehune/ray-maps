import mitsuba as mi
import drjit as dr
from qmc.sampler.constants import SCRAMBLERS, SAMPLERS
from qmc.sampler.sobol_sampler import SobolSampler
from qmc.sampler.padded_sobol_sampler import PaddedSobolSampler
from qmc.sampler.sampler import Sampler
mi.set_variant('scalar_rgb')


def mis_weight(pdf_a, pdf_b):
    """Power heuristic with beta=2."""
    a2 = pdf_a * pdf_a
    b2 = pdf_b * pdf_b
    denom = a2 + b2
    return a2 / denom if denom > 0.0 else 0.0


class BasicPathTracer(mi.SamplingIntegrator):
    def __init__(self, props):
        super().__init__(props)
        self.max_depth = props.get('max_depth', 8)
        self.rr_depth  = props.get('rr_depth', 3)
    def sample(self,
               scene:   mi.Scene,
               sampler: mi.Sampler,
               ray:     mi.RayDifferential3f):

        bsdf_ctx = mi.BSDFContext()

        ray  = mi.Ray3f(ray)
        L    = mi.Color3f(0.0)
        beta = mi.Color3f(1.0)

        # State carried from the previous vertex so we can MIS-weight any
        # emitter we hit along the BSDF-sampled path against the NEE that
        # was performed at that previous vertex.
        prev_si         = dr.zeros(mi.SurfaceInteraction3f)
        prev_bsdf_pdf   = 1.0
        prev_bsdf_delta = True   # camera ray: take the emitter at full weight

        depth = 0
        while depth < self.max_depth:
            # Each decision is addressed by (depth, slot) below, so the dimension
            # is fixed regardless of branching — nothing to set up per bounce.
            si = scene.ray_intersect(ray)

            # hits emitter directly
            emitter = si.emitter(scene)
            if emitter is not None:
                if prev_bsdf_delta:
                    mis = 1.0
                else:
                    ds = mi.DirectionSample3f(scene, si, prev_si)
                    em_pdf = scene.pdf_emitter_direction(prev_si, ds)
                    mis = mis_weight(prev_bsdf_pdf, em_pdf)
                L += beta * mis * emitter.eval(si)

            if not si.is_valid():
                break

            bsdf = si.bsdf(ray)

            # NEE, skipped for specular BSDFs. The "nee" slot stays reserved
            # whether or not we draw it, so the BSDF sample below is unaffected.
            nee = (sampler.bounce_2d(depth, "nee"))
            if mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth):
                ds, em_weight = scene.sample_emitter_direction(
                    si, mi.Point2f(*nee), True)

                if ds.pdf > 0.0:
                    wo_local = si.to_local(ds.d)
                    bsdf_val = bsdf.eval(bsdf_ctx, si, wo_local)
                    bsdf_pdf = bsdf.pdf(bsdf_ctx, si, wo_local)
                    mis = 1.0 if ds.delta else mis_weight(float(ds.pdf), float(bsdf_pdf))
                    L += beta * mis * bsdf_val * em_weight

            # next bounce
            bsdf_sample, bsdf_weight = bsdf.sample(
                bsdf_ctx, si,
                sampler.bounce_1d(depth, "lobe"),
                mi.Point2f(*sampler.bounce_2d(depth, "dir"))
            )

            beta *= bsdf_weight
            if dr.all(beta == 0.0):
                break

            ray = si.spawn_ray(si.to_world(bsdf_sample.wo))

            # Carry state for the next iteration's emitter-hit MIS weight.
            prev_si         = mi.SurfaceInteraction3f(si)
            prev_bsdf_pdf   = float(bsdf_sample.pdf)
            prev_bsdf_delta = mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Delta)

            # RR. depth is incremented first, so the just-finished bounce is
            # (depth-1) — address RR with it to keep it in that bounce's block.
            depth += 1
            if depth >= self.rr_depth:
                rr_prob = min(float(dr.max(beta)), 0.95)
                if float(sampler.bounce_1d(depth - 1, "rr")) > rr_prob:
                    break
                beta /= rr_prob

        return L, True, []
