"""Vectorized (wavefront) version of the MIS path tracer.

Mirrors the logic of the scalar ``pathtracer.BasicPathTracer`` but is written
with Dr.Jit masked control flow so it runs in ``llvm_ad_rgb`` / ``cuda_ad_rgb``
modes (10-50x faster than the scalar Python integrator).

The active variant must already be set by the caller *before* this module is
imported, because the ``mi.SamplingIntegrator`` base class is resolved at class
definition time.
"""

import mitsuba as mi
import drjit as dr


def mis_weight(pdf_a, pdf_b):
    """Power heuristic with beta=2 (vectorized, divide-by-zero safe)."""
    a2 = dr.square(pdf_a)
    b2 = dr.square(pdf_b)
    w  = a2 / (a2 + b2)
    return dr.select(dr.isfinite(w), w, 0.0)


class BasicPathTracerVec(mi.SamplingIntegrator):
    def __init__(self, props):
        super().__init__(props)
        self.max_depth = props.get('max_depth', 8)
        self.rr_depth  = props.get('rr_depth', 3)

    @dr.syntax
    def sample(self,
               scene:   mi.Scene,
               sampler: mi.Sampler,
               ray:     mi.RayDifferential3f,
               medium:  mi.Medium = None,
               active:  bool = True):

        bsdf_ctx = mi.BSDFContext()

        ray    = mi.Ray3f(ray)
        L      = mi.Spectrum(0.0)
        beta   = mi.Spectrum(1.0)
        eta    = mi.Float(1.0)
        depth  = mi.UInt32(0)
        active = mi.Bool(active)

        # Per-lane state carried from the previous vertex for the emitter-hit
        # MIS weight (BSDF-sampling side of the direct-light estimate).
        prev_si         = dr.zeros(mi.SurfaceInteraction3f)
        prev_bsdf_pdf   = mi.Float(1.0)
        prev_bsdf_delta = mi.Bool(True)

        while dr.hint(active, max_iterations=self.max_depth):

            si = scene.ray_intersect(ray, active=active)

            # --------------------------------------------------------------
            # Emitter hit: radiance arriving along the BSDF-sampled path
            # (or directly from the camera), MIS-weighted against the NEE
            # performed at the previous vertex. For an escaped ray, ds.emitter
            # is the environment emitter (eval returns 0 if there is none).
            # --------------------------------------------------------------
            ds       = mi.DirectionSample3f(scene, si=si, ref=prev_si)
            em_pdf   = scene.pdf_emitter_direction(prev_si, ds, ~prev_bsdf_delta)
            mis_bsdf = mis_weight(prev_bsdf_pdf, em_pdf)
            L += beta * mis_bsdf * ds.emitter.eval(si, active)

            active_next = active & si.is_valid() & (depth + 1 < self.max_depth)

            bsdf = si.bsdf(ray)

            # --------------------------------------------------------------
            # Next event estimation (only on smooth lobes).
            # --------------------------------------------------------------
            active_em = active_next & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)
            ds_em, em_weight = scene.sample_emitter_direction(
                si, sampler.next_2d(), True, active_em)
            active_em &= (ds_em.pdf != 0.0)

            wo_local = si.to_local(ds_em.d)
            bsdf_val, bsdf_pdf = bsdf.eval_pdf(bsdf_ctx, si, wo_local, active_em)
            mis_em = dr.select(ds_em.delta, 1.0, mis_weight(ds_em.pdf, bsdf_pdf))
            L[active_em] += beta * mis_em * bsdf_val * em_weight

            # --------------------------------------------------------------
            # BSDF sampling (next bounce).
            # --------------------------------------------------------------
            bsdf_sample, bsdf_weight = bsdf.sample(
                bsdf_ctx, si, sampler.next_1d(), sampler.next_2d(), active_next)

            ray  = si.spawn_ray(si.to_world(bsdf_sample.wo))
            beta = beta * bsdf_weight
            eta  = eta * bsdf_sample.eta

            # Carry state for the next iteration's emitter-hit MIS weight.
            prev_si         = si
            prev_bsdf_pdf   = bsdf_sample.pdf
            prev_bsdf_delta = mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Delta)

            # --------------------------------------------------------------
            # Russian roulette.
            # --------------------------------------------------------------
            depth[si.is_valid()] += 1
            beta_max     = dr.max(beta)
            active_next &= (beta_max != 0.0)

            rr_prob   = dr.minimum(beta_max * dr.square(eta), 0.95)
            rr_active = depth >= self.rr_depth
            beta[rr_active] *= dr.rcp(rr_prob)
            rr_continue = sampler.next_1d() < rr_prob

            active = active_next & (~rr_active | rr_continue)

        return L, (depth != 0), []
