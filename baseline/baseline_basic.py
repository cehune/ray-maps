import argparse
import re
from pathlib import Path
import numpy as np
import mitsuba as mi
import drjit as dr
import time

mi.set_variant('llvm_ad_rgb')


def load_scene_box_rfilter(scene_path, **kwargs):
    """Load a Mitsuba scene XML forcing the film's reconstruction filter to box.

    The authored tent/gaussian filters splat each sample onto neighbouring
    pixels, blooming bright emitters ~1px outward. A per-pixel renderer (the
    segment BLT accumulates one sample into its own pixel) has no such bloom, so
    against a tent/gaussian reference every light shows a dark ring in the diff —
    a fixed per-pixel error that never averages out. A box rfilter makes the
    reference apples-to-apples. We inject via a sibling temp file so the scene's
    relative asset paths still resolve.
    """
    scene_path = Path(scene_path)
    xml = scene_path.read_text()
    if re.search(r"<rfilter\b[^>]*?/>", xml):
        xml = re.sub(r"<rfilter\b[^>]*?/>", '<rfilter type="box"/>', xml, count=1)
    elif re.search(r"<rfilter\b.*?</rfilter>", xml, re.S):
        xml = re.sub(r"<rfilter\b.*?</rfilter>", '<rfilter type="box"/>', xml, count=1, flags=re.S)
    else:  # no rfilter authored — inject one inside the first <film ...>
        xml = re.sub(r"(<film\b[^>]*>)", r'\1\n\t\t\t<rfilter type="box"/>', xml, count=1)
    tmp = scene_path.with_name(f"._box_{scene_path.name}")
    tmp.write_text(xml)
    try:
        return mi.load_file(str(tmp), **kwargs)
    finally:
        tmp.unlink()


# --- Integrator ---------------------------------------------------------------

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
        # use_nee is a plain Python bool, so the `if self.use_nee` branches below
        # are resolved at trace time -- they are NOT data-dependent control flow.
        self.use_nee   = props.get('use_nee', True)

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
            # (or directly from the camera).
            #
            # With NEE on, this contribution is MIS-weighted against the NEE
            # done at the previous vertex. With NEE OFF there is no competing
            # strategy, so the weight must be 1.0 -- otherwise the energy NEE
            # would have carried is silently dropped and the image is biased.
            # --------------------------------------------------------------
            ds = mi.DirectionSample3f(scene, si=si, ref=prev_si)
            if self.use_nee:
                em_pdf   = scene.pdf_emitter_direction(prev_si, ds, ~prev_bsdf_delta)
                mis_bsdf = mis_weight(prev_bsdf_pdf, em_pdf)
            else:
                mis_bsdf = mi.Float(1.0)
            L += beta * mis_bsdf * ds.emitter.eval(si, active)

            active_next = active & si.is_valid() & (depth + 1 < self.max_depth)

            bsdf = si.bsdf(ray)

            # --------------------------------------------------------------
            # Next event estimation (only on smooth lobes, only if enabled).
            # --------------------------------------------------------------
            if self.use_nee:
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
            beta_max     = dr.max(beta)           # per-lane max over RGB channels
            active_next &= (beta_max != 0.0)

            rr_prob   = dr.minimum(beta_max * dr.square(eta), 0.95)
            rr_active = depth >= self.rr_depth
            beta[rr_active] *= dr.rcp(rr_prob)
            rr_continue = sampler.next_1d() < rr_prob

            active = active_next & (~rr_active | rr_continue)

        return L, (depth != 0), []


mi.register_integrator("basic_pt_vec", lambda props: BasicPathTracerVec(props))


# --- I/O helpers --------------------------------------------------------------

def save_render(img, stem: Path):
    """Write both a linear EXR and a tonemapped (sRGB, 8-bit) PNG.

    Going through an explicit mi.Bitmap is deliberate: mi.util.write_bitmap on a
    render tensor can produce 0-byte EXRs in this build, whereas Bitmap.write is
    reliable and lets us reuse the same buffer for the PNG.
    """
    bmp = mi.Bitmap(img)
    bmp.write(str(stem.with_suffix(".exr")))
    ldr = bmp.convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8, srgb_gamma=True)
    ldr.write(str(stem.with_suffix(".png")))


def render_chunked(scene, integrator, total_spp, chunk, base_seed=0):
    """Render total_spp by accumulating bounded passes.

    Mitsuba's LLVM/CUDA backend vectorizes a render as a single wavefront of
    size width * height * spp. A high-spp reference (e.g. 16384 spp at 300x300
    is ~1.5e9 lanes) therefore tries to allocate tens of GB at once and Dr.Jit
    throws, surfacing as an opaque drjit.custom(_RenderOp) error. Splitting into
    `chunk`-spp passes keeps each wavefront small; the samples-per-pass weighted
    mean is identical in expectation to a single big render.
    """
    acc, done, p = None, 0, 0
    while done < total_spp:
        s = min(chunk, total_spp - done)
        contrib = mi.TensorXf(mi.render(scene, spp=s, integrator=integrator,
                                        seed=base_seed + p)) * s
        acc = contrib if acc is None else acc + contrib
        done += s
        p += 1
    return acc / done


def authored_max_depth(scene):
    """Read max_depth from the scene's authored integrator.

    Returns -1 if the scene has no integrator or an unbounded one. The
    integrator doesn't expose max_depth via the Python API (mi.traverse is
    empty for it), but its repr is e.g. 'PathIntegrator[ max_depth = 6, ... ]'.
    """
    ig = scene.integrator()
    if ig is None:
        return -1
    m = re.search(r"max_depth\s*=\s*(-?\d+)", str(ig))
    return int(m.group(1)) if m else -1


# --- Driver -------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("sample", help="Sample directory name")
    p.add_argument("--width", type=int, default=None,
                   help="Output width; height is derived from the scene aspect ratio.")
    p.add_argument("--nee", action="store_true",
                   help="Enable next event estimation (default: off, BSDF sampling only).")
    p.add_argument("--max-depth", type=int, default=None,
                   help="Path-length cap. If omitted, taken from the scene's "
                        "authored integrator (unbounded is clamped to 64). The "
                        "reference and test integrator always share this value.")
    p.add_argument("--rr-depth", type=int, default=3)
    p.add_argument("--ref-spp", type=int, default=16384)
    p.add_argument("--ref-chunk", type=int, default=256,
                   help="Samples per reference pass; reference is accumulated in "
                        "chunks so the wavefront (pixels*spp) does not blow memory.")
    return p.parse_args()


def main():
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    scene_path = (script_dir / ".." / "samples" / args.sample / "scene.xml").resolve()

    # 1. Read the authored resolution (cheap parse, not a render).
    base_scene = load_scene_box_rfilter(scene_path)
    fs = base_scene.sensors()[0].film().crop_size()
    w0, h0 = int(fs[0]), int(fs[1])
    aspect = w0 / h0

    # 2. Resolve render resolution. Only reload when --width is given; passing
    #    resx/resy to a scene that lacks $resx/$resy raises "unused parameters"
    #    rather than silently rendering at the authored size, so the default
    #    path must NOT reload. When we do reload, assert the film actually bound
    #    to the requested size -- otherwise the RMSE-vs-reference is meaningless.
    if args.width is not None:
        width  = args.width
        height = max(1, round(width / aspect))
        scene  = load_scene_box_rfilter(scene_path, resx=width, resy=height)
        aw, ah = (int(v) for v in scene.sensors()[0].film().crop_size())
        if (aw, ah) != (width, height):
            raise RuntimeError(
                f"Requested {width}x{height} but film built at {aw}x{ah}. "
                f"Add $resx/$resy to the <film> in {scene_path.name}.")
    else:
        width, height = w0, h0
        scene = base_scene

    # 3. Resolve ONE finite max_depth shared by the reference and the test
    #    integrator. In a closed scene the path-length cap controls how much
    #    indirect light accumulates, so a mismatch shows up as a global exposure
    #    gap (the brightness difference you saw). An unbounded authored depth
    #    (-1) is clamped finite so the vec loop terminates and the reference
    #    carries no bias floor against the (finite) test integrator.
    auth_md = authored_max_depth(scene)
    if args.max_depth is not None:
        max_depth = args.max_depth
    elif auth_md >= 1:
        max_depth = auth_md
    else:
        max_depth = 64

    scale = width / w0
    print(f"Sample        : {args.sample}")
    print(f"Authored res  : {w0}x{h0}")
    print(f"Rendering at  : {width}x{height}  ({scale:.4g}x authored)")
    print(f"max_depth     : {max_depth}"
          + ("" if args.max_depth is not None else f"  (authored: {auth_md})"))

    # 4. Test integrator with the NEE toggle wired in.
    custom_integrator = mi.load_dict({
        'type':      'basic_pt_vec',
        'max_depth': max_depth,
        'rr_depth':  args.rr_depth,
        'use_nee':   args.nee,
    })

    # 5. Output directory.
    nee_tag = "nee" if args.nee else "no-nee"
    out_dir = script_dir / f"spp_renders_basic-{args.sample}-{width}x{height}-vec-{nee_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 6. Reference: built-in path tracer at the SAME max_depth and on the SAME
    #    scene object (identical framing), chunked so the wavefront (pixels*spp)
    #    stays bounded at high spp.
    ref_exr = out_dir / "reference.exr"
    if ref_exr.exists():
        reference = np.array(mi.Bitmap(str(ref_exr)))
    else:
        print(f"Rendering reference ({args.ref_spp} spp in {args.ref_chunk}-spp passes)...")
        ref_integrator = mi.load_dict({'type': 'path', 'max_depth': max_depth})
        ref_img   = render_chunked(scene, ref_integrator, args.ref_spp, args.ref_chunk)
        save_render(ref_img, out_dir / "reference")
        reference = np.asarray(ref_img)

    # 6. SPP sweep with the vectorized PT; save EXR + PNG for each.
    spps = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    errs = []
    print(f"output dir {out_dir}")
    print(f"Starting sweep: {width}x{height} | NEE: {nee_tag} | "
          f"max_depth={max_depth} rr_depth={args.rr_depth}")
    for i, spp in enumerate(spps):
        t0  = time.time()
        img = mi.render(scene, spp=spp, integrator=custom_integrator, seed=1000 + i)

        err = float(np.sqrt(np.mean((np.asarray(img) - reference) ** 2)))
        errs.append(err)

        save_render(img, out_dir / f"{spp}")
        print(f"  spp={spp:>4} | RMSE={err:.6f} | Time={time.time()-t0:.2f}s")

    slope, _ = np.polyfit(np.log(spps), np.log(errs), 1)
    print(f"Final Slope: {slope:.3f}")


if __name__ == "__main__":
    main()