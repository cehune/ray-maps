import argparse
import os
import sys
import time

import mitsuba as mi

# Mitsuba binds the integrator base class (mi.SamplingIntegrator) to the active
# variant at *import* time, and the scalar and vec integrators need different
# variants — so the variant must be chosen before either one is imported.
# Decide it from argv up front; non-vec (scalar) is the standard path. The
# imports below are therefore guarded, but still top-level (never inside a fn).
_VEC = '--vec' in sys.argv
mi.set_variant('llvm_ad_rgb' if _VEC else 'scalar_rgb')

if _VEC:
    from qmc.pathtracer_vec import BasicPathTracerVec
else:
    from qmc.renderer import Renderer
    from qmc.setup import build_sampler


def main():
    p = argparse.ArgumentParser(description="QMC path-tracer test bed")
    p.add_argument('--width',  type=int, default=300)
    p.add_argument('--height', type=int, default=300)
    p.add_argument('--spp',    type=int, default=64)
    p.add_argument('--max-depth', type=int, default=8)
    p.add_argument('--rr-depth',  type=int, default=3)

    # scalar sampler controls (ignored under --vec)
    p.add_argument('--sampler',  default='sobol',
                   choices=['sobol', 'padded_sobol', 'independent'])
    p.add_argument('--max-dim',  type=int, default=64,
                   help="Sobol' dimensions to build/cap; dims beyond this are "
                        "padded with random (plain sobol only)")
    p.add_argument('--scramble', default='owen', choices=['none', 'xor', 'owen'])
    p.add_argument('--seed',     type=int, default=0)

    p.add_argument('--vec', action='store_true',
                   help="run the fast LLVM wavefront vec integrator with "
                        "Mitsuba's default sampler (QMC not wired into vec yet)")
    p.add_argument('--scene', default="/Users/celine/Documents/projects/ray-maps/samples/lamp/scene.xml")
    p.add_argument('--out-dir', default="/Users/celine/Documents/projects/ray-maps/quasi_monte_carlo/outputs")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    img, tag = render_vec(args) if args.vec else render_scalar(args)

    # one render, both formats
    bmp = mi.Bitmap(img)
    bmp.write(os.path.join(args.out_dir, f"render_{tag}.exr"))
    bmp.convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8,
                srgb_gamma=True).write(os.path.join(args.out_dir, f"render_{tag}.png"))
    print(f"Saved render_{tag}.{{exr,png}} to {args.out_dir}")


def render_scalar(args):
    """Scalar driver: our Renderer + any QMC/independent Sampler."""
    scene   = mi.load_file(args.scene, resx=args.width, resy=args.height, spp=args.spp)
    sampler = build_sampler(args.sampler, spp=args.spp, max_dim=args.max_dim,
                            scramble_method=args.scramble, seed=args.seed)
    r = Renderer(scene, max_depth=args.max_depth, rr_depth=args.rr_depth)

    t = time.time()
    img = r.render(sampler, args.spp)
    print(f"scalar ({args.sampler}) render: {time.time() - t:.1f}s")

    dimtag = f"_d{args.max_dim}" if args.sampler == 'sobol' else ""
    tag = f"{args.width}x{args.height}_{args.spp}spp_{args.sampler}{dimtag}_{args.scramble}"
    return img, tag


def render_vec(args):
    """Fast LLVM wavefront via mi.render with Mitsuba's default sampler."""
    itype = 'basic_pt_vec'
    mi.register_integrator(itype, lambda props: BasicPathTracerVec(props))
    scene = mi.load_file(args.scene, resx=args.width, resy=args.height, spp=args.spp)
    integrator = mi.load_dict({'type': itype,
                               'max_depth': args.max_depth, 'rr_depth': args.rr_depth})

    t = time.time()
    img = mi.render(scene, spp=args.spp, integrator=integrator)
    print(f"vec render: {time.time() - t:.1f}s")

    tag = f"{args.width}x{args.height}_{args.spp}spp_vec"
    return img, tag


if __name__ == '__main__':
    main()
