import mitsuba as mi
import os
import time
import argparse
import time

def main():
    p = argparse.ArgumentParser(description="Basic MIS path tracer (QMC test bed)")
    p.add_argument('--width',  type=int, default=800, help="image width in pixels")
    p.add_argument('--height', type=int, default=800, help="image height in pixels")
    p.add_argument('--spp',    type=int, default=64,   help="samples per pixel")
    p.add_argument('--max-depth', type=int, default=8)
    p.add_argument('--rr-depth',  type=int, default=3)
    p.add_argument('--scalar', action='store_true',
                   help="use the slow scalar reference integrator instead of LLVM wavefront")
    p.add_argument('--scene', default="/Users/celine/Documents/projects/ray-maps/samples/lamp/scene.xml")
    p.add_argument('--out-dir', default="/Users/celine/Documents/projects/ray-maps/quasi_monte_carlo/outputs")
    args = p.parse_args()

    # The variant must be set before the integrator module is imported, since
    # mi.SamplingIntegrator is resolved at class-definition time.
    if args.scalar:
        mi.set_variant('scalar_rgb')
        from qmc.pathtracer import BasicPathTracer as Integrator
        itype = 'basic_pt'
    else:
        mi.set_variant('llvm_ad_rgb')
        from qmc.pathtracer_vec import BasicPathTracerVec as Integrator
        itype = 'basic_pt_vec'

    os.makedirs(args.out_dir, exist_ok=True)
    mi.register_integrator(itype, lambda props: Integrator(props))

    # resx/resy/spp are wired as overridable <default>s in the scene .xml,
    # so we can set the film resolution at load time.
    scene = mi.load_file(args.scene, resx=args.width, resy=args.height, spp=args.spp)
    integrator = mi.load_dict({
        'type': itype,
        'max_depth': args.max_depth,
        'rr_depth': args.rr_depth
    })

    t = time.time()
    image = mi.render(scene, spp=args.spp, integrator=integrator)
    dt = time.time() - t

    # image is a TensorXf — convert to bitmap and save
    mode = 'scalar' if args.scalar else 'vec'
    tag  = f"{args.width}x{args.height}_{args.spp}spp_{mode}"
    bmp  = mi.Bitmap(image)
    bmp.write(os.path.join(args.out_dir, f"render_{tag}.exr"))
    bmp_png = bmp.convert(
        pixel_format=mi.Bitmap.PixelFormat.RGB,
        component_format=mi.Struct.Type.UInt8,
        srgb_gamma=True   # applies gamma + clamps to [0,1]
    )
    bmp_png.write(os.path.join(args.out_dir, f"render_{tag}.png"))
    print(f"Saved render_{tag}.{{exr,png}} to {args.out_dir}  ({dt:.1f}s)")

if __name__ == '__main__':
    main()
