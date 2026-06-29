"""QMC path-tracer test bed.

    python src/qmc/main.py --scene lamp
    python src/qmc/main.py --scene cbox --width 256 --sampler sobol --spp 64
    python src/qmc/main.py --scene lamp --vec          # fast LLVM reference

--scene takes a sample name under ../../samples (cbox, lamp, dragon, matpreview)
or a path to a scene .xml. --width sets the width; the height is derived from the
scene's authored aspect ratio (baseline_basic.py convention). Omit --width to
render at the authored resolution.
"""
import argparse
import os
import sys
import time
from pathlib import Path

# Run standalone: put .../quasi_monte_carlo/src on the path so `import qmc` works
# without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mitsuba as mi

SAMPLES_DIR = Path(__file__).resolve().parents[3] / "samples"   # .../ray-maps/samples

# Mitsuba binds the integrator base class to the active variant at *import* time,
# and the scalar and vec integrators need different variants — so the variant
# must be chosen before either is imported. Decide it from argv up front.
_VEC = '--vec' in sys.argv
mi.set_variant('llvm_ad_rgb' if _VEC else 'scalar_rgb')

if _VEC:
    from qmc.pathtracer_vec import BasicPathTracerVec
else:
    from qmc.renderer import Renderer
    from qmc.setup import build_sampler


def _resolve_scene(scene):
    """Sample NAME ('cbox', 'lamp', 'dragon', 'matpreview') ->
    samples/<name>/scene.xml, or a direct path to a scene .xml (or a scene dir)."""
    p = Path(scene)
    if p.suffix == ".xml" and p.exists():
        return str(p)
    if p.is_dir() and (p / "scene.xml").exists():
        return str(p / "scene.xml")
    cand = SAMPLES_DIR / str(scene) / "scene.xml"
    if cand.exists():
        return str(cand)
    if p.exists():
        return str(p)
    avail = (sorted(d.name for d in SAMPLES_DIR.iterdir() if d.is_dir())
             if SAMPLES_DIR.is_dir() else [])
    raise FileNotFoundError(f"scene {scene!r} not found (looked for {cand}). Available: {avail}")


def _resolution(scene_path, width):
    """Derive (width, height) preserving the scene's authored aspect ratio."""
    fs = mi.load_file(scene_path).sensors()[0].film().crop_size()
    w0, h0 = int(fs[0]), int(fs[1])
    return width, max(1, round(width * h0 / w0))


def main():
    p = argparse.ArgumentParser(description="QMC path-tracer test bed")
    p.add_argument('--scene', default="lamp",
                   help="sample name under samples/ (cbox, lamp, dragon, matpreview) or a path")
    p.add_argument('--width', type=int, default=300,
                   help="output width; height is derived from the scene's authored "
                        "aspect ratio. (Scalar QMC is a per-pixel Python loop, so the "
                        "authored 1024² is impractical — pass --width 1024 only if you mean it.)")
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
    p.add_argument('--out-dir',
                   default=str(Path(__file__).resolve().parents[2] / "outputs"))
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    scene_path = _resolve_scene(args.scene)
    width, height = _resolution(scene_path, args.width)
    print(f"[qmc] {args.scene}: render {width}x{height}  spp={args.spp}  "
          f"sampler={'vec' if args.vec else args.sampler}")

    img, tag = (render_vec(args, scene_path, width, height) if args.vec
                else render_scalar(args, scene_path, width, height))

    # one render, both formats
    bmp = mi.Bitmap(img)
    bmp.write(os.path.join(args.out_dir, f"render_{tag}.exr"))
    bmp.convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8,
                srgb_gamma=True).write(os.path.join(args.out_dir, f"render_{tag}.png"))
    print(f"Saved render_{tag}.{{exr,png}} to {args.out_dir}")


def render_scalar(args, scene_path, width, height):
    """Scalar driver: our Renderer + any QMC/independent Sampler."""
    scene   = mi.load_file(scene_path, resx=width, resy=height, spp=args.spp)
    sampler = build_sampler(args.sampler, spp=args.spp, max_dim=args.max_dim,
                            scramble_method=args.scramble, seed=args.seed)
    r = Renderer(scene, max_depth=args.max_depth, rr_depth=args.rr_depth)

    t = time.time()
    img = r.render(sampler, args.spp)
    print(f"scalar ({args.sampler}) render: {time.time() - t:.1f}s")

    dimtag = f"_d{args.max_dim}" if args.sampler == 'sobol' else ""
    tag = f"{width}x{height}_{args.spp}spp_{args.sampler}{dimtag}_{args.scramble}"
    return img, tag


def render_vec(args, scene_path, width, height):
    """Fast LLVM wavefront via mi.render with Mitsuba's default sampler."""
    itype = 'basic_pt_vec'
    mi.register_integrator(itype, lambda props: BasicPathTracerVec(props))
    scene = mi.load_file(scene_path, resx=width, resy=height, spp=args.spp)
    integrator = mi.load_dict({'type': itype,
                               'max_depth': args.max_depth, 'rr_depth': args.rr_depth})

    t = time.time()
    img = mi.render(scene, spp=args.spp, integrator=integrator)
    print(f"vec render: {time.time() - t:.1f}s")

    tag = f"{width}x{height}_{args.spp}spp_vec"
    return img, tag


if __name__ == '__main__':
    main()
