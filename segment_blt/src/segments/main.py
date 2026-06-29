"""Render a sample scene with the segment-BLT renderer.

    python src/segments/main.py --scene cbox
    python src/segments/main.py --scene lamp  --width 256
    python src/segments/main.py --scene dragon --techniques camera raytracing --rt-per-pixel 4

--scene takes a sample name under ../../samples (cbox, lamp, dragon, matpreview)
or a path to a scene .xml. --width sets the width; the height is derived from the
scene's authored aspect ratio (baseline_basic.py convention). Omit --width to
render at the authored resolution.
"""
import argparse
import sys
from pathlib import Path

# Run standalone: put .../segment_blt/src on the path so `import segments` works.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mitsuba as mi
from segments.renderer import Renderer, _resolve_scene   # sets scalar_rgb on import


def _authored_size(scene_path):
    """(w0, h0) from the scene's authored film — a cheap parse, not a render."""
    fs = mi.load_file(str(scene_path)).sensors()[0].film().crop_size()
    return int(fs[0]), int(fs[1])


def _default_out(scene):
    p = Path(scene)
    if p.name == "scene.xml":
        return f"{p.parent.name}.png"     # .../cbox/scene.xml -> cbox.png
    if p.suffix == ".xml":
        return f"{p.stem}.png"
    return f"{p.name}.png"                 # bare sample name -> name.png


def main():
    ap = argparse.ArgumentParser(description="Render a sample scene with segment BLT.")
    ap.add_argument("--scene", required=True,
                    help="sample name under samples/ (cbox, lamp, dragon, matpreview) "
                         "or a path to a scene .xml")
    ap.add_argument("--out", default=None, help="output image (default: <scene>.png)")
    ap.add_argument("--mode", default="vec", choices=("vec", "nonvec", "ref"))
    ap.add_argument("--width", type=int, default=128,
                    help="output width; height is derived from the scene's authored "
                         "aspect ratio. (This renderer is a per-pixel Python loop, so "
                         "the authored 1024² is impractical — pass --width 1024 only if "
                         "you mean it.)")
    ap.add_argument("--iters", type=int, default=8, help="render iterations / spp")
    ap.add_argument("--techniques", nargs="+", default=["camera"],
                    choices=("camera", "raytracing"),
                    help="transport sources, space-separated "
                         "(e.g. --techniques camera raytracing)")
    ap.add_argument("--rt-per-pixel", type=int, default=8,
                    help="independent Eq.13 segments per pixel when raytracing is on")
    ap.add_argument("--light", action="store_true", help="also trace light paths")
    ap.add_argument("--prop-iters", type=int, default=4,
                    help="propagation (merge) bounces. Sweep 1->4 to test multi-bounce "
                         "compounding: if raytracing brightens with more bounces, |M| is "
                         "compounding through indirect chains.")
    args = ap.parse_args()

    scene_path = _resolve_scene(args.scene)
    w0, h0 = _authored_size(scene_path)
    width = args.width
    height = max(1, round(width * h0 / w0))   # preserve authored aspect ratio
    print(f"[main] {args.scene}: authored {w0}x{h0}, aspect-matched -> {width}x{height}  mode={args.mode}")

    out = args.out or _default_out(args.scene)
    Renderer().save_render_by_scene_path(
        scene_path, out, width=width, height=height, n_iterations=args.iters,
        mode=args.mode, techniques=args.techniques,
        rt_segments_per_pixel=args.rt_per_pixel, add_light_samples=args.light,
        num_prop_iterations=args.prop_iters)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
