"""Render a sample scene with the segment-BLT renderer.

    python main.py --scene cbox
    python main.py --scene lamp  --width 256 --iters 16
    python main.py --scene dragon --techniques camera raytracing --rt-per-pixel 4

--scene takes a sample name under ../samples (cbox, lamp, dragon, matpreview) or
a direct path to a scene .xml.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from segments.renderer import Renderer, Technique   # noqa: E402


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
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--height", type=int, default=128)
    ap.add_argument("--iters", type=int, default=8, help="render iterations / spp")
    ap.add_argument("--techniques", nargs="+", default=["camera"],
                    choices=("camera", "raytracing"),
                    help="transport sources, space-separated "
                         "(e.g. --techniques camera raytracing)")
    ap.add_argument("--rt-per-pixel", type=int, default=1,
                    help="independent Eq.13 segments per pixel when raytracing is on")
    ap.add_argument("--light", action="store_true", help="also trace light paths")
    args = ap.parse_args()

    out = args.out or _default_out(args.scene)
    Renderer().save_render_by_scene_path(
        args.scene, out,
        width=args.width, height=args.height, n_iterations=args.iters,
        mode=args.mode, techniques=args.techniques,
        rt_segments_per_pixel=args.rt_per_pixel, add_light_samples=args.light,
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
