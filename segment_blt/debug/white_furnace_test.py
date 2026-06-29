"""White-furnace normalization check for the ray-tracing segment technique.

A perfect Lambertian sphere (albedo 1) inside a uniform emitter (L_e = 1). In a
uniform field a perfect reflector returns exactly L_e, so the whole image must
converge to 1.0 — analytically, independent of geometry. That makes the target a
KNOWN constant, so any deviation in the raytracing render is the RT merge's
normalization error, read off directly (no eyeballing cbox).

Because the missing factor is suspected to scale with the scene area |M|, the
test prints |M| too: run it here and on cbox and compare bias-vs-|M|.

    python debug/white_furnace_test.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import mitsuba as mi
from segments.renderer import Renderer, _resolve_scene   # sets scalar_rgb on import

W = H = 64          # small: the scalar/per-pixel renderers are Python loops
ITERS = 16


def central_mean(img, frac=0.4):
    """Mean over the central frac×frac box — where the test sphere sits."""
    a = np.asarray(img)
    h, w = a.shape[:2]
    y0, y1 = int(h * (0.5 - frac / 2)), int(h * (0.5 + frac / 2))
    x0, x1 = int(w * (0.5 - frac / 2)), int(w * (0.5 + frac / 2))
    return float(a[y0:y1, x0:x1].mean())


def main():
    scene_path = _resolve_scene("white_furnace")
    r = Renderer()

    # sanity at the base size — ref and camera MUST both read ~1.0, else the
    # test means nothing.
    base = mi.load_file(scene_path, resx=W, resy=H, outer_r=4.0)
    ref = central_mean(r.render(base, W, H, spp=ITERS))
    cam = central_mean(r.render_vec(base, height=H, width=W, n_iterations=ITERS,
                                    verbose=False, add_light_samples=False,
                                    segment_source="camera"))
    print(f"\nsanity (outer_r=4): ref(PT)={ref:.4f}  camera={cam:.4f}  "
          f"-> both must be ~1.0\n")

    # |M| sweep: grow the enclosure. The inner-sphere answer is 1.0 regardless of
    # how big the enclosure is, so any trend in rt/ref is purely the |M|
    # dependence of the RT normalization.
    print(f"{'outer_r':>8s}{'|M|':>9s}{'raytrace':>10s}{'rt/ref':>9s}")
    for outer_r in (4.0, 8.0, 16.0):
        scene = mi.load_file(scene_path, resx=W, resy=H, outer_r=outer_r)
        M = float(sum(float(s.surface_area()) for s in scene.shapes()))
        rt = central_mean(r.render_vec(scene, height=H, width=W, n_iterations=ITERS,
                                       verbose=False, add_light_samples=False,
                                       segment_source="raytracing"))
        print(f"{outer_r:8.0f}{M:9.0f}{rt:10.4f}{rt / max(ref, 1e-9):9.4f}")

    print("\nrt/ref ~halves when |M| ~doubles -> bias proportional to 1/|M| (the RT")
    print("                                    contribution is missing a *|M|).")
    print("rt/ref flat across |M|           -> |M|-independent (kernel size / n_RT).")


if __name__ == "__main__":
    main()
