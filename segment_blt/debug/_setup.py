"""Shared setup for debug scripts: import-path patching + scene builder."""
import os, sys
import mitsuba as mi

_SRC = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

mi.set_variant("scalar_rgb")


def cornell_scene(width: int, height: int):
    # Single source of truth: the SAME scene baseline.py renders its references
    # from (samples/cbox/scene.xml), so BLT renders and references are the same
    # scene. The BLT does its own per-pixel box accumulation, so the scene's
    # rfilter is irrelevant here — only geometry/camera/light must match, and
    # now they do. (Previously this was mi.cornell_box(), a DIFFERENT scene:
    # different camera, geometry, and light spectrum — which made every
    # mean_ratio vs a baseline reference meaningless.)
    # debug/ is two levels under the repo root; samples/ lives at the root.
    scene_path = os.path.join(os.path.dirname(__file__), "..", "..", "samples", "cbox", "scene.xml")
    return mi.load_file(scene_path, resx=width, resy=height)
