"""Shared setup for debug scripts: import-path patching + scene builder."""
import os, sys
import mitsuba as mi

_SRC = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

mi.set_variant("scalar_rgb")


def cornell_scene(width: int, height: int):
    d = mi.cornell_box()
    d["sensor"]["film"]["width"] = width
    d["sensor"]["film"]["height"] = height
    d["integrator"] = {"type": "path", "max_depth": -1}
    return mi.load_dict(d)
