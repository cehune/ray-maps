"""Shared setup for QMC debug scripts: import-path patching + variant.

Anchored on THIS file, never on the cwd, so scripts run from anywhere. Puts
``quasi_monte_carlo/src`` on sys.path (so ``import qmc`` works without an
editable install) and selects the scalar Mitsuba variant — the scalar
``Renderer`` + QMC samplers run under ``scalar_rgb`` (the vec integrator, which
needs ``llvm_ad_rgb``, is NOT used here; it only produced the reference EXR).
"""
import os
import sys

import mitsuba as mi

_SRC = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# The proven box-rfilter scene loader lives in baseline/baseline.py. Reuse it
# rather than re-deriving the (subtle) rfilter-injection regex here.
_BASELINE = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "baseline"))
if _BASELINE not in sys.path:
    sys.path.insert(0, _BASELINE)

mi.set_variant("scalar_rgb")
