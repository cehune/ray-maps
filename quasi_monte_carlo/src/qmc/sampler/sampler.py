"""
Interface for all samplers.

A sampler maps a *named path decision* — the camera lens, bounce b's BSDF
direction, etc. — to one or two uniforms in [0,1). The decision is addressed, not
counted: the integrator asks for ``bounce_2d(b, "dir")`` and the sampler turns
that into a fixed logical dimension. Because the dimension is a pure function of
the decision, it is identical no matter how the path branched to reach it — a
specular vertex skipping NEE, Russian roulette firing early, paths of different
lengths all address the same dimension for the same decision. There is no cursor
to fall out of sync, so dimension misalignment is unrepresentable.

Layout (shared contract with the integrator):

    camera   : jitter (2D) + lens (2D)                      -> dims 0..3
    bounce b : a block of BOUNCE_STRIDE dims at CAMERA_DIMS + b*BOUNCE_STRIDE
                 nee (2D) @ +0,1   lobe (1D) @ +2
                 dir (2D) @ +3,4   rr   (1D) @ +5

BOUNCE_STRIDE must be >= the most dimensions any single bounce can draw (today:
NEE 2 + lobe 1 + dir 2 + RR 1 = 6). Sub-slots for a 2D decision reserve two
consecutive dimensions so nothing overlaps.

Subclasses implement only ``setup_path`` and the two value primitives ``_one`` /
``_pair``; the layout and the public API live here, in one place.
"""

from abc import ABC, abstractmethod

CAMERA_DIMS = 4
BOUNCE_STRIDE = 6
CAM = {"jitter": 0, "lens": 2}                    # logical dim of each camera slot
SLOT = {"nee": 0, "lobe": 2, "dir": 3, "rr": 5}   # offset within a bounce block


class Sampler(ABC):
    camera_dims = CAMERA_DIMS
    stride = BOUNCE_STRIDE

    # ── decision -> logical dimension (the entire layout lives here) ───────────
    def _cam_dim(self, slot: str) -> int:
        return CAM[slot]

    def _bounce_dim(self, bounce: int, slot: str) -> int:
        return self.camera_dims + int(bounce) * self.stride + SLOT[slot]

    # ── public addressable API (stateless, no call-order dependence) ──────────
    def cam_2d(self, slot: str) -> tuple[float, float]:
        return self._pair(self._cam_dim(slot))

    def bounce_2d(self, bounce: int, slot: str) -> tuple[float, float]:
        return self._pair(self._bounce_dim(bounce, slot))

    def bounce_1d(self, bounce: int, slot: str) -> float:
        return self._one(self._bounce_dim(bounce, slot))

    # ── per-sampler primitives ────────────────────────────────────────────────
    @abstractmethod
    def setup_path(self, px, py, sample_index):
        """Bind the sampler to a (pixel, sample) before the path is traced."""

    @abstractmethod
    def _one(self, dim: int) -> float:
        """One uniform in [0,1) for logical dimension `dim`."""

    @abstractmethod
    def _pair(self, dim: int) -> tuple[float, float]:
        """A coherent 2D sample anchored at logical dimension `dim`."""
