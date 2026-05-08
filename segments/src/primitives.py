import mitsuba as mi
import drjit as dr
from dataclasses import dataclass, field

mi.set_variant('scalar_rgb')
"""
available for mac 26
['scalar_rgb', 'scalar_spectral', 'scalar_spectral_polarized', 
'llvm_ad_rgb', 'llvm_ad_mono', 'llvm_ad_mono_polarized', 
'llvm_ad_spectral', 'llvm_ad_spectral_polarized']
"""

@dataclass
class SurfacePoint:
    si: mi.SurfaceInteraction3f = None
    p: mi.Vector3f = None
    n: mi.Vector3f = None
    bsdf: mi.BSDF = None
    Le: mi.Color3f = field(default_factory=lambda: mi.Color3f(0.0))
    is_camera: bool = False
    is_light: bool = False

    def __post_init__(self):
        assert ((self.p is not None and self.n is not None) or self.si is not None), "Need to insert some params"

        if (self.si is not None):
            assert self.si.is_valid(), "Cannot construct SurfacePoint from invalid intersection"
            self.p = self.si.p
            self.n = self.si.n
            self.bsdf = self.si.bsdf()

@dataclass
class Segment:
    x: SurfacePoint
    y: SurfacePoint
    throughput: mi.Color3f = field(default_factory=lambda: mi.Color3f(1.0))
    pdf: float = 1.0
    visible: bool = True
    is_camera_path: bool = True

    def __post_init__(self):
        if self.y.si is not None:
            assert self.y.si.is_valid(), "Segment endpoint y is invalid"

        difference = self.y.p - self.x.p
        self.len = dr.norm(difference)

        difference = self.y.p - self.x.p
        self.len = dr.norm(difference)
        
        if self.len < 1e-4:  # tune this threshold to your scene scale
            raise ValueError(f"Degenerate segment: length {self.len} too small")
        # DIR IS EQUIVALENT TO S HAT IN THE PAPER
        self.dir = difference / self.len

        # geometric term is (normal of x dot dir) * (normal of y dot dir) / (len^2)
        # seen write after equation 4
        # abs because we can get negative values if the normal is facing away from the direction
        self.geom_term = (dr.abs(dr.dot(self.x.n, self.dir)) * dr.abs(dr.dot(self.y.n, self.dir))) * \
            (1.0 / (self.len * self.len))
        
