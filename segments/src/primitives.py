import mitsuba as mi
import drjit as dr
from dataclasses import dataclass, field
from enum import IntEnum

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
    Le: mi.Color3f = field(default_factory=lambda: mi.Color3f(0.0))
    is_camera: bool = False
    is_light: bool = False

    @property
    def p(self) -> mi.Vector3f:
        return self.si.p

    @property
    def n(self) -> mi.Vector3f:
        return self.si.n

    @staticmethod
    def from_intersection(si: mi.SurfaceInteraction3f, **kwargs) -> 'SurfacePoint':
        assert si.is_valid()
        return SurfacePoint(si=si, **kwargs)
    
"""
endpoints dont have native surface interactions, this just generates one for us
"""
def make_endpoint_si(p: mi.Vector3f, n: mi.Vector3f) -> mi.SurfaceInteraction3f:
    si = mi.SurfaceInteraction3f()
    si.p = p
    si.n = n
    si.sh_frame = mi.Frame3f(n)  # at least a valid frame derived from n
    return si

class SegmentTechnique(IntEnum):
    CAMERA = 0
    LIGHT  = 1
    BRIDGE = 2

@dataclass
class Segment:
    x: SurfacePoint
    y: SurfacePoint
    wo_local: mi.Vector3f = None
    wi_local: mi.Vector3f = None
    throughput: mi.Color3f = field(default_factory=lambda: mi.Color3f(1.0))
    # radiance in is the Le term, radiance out is found via the integral
    radiance_in: mi.Color3f = field(default_factory=lambda: mi.Color3f(0.0)) # incoming radiance before prop
    radiance_out: mi.Color3f = field(default_factory=lambda: mi.Color3f(0.0)) # after propogation
    # probability equation (15)
    pdf: float = 1.0 
    mmis_weight: float = 0.0
    cluster_idx: int = -1  
    technique: SegmentTechnique = SegmentTechnique.CAMERA
    
    def __post_init__(self):
        # assert self.y.si.is_valid(), "endpoint is not valid on segment!!!!"
        difference = self.y.p - self.x.p
        self.len = dr.norm(difference)
        
        if self.len < 1e-8:  # tune this threshold to your scene scale
            raise ValueError(f"Degenerate segment: length {self.len} too small")
        # DIR IS EQUIVALENT TO S HAT IN THE PAPER
        self.dir = difference / self.len

        # geometric term is (normal of x dot dir) * (normal of y dot dir) / (len^2)
        # seen write after equation 4
        # abs because we can get negative values if the normal is facing away from the direction
        self.geom_term = (dr.abs(dr.dot(self.x.n, self.dir)) * dr.abs(dr.dot(self.y.n, self.dir))) * \
            (1.0 / (self.len * self.len))
        
        wi_world = dr.normalize(self.x.p - self.y.p)
        self.wi_local = mi.Frame3f(self.x.n).to_local(wi_world)
        self.y.si.wi = self.wi_local

        
