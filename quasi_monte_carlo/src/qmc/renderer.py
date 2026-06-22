import mitsuba as mi
import numpy as np

from qmc.pathtracer import BasicPathTracer
from qmc.sampler.sampler import Sampler


class Renderer:
    """Scalar QMC driver. Holds one integrator; you hand render() any Sampler."""

    def __init__(self, scene, *, max_depth=8, rr_depth=3):
        self.scene = scene
        props = mi.Properties()
        props["max_depth"] = max_depth
        props["rr_depth"]  = rr_depth
        self.integrator = BasicPathTracer(props)        # one integrator, against the interface

    def render(self, sampler, spp):
        """Returns an (H, W, 3) float32 buffer: box filter, mean over spp."""
        if not isinstance(sampler, Sampler):
            raise TypeError("render() takes a QMC Sampler, not a mi.Sampler")

        sensor = self.scene.sensors()[0]
        size   = sensor.film().crop_size()              # ScalarVector2u: x=width, y=height
        W, H   = int(size.x), int(size.y)
        accum  = np.zeros((H, W, 3), dtype=np.float64)

        for y in range(H):
            for x in range(W):
                for i in range(spp):
                    sampler.setup_path(x, y, i)         # bind this (pixel, sample)

                    jx, jy = sampler.cam_2d("jitter")   # dims 0,1: pixel jitter
                    lens   = mi.Point2f(*sampler.cam_2d("lens"))   # dims 2,3: aperture
                    pos    = mi.Point2f((x + jx) / W, (y + jy) / H)

                    ray, weight = sensor.sample_ray(0.0, 0.5, pos, lens)
                    L, _, _ = self.integrator.sample(self.scene, sampler, ray)

                    accum[y, x] += np.asarray(weight * L, dtype=np.float64).reshape(3)

        return (accum / spp).astype(np.float32)

    def save_render(self, sampler, spp, out_path):
        """Render, then write. EXR direct, PNG sRGB-gamma encoded."""
        img = self.render(sampler, spp)
        bmp = mi.Bitmap(img)
        if out_path.lower().endswith(".png"):
            bmp = bmp.convert(mi.Bitmap.PixelFormat.RGB,
                              mi.Struct.Type.UInt8, srgb_gamma=True)
        bmp.write(out_path)
        return img