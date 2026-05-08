import mitsuba as mi
from primitives import *
from sampler import *
import numpy as np
import matplotlib.pyplot as plt


def render(scene, width=512, height=512, spp=64):
    sensor  = scene.sensors()[0]
    sampler = mi.load_dict({'type': 'independent', 'sample_count': spp})
    sampler.seed(0, width * height)

    accum = np.zeros((height, width, 3), dtype=np.float32)

    for y in range(height):
        for x in range(width):
            pixel_color = mi.Color3f(0.0)

            for _ in range(spp):
                # sample position within pixel + aperture sample
                pos_sample  = mi.Point2f(x + sampler.next_1d(),
                                         y + sampler.next_1d())
                ap_sample   = mi.Point2f(sampler.next_1d(), sampler.next_1d())

                # normalized film coords [0,1]^2
                film_pos    = mi.Point2f(pos_sample.x / width,
                                         pos_sample.y / height)

                ray, ray_weight = sensor.sample_ray(
                    time=0.0,
                    sample1=sampler.next_1d(),
                    sample2=film_pos,
                    sample3=ap_sample
                )

                segments    = sample_camera_path(scene, sampler, ray)
                L           = evaluate_path_base(segments)
                pixel_color += ray_weight * L

            accum[y, x] = np.array([
                float(pixel_color[0]),
                float(pixel_color[1]),
                float(pixel_color[2])
            ]) / spp

        if y % 32 == 0:
            print(f'  row {y}/{height}')

    return accum


# ---- main ------------------------------------------------------------------

def main():
    mi.set_variant('scalar_rgb')

    scene  = mi.load_file('../samples/cbox/cbox.xml')
    image  = render(scene, width=128, height=128, spp=32)

    # tone map and save
    plt.imsave('output.png', np.clip(image ** (1/2.2), 0, 1))
    print('saved output.png')


if __name__ == '__main__':
    main()