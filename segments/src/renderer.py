from pool import SegmentPool
from sampler import *
from primitives import *
import matplotlib.pyplot as plt
from dataclasses import dataclass, field

import numpy as np

@dataclass
class Renderer:
    segment_pool: SegmentPool = field(default_factory=SegmentPool)
    variant: str = 'scalar_rgb'
    # this is just for step 5 gathering, only the first segment per chain actually contributes
    # because it contains the Wp (sensor responsivity) term
    camera_first_segments: list = field(default_factory=list)
    
    def sample_step(self, scene, sensor, sampler, height, width, is_bidirectional=False):
        for y in range(height):
            for x in range(width):
                pixel_idx = y * width + x
                pos_sample = mi.Point2f(x + sampler.next_1d(), y + sampler.next_1d())
                ap_sample  = mi.Point2f(sampler.next_1d(), sampler.next_1d())
                film_pos   = mi.Point2f(pos_sample.x / width, pos_sample.y / height)

                ray, ray_weight = sensor.sample_ray(
                    time=0.0,
                    sample1=sampler.next_1d(),
                    sample2=film_pos,
                    sample3=ap_sample
                )
                si = scene.ray_intersect(ray)

                segments = sample_camera_path(scene, sampler, ray, ray_weight, si)
                if segments: # if it just hits something invalid, then the segment itself is invalid
                    self.camera_first_segments.append(segments[0])
                    self.segment_pool.add_camera_path(segments)
                else:
                    self.camera_first_segments.append(None)

    
    def render_iterate_(self, scene, sensor, sampler, height = 128, width = 128, spp = 16):
        # we clear on every iteration lol
        self.segment_pool.clear()
        self.camera_first_segments.clear() 

        # step 1 is sampling globally!!
        self.sample_step(scene, sensor, sampler, height, width, spp)

        # 2. clustering

        # 3. initialize radiance in
        for path in self.segment_pool.camera_paths:
            for seg in path:
                seg.radiance_in = seg.y.Le  # zero for non-emissive, Le for light hits
        
        # 4. mmis weights

        # 5. propogate backward
        for path in self.segment_pool.camera_paths:
            for i in reversed(range(len(path) - 1)):
                path[i].radiance_in = path[i].throughput * path[i + 1].radiance_in

        # 6. gather
        accum = np.zeros((height * width, 3))
        for pixel_idx, first_seg in enumerate(self.camera_first_segments):
            if first_seg is None:
                continue
            val = first_seg.throughput * first_seg.radiance_in
            accum[pixel_idx] = [float(val[0]), float(val[1]), float(val[2])]

        return accum.reshape(height, width, 3)
    
    def render_iterate_loop(self, scene, height=128, width=128, n_iterations=64):
        sensor = scene.sensors()[0]
        accum  = np.zeros((height, width, 3), dtype=np.float32)

        for i in range(n_iterations):
            sampler = mi.load_dict({'type': 'independent', 'sample_count': 1})
            sampler.seed(i, width * height)

            accum += self.render_iterate_(scene, sensor, sampler, height, width)

            if i % 8 == 0:
                print(f'  iteration {i}/{n_iterations}')

        return accum / n_iterations
    
    def save_render_by_scene_path(self, scene_path, save_file_path = "output.png",  width=128, height=128, spp=32):
        mi.set_variant(self.variant)

        scene  = mi.load_file(scene_path)
        # image  = self.render(scene, width, height, spp)
        image = self.render_iterate_loop(scene, height, width, spp)

        # tone map and save
        plt.imsave(save_file_path, np.clip(image ** (1/2.2), 0, 1))
        print(f'saved ${save_file_path}')


    
    def render(self, scene, width=512, height=512, spp=64):
        sensor  = scene.sensors()[0]
        sampler = mi.load_dict({'type': 'independent', 'sample_count': spp})
        sampler.seed(0, width * height)

        accum = np.zeros((height, width, 3), dtype=np.float32)

        for y in range(height):
            for x in range(width):
                pixel_color = mi.Color3f(0.0)

                # pixel approximator integral sum
                for _ in range(spp):
                    # sample position within pixel + aperture sample
                    pos_sample  = mi.Point2f(x + sampler.next_1d(),
                                            y + sampler.next_1d())
                    ap_sample   = mi.Point2f(sampler.next_1d(), sampler.next_1d())

                    # normalized film coords [0,1]^2
                    film_pos    = mi.Point2f(pos_sample.x / width,
                                            pos_sample.y / height)

                    # ray weight contains Wp * G / p so we only need to take care of the radiance
                    ray, ray_weight = sensor.sample_ray(
                        time=0.0,
                        sample1=sampler.next_1d(),
                        sample2=film_pos,
                        sample3=ap_sample
                    )
                    si = scene.ray_intersect(ray)

                    segments    = sample_camera_path(scene, sampler, ray, ray_weight, si)

                    L           = evaluate_path_base(segments)
                    self.segment_pool.add_camera_path(segments)
                    pixel_color += ray_weight * L

                accum[y, x] = np.array([
                    float(pixel_color[0]),
                    float(pixel_color[1]),
                    float(pixel_color[2])
                ]) / spp

            if y % 32 == 0:
                print(f'  row {y}/{height}')

        return accum
    
