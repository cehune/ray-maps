from pool import SegmentPool, SegmentPoolV2
from segments.src.segment_gen_sampler import *
from primitives import *
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from segments.src.cluster import Cluster
from segments.src.mmis import MMIS
from segments.src.sampler_types import *
from segments.src.propagation import Propagation

import numpy as np

@dataclass
class Renderer:
    segment_pool: SegmentPool = field(default_factory=SegmentPool)
    segment_pool_v2: SegmentPoolV2 = None
    variant: str = 'scalar_rgb'
    # this is just for step 5 gathering, only the first segment per chain actually contributes
    # because it contains the Wp (sensor responsivity) term
    camera_first_segments: list = field(default_factory=list)
    
    variant: str = 'scalar_rgb'

    def render_iterate_(self, scene, sensor, sampler, height, width):
        segment_pool = SegmentPool()
        camera_first_segments = []

        # 1. sample segments
        for y in range(height):
            for x in range(width):
                pos_sample = mi.Point2f(x + sampler.next_1d(), y + sampler.next_1d())
                ap_sample  = mi.Point2f(sampler.next_1d(), sampler.next_1d())
                film_pos   = mi.Point2f(pos_sample.x / width, pos_sample.y / height)

                ray, ray_weight = sensor.sample_ray(
                    time=0.0,
                    sample1=sampler.next_1d(),
                    sample2=film_pos,
                    sample3=ap_sample,
                )
                si = scene.ray_intersect(ray)
                segments = sample_camera_path(scene, sampler, ray, ray_weight, si)

                if segments:
                    camera_first_segments.append(segments[0])
                    segment_pool.add_path(segments)
                else:
                    camera_first_segments.append(None)

        # 2. cluster
        cluster = Cluster()
        cluster.set_scene_aabb(scene.bbox())
        cluster.set_segments(segment_pool.segments)  # flat list
        cluster.cluster(np.random.default_rng(seed=42))

        # 3. initialise radiance_in from Le (set at sampling time on each segment)
        # Le is already on each segment — just copy to radiance_in
        for segment in segment_pool.segments:
            segment.radiance_in = segment.Le

        # 4. MMIS weights
        self.mmis.compute_all_mmis_weights(cluster, self.samplers)

        # 5. propagate
        self.propagation.iterate_propogation(cluster, self.samplers)

        # 6. final gather — only camera-connected segments contribute W_rho
        accum = np.zeros((height * width, 3))
        for pixel_idx, first_seg in enumerate(camera_first_segments):
            if first_seg is None:
                continue
            # W_rho * G * <L> * mmis_weight
            val = first_seg.throughput * first_seg.geom_term * \
                  first_seg.radiance_in * first_seg.mmis_weight
            accum[pixel_idx] = [float(val.x), float(val.y), float(val.z)]

        return accum.reshape(height, width, 3)

    def render_iterate_loop(self, scene, height=128, width=128, n_iterations=64):
        sensor = scene.sensors()[0]
        accum  = np.zeros((height, width, 3), dtype=np.float32)

        self.mmis        = MMIS()
        self.propagation = Propagation(
            kernel_radius=4 * self._avg_pixel_footprint(sensor, height, width),
            kernel_weight=0.67 ** (1 / 1),  # alpha=0.67 per Knaus-Zwicker
            num_prop_iterations=4,
        )
        self.samplers = Sampler.build_registry(SequentialSampler())

        for i in range(n_iterations):
            sampler = mi.load_dict({'type': 'independent', 'sample_count': 1})
            sampler.seed(i, width * height)
            accum += self.render_iterate_(scene, sensor, sampler, height, width)

            if i % 8 == 0:
                print(f'  iteration {i}/{n_iterations}')

        return accum / n_iterations

    @staticmethod
    def _avg_pixel_footprint(sensor, height, width) -> float:
        """Approximate average pixel footprint for initial kernel radius."""
        fov = sensor.film().size()
        return max(1.0 / height, 1.0 / width) * 4.0

    def save_render_by_scene_path(self, scene_path, save_file_path="output.png",
                                   width=128, height=128, n_iterations=32):
        mi.set_variant(self.variant)
        scene = mi.load_file(scene_path)
        image = self.render_iterate_loop(scene, height, width, n_iterations)
        plt.imsave(save_file_path, np.clip(image ** (1 / 2.2), 0, 1))
        print(f'saved {save_file_path}')


    
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
    
