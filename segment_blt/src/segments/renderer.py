from segments.pool import SegmentPool, SegmentPoolV2
from segments.segment_gen_sampler import *
from segments.primitives import *
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from segments.cluster import Cluster
from segments.mmis import MMIS
from segments.sampler_types import *
from segments.propagation import Propagation
from segments.pair_cache import build_pair_cache
import time
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_blt_components(scene, kernel_radius=16, kernel_weight=0.67, num_prop_iterations=4):
    mmis        = MMIS()
    propagation = Propagation(
        kernel_radius=kernel_radius,
        kernel_weight=kernel_weight,
        num_prop_iterations=num_prop_iterations,
    )
    cluster = Cluster()
    cluster.set_scene_aabb(scene.bbox())
    samplers = Sampler.build_registry(SequentialSampler())
    return cluster, mmis, propagation, samplers


def _sample_segments(scene, sensor, sampler, height, width):
    """
    Sample one camera path per pixel.
    Returns (segment_pool, camera_first_segments).
    camera_first_segments[pixel_idx] is segments[0] for that pixel, or None.
    """
    segment_pool = SegmentPoolV2()
    camera_first_segments = []

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

    return segment_pool, camera_first_segments


def _final_gather(camera_first_segments, height, width):
    """
    Collect radiance_in * throughput * mmis_weight per pixel after propagation.
    """
    accum = np.zeros((height * width, 3))
    for pixel_idx, first_seg in enumerate(camera_first_segments):
        if first_seg is None:
            continue
        val = first_seg.throughput * first_seg.radiance_in * first_seg.mmis_weight
        accum[pixel_idx] = [float(val.x), float(val.y), float(val.z)]
    return accum.reshape(height, width, 3)

def _diag(label, cluster, pair_cache, camera_first_segments):
    """Print key diagnostic stats to help locate where radiance is lost."""
    S = len(cluster.segments)
    P = len(pair_cache.prop_i)
    M = len(pair_cache.mmis_j)
    T = len(pair_cache.trivial_mmis)

    n_light_segs = sum(1 for s in cluster.segments if s.y.is_light)
    n_nonzero_Le = sum(1 for s in cluster.segments if float(s.Le.x) > 0)
    n_nonzero_mmis = sum(1 for s in cluster.segments if s.mmis_weight > 0)
    n_nonzero_rad  = sum(
        1 for s in cluster.segments if float(s.radiance_in.x) > 0
    )
    n_nonzero_out  = sum(
        1 for s in camera_first_segments
        if s is not None and float(s.radiance_in.x) > 0
    )

    print(f"  [{label}] S={S} segs | P={P} prop-pairs | M={M} mmis-pairs | "
          f"T={T} trivial")
    print(f"         light-terminal={n_light_segs} | Le>0={n_nonzero_Le} | "
          f"mmis_w>0={n_nonzero_mmis} | rad_in>0={n_nonzero_rad} | "
          f"camera-first rad>0={n_nonzero_out}")

@dataclass
class Renderer:
    segment_pool: SegmentPool = field(default_factory=SegmentPool)
    segment_pool_v2: SegmentPoolV2 = None
    variant: str = 'scalar_rgb'

    # ── 1. Reference path tracer (no BLT) ────────────────────────────────────

    def render(self, scene, width=512, height=512, spp=64):
        """
        Simple unidirectional path tracer.  Useful as a reference image.
        Uses evaluate_path_base: only direct-light-hit paths contribute.
        """
        sensor  = scene.sensors()[0]
        sampler = mi.load_dict({'type': 'independent', 'sample_count': spp})
        sampler.seed(0, width * height)

        accum = np.zeros((height, width, 3), dtype=np.float32)

        for y in range(height):
            for x in range(width):
                pixel_color = mi.Color3f(0.0)

                for _ in range(spp):
                    pos_sample = mi.Point2f(x + sampler.next_1d(),
                                            y + sampler.next_1d())
                    ap_sample  = mi.Point2f(sampler.next_1d(), sampler.next_1d())
                    film_pos   = mi.Point2f(pos_sample.x / width,
                                            pos_sample.y / height)

                    ray, ray_weight = sensor.sample_ray(
                        time=0.0,
                        sample1=sampler.next_1d(),
                        sample2=film_pos,
                        sample3=ap_sample,
                    )
                    si = scene.ray_intersect(ray)
                    segments   = sample_camera_path(scene, sampler, ray, ray_weight, si)
                    L          = evaluate_path_base(segments)
                    self.segment_pool.add_camera_path(segments)
                    pixel_color += ray_weight * L

                accum[y, x] = np.array([
                    float(pixel_color[0]),
                    float(pixel_color[1]),
                    float(pixel_color[2]),
                ]) / spp

            if y % 32 == 0:
                print(f'  row {y}/{height}')

        return accum

    # ── 2. BLT non-vectorized (reference BLT implementation) ─────────────────

    def _render_iterate_nonvec(self, scene, sensor, sampler, height, width,
                               cluster, mmis, propagation, samplers, iteration=0,
                               verbose=False):
        segment_pool, camera_first_segments = _sample_segments(
            scene, sensor, sampler, height, width
        )

        if not segment_pool.paths:
            return np.zeros((height, width, 3))

        cluster.set_segments(segment_pool.paths)
        cluster.cluster(np.random.default_rng(seed=iteration))

        # MMIS weights (non-vec)
        mmis.compute_all_mmis_weights(cluster, samplers)

        # Propagation (non-vec)
        propagation.iterate_propogation(cluster, samplers)

        if verbose:
            pair_cache = build_pair_cache(
                cluster, samplers
            )
            _diag("nonvec", cluster, pair_cache, camera_first_segments)

        return _final_gather(camera_first_segments, height, width)

    def render_nonvec(self, scene, height=128, width=128, n_iterations=8,
                      kernel_radius=16, kernel_weight=0.67,
                      num_prop_iterations=4, verbose=True):
        """
        BLT render using the non-vectorized MMIS + propagation path.
        Use as a reference to compare against render_vec().
        """
        sensor = scene.sensors()[0]
        accum  = np.zeros((height, width, 3), dtype=np.float32)
        cluster, mmis, propagation, samplers = _make_blt_components(
            scene, kernel_radius, kernel_weight, num_prop_iterations
        )

        for i in range(n_iterations):
            print(f"[nonvec] iter {i}/{n_iterations}"
                  f" — r={propagation.kernel_radius:.3f}")
            sampler = mi.load_dict({'type': 'independent', 'sample_count': 1})
            sampler.seed(i, width * height)
            accum += self._render_iterate_nonvec(
                scene, sensor, sampler, height, width,
                cluster, mmis, propagation, samplers,
                iteration=i, verbose=verbose,
            )

        return accum / n_iterations

    # ── 3. BLT vectorized ─────────────────────────────────────────────────────

    def _render_iterate_vec(self, scene, sensor, sampler, height, width,
                            cluster, mmis, propagation, samplers, iteration=0,
                            verbose=False):
        
        t0 = time.time()
        segment_pool, camera_first_segments = _sample_segments(
            scene, sensor, sampler, height, width
        )
        print(f"sampling: {time.time()-t0:.1f}s")

        if not segment_pool.paths:
            return np.zeros((height, width, 3))
        t0 = time.time()
        cluster.set_segments(segment_pool.paths)
        cluster.cluster(np.random.default_rng(seed=iteration))
        print(f"cluster: {time.time()-t0:.1f}s")       
        t0 = time.time()
        # Build pair cache once — BSDF + PDF computed here, reused in propagation
        pair_cache = build_pair_cache(cluster, samplers)
        print(f"pair: {time.time()-t0:.1f}s")

        # MMIS weights (vec)
        t0 = time.time()
        mmis.compute_all_mmis_weights_vec(cluster, pair_cache)
        print(f"mmis: {time.time()-t0:.1f}s")

        # Propagation (vec)
        t0 = time.time()
        propagation.iterate_propogation_vec(cluster, pair_cache)
        print(f"prop: {time.time()-t0:.1f}s")

        if verbose:
            _diag("vec", cluster, pair_cache, camera_first_segments)

        t0 = time.time()
        bin = _final_gather(camera_first_segments, height, width)
        print(f"sampling: {time.time()-t0:.1f}s")
        return bin

    def render_vec(self, scene, height=128, width=128, n_iterations=8,
                   kernel_radius=16, kernel_weight=0.67,
                   num_prop_iterations=8, verbose=True):
        """
        BLT render using the vectorized (numpy) MMIS + propagation path.
        """
        sensor = scene.sensors()[0]
        accum  = np.zeros((height, width, 3), dtype=np.float32)
        cluster, mmis, propagation, samplers = _make_blt_components(
            scene, kernel_radius, kernel_weight, num_prop_iterations
        )

        for i in range(n_iterations):
            print(f"[vec] iter {i}/{n_iterations}"
                  f" — r={propagation.kernel_radius:.3f}")
            sampler = mi.load_dict({'type': 'independent', 'sample_count': 1})
            sampler.seed(i, width * height)
            accum += self._render_iterate_vec(
                scene, sensor, sampler, height, width,
                cluster, mmis, propagation, samplers,
                iteration=i, verbose=verbose,
            )

        return accum / n_iterations

    # ── 4. Save helpers ───────────────────────────────────────────────────────

    def save_render_by_scene_path(self, scene_path, save_file_path="output.png",
                                  width=128, height=128, n_iterations=16,
                                  mode='vec'):
        """
        mode: 'ref'    → simple path tracer (render)
              'nonvec' → BLT non-vectorized
              'vec'    → BLT vectorized  (default)
        """
        mi.set_variant(self.variant)
        scene = mi.load_file(scene_path)

        if mode == 'ref':
            image = self.render(scene, width, height, spp=n_iterations)
        elif mode == 'nonvec':
            image = self.render_nonvec(scene, height, width,
                                       n_iterations=n_iterations)
        else:
            image = self.render_vec(scene, height, width,
                                    n_iterations=n_iterations, verbose=False)

        plt.imsave(save_file_path, np.clip(image ** (1 / 2.2), 0, 1))
        print(f'saved {save_file_path}')

    # ── Legacy entry point (kept for back-compat) ─────────────────────────────

    def render_iterate_loop(self, scene, height=128, width=128, n_iterations=8):
        """Alias for render_vec() with default BLT parameters."""
        return self.render_vec(scene, height, width, n_iterations=n_iterations)
