import numpy as np
import mitsuba as mi
from segments.tests.utils import *
from segments.src.renderer import *

mi.set_variant('scalar_rgb')

class TestRenderer:
    # no init or else pytest cant find :(
    def setup_method(self):
        self.renderer = Renderer()

    def test_first_segments_count_equals_pixels(self, scene_sensor_sampler):
        """One entry in camera_first_segments per pixel, no more no less."""
        scene, sensor, sampler = scene_sensor_sampler
        height, width = 32, 32
        self.renderer.sample_step(scene, sensor, sampler, height, width)
        assert len(self.renderer.camera_first_segments) == height * width

    def test_first_segments_are_none_or_segment(self, scene_sensor_sampler):
        """Every entry is either None (missed ray) or a valid Segment."""
        scene, sensor, sampler = scene_sensor_sampler
        self.renderer.sample_step(scene, sensor, sampler, 32, 32)
        for entry in self.renderer.camera_first_segments:
            assert entry is None or isinstance(entry, Segment)

    def test_at_least_some_segments_generated(self, scene_sensor_sampler):
        """Scene has geometry so most rays should hit something."""
        scene, sensor, sampler = scene_sensor_sampler
        self.renderer.sample_step(scene, sensor, sampler, 32, 32)
        non_none = [s for s in self.renderer.camera_first_segments if s is not None]
        assert len(non_none) > 0

    def test_first_segment_technique_is_camera(self, scene_sensor_sampler):
        """First segment of every camera path must be tagged CAMERA."""
        scene, sensor, sampler = scene_sensor_sampler
        self.renderer.sample_step(scene, sensor, sampler, 32, 32)
        for seg in self.renderer.camera_first_segments:
            if seg is not None:
                assert seg.technique == SegmentTechnique.CAMERA

    def test_no_degenerate_segments(self, scene_sensor_sampler):
        """No segment should have zero length or NaN positions."""
        scene, sensor, sampler = scene_sensor_sampler
        self.renderer.sample_step(scene, sensor, sampler, 32, 32)
        for path in self.renderer.segment_pool.camera_paths:
            for seg in path:
                assert float(seg.len) > 1e-4, "Degenerate segment length"
                px, py, pz = float(seg.x.p.x), float(seg.x.p.y), float(seg.x.p.z)
                assert not any(np.isnan([px, py, pz])), "NaN in segment x position"

    def test_radiance_in_initialized_zero(self, scene_sensor_sampler):
        """f_in should start at zero before propagation."""
        scene, sensor, sampler = scene_sensor_sampler
        self.renderer.sample_step(scene, sensor, sampler, 32, 32)
        for path in self.renderer.segment_pool.camera_paths:
            for seg in path:
                val = seg.radiance_in
                assert float(val[0]) == 0.0
                assert float(val[1]) == 0.0
                assert float(val[2]) == 0.0

    def test_throughput_is_finite(self, scene_sensor_sampler):
        """No inf or NaN throughput values."""
        scene, sensor, sampler = scene_sensor_sampler
        self.renderer.sample_step(scene, sensor, sampler, 32, 32)
        for path in self.renderer.segment_pool.camera_paths:
            for seg in path:
                for c in range(3):
                    v = float(seg.throughput[c])
                    assert np.isfinite(v), f"Non-finite throughput: {v}"

    def test_camera_paths_match_first_segments(self, scene_sensor_sampler):
        """First segment of each path matches camera_first_segments entry."""
        scene, sensor, sampler = scene_sensor_sampler
        self.renderer.sample_step(scene, sensor, sampler, 32, 32)
        path_firsts = [p[0] for p in self.renderer.segment_pool.camera_paths]
        seg_firsts  = [s for s in self.renderer.camera_first_segments if s is not None]
        assert len(path_firsts) == len(seg_firsts)
        for pf, sf in zip(path_firsts, seg_firsts):
            assert pf is sf, "First segment mismatch — possible copy instead of reference"

    def test_clear_between_calls(self, scene_sensor_sampler):
        """Calling sample_step twice without clearing should not be done —
        verify clear() resets state correctly."""
        scene, sensor, sampler = scene_sensor_sampler
        self.renderer.sample_step(scene, sensor, sampler, 32, 32)
        count_first = len(self.renderer.camera_first_segments)

        self.renderer.segment_pool.clear()
        self.renderer.camera_first_segments.clear()

        sampler.seed(1, 32 * 32)
        self.renderer.sample_step(scene, sensor, sampler, 32, 32)
        count_second = len(self.renderer.camera_first_segments)

        assert count_first == count_second


    """
    
    RENDER ITERATE
    
    """
    def test_output_shape_render_iterate(self, simple_scene):
        """Output must be (height, width, 3)."""
        height, width = 32, 32
        sensor  = simple_scene.sensors()[0]
        sampler = mi.load_dict({'type': 'independent', 'sample_count': 1})
        sampler.seed(0, height * width)
        result = self.renderer.render_iterate_(simple_scene, sensor, sampler, height, width)
        assert result.shape == (height, width, 3)

    def test_output_non_negative_render_iterate(self, simple_scene):
        """Radiance is non-negative everywhere."""
        height, width = 32, 32
        sensor  = simple_scene.sensors()[0]
        sampler = mi.load_dict({'type': 'independent', 'sample_count': 1})
        sampler.seed(0, height * width)
        result = self.renderer.render_iterate_(simple_scene, sensor, sampler, height, width)
        assert np.all(result >= 0.0), "Negative radiance values found"

    def test_output_finite_render_iterate(self, simple_scene):
        """No NaN or inf in output."""
        height, width = 32, 32
        sensor  = simple_scene.sensors()[0]
        sampler = mi.load_dict({'type': 'independent', 'sample_count': 1})
        sampler.seed(0, height * width)
        result = self.renderer.render_iterate_(simple_scene, sensor, sampler, height, width)
        assert np.all(np.isfinite(result)), "Non-finite values in output"

    def test_some_nonzero_pixels_render_iterate(self, simple_scene):
        """At least some pixels should be lit — scene has a visible light."""
        height, width = 32, 32
        sensor  = simple_scene.sensors()[0]
        sampler = mi.load_dict({'type': 'independent', 'sample_count': 1})
        sampler.seed(0, height * width)
        result = self.renderer.render_iterate_(simple_scene, sensor, sampler, height, width)
        assert np.any(result > 0.0), "All pixels are black — propagation likely broken"

    def test_accumulation_reduces_variance_render_iterate(self, simple_scene):
        """More iterations should reduce per-pixel variance."""
        height, width = 32, 32
        sensor = simple_scene.sensors()[0]

        def run_n(n):
            accum = np.zeros((height, width, 3), dtype=np.float32)
            for i in range(n):
                sampler = mi.load_dict({'type': 'independent', 'sample_count': 1})
                sampler.seed(i, height * width)
                accum += self.renderer.render_iterate_(simple_scene, sensor, sampler, height, width)
                self.renderer.segment_pool.clear()
                self.renderer.camera_first_segments.clear()
            return accum / n

        img_4  = run_n(4)
        img_32 = run_n(32)

        # variance of 32-iteration image should be lower
        var_4  = np.var(img_4)
        var_32 = np.var(img_32)
        assert var_32 < var_4, f"More iterations did not reduce variance: {var_4:.4f} vs {var_32:.4f}"

    def test_matches_reference_renderer_render_iterate(self, simple_scene):
        # TODO: Remove evnetually, will be unstable with clustering
        """
        Statistical equivalence with reference render() at same sample count.
        Mean pixel value should be within 20% — loose tolerance for MC noise.
        """
        height, width = 32, 32
        n_iter = 64
        sensor = simple_scene.sensors()[0]

        # new renderer
        accum = np.zeros((height, width, 3), dtype=np.float32)
        for i in range(n_iter):
            sampler = mi.load_dict({'type': 'independent', 'sample_count': 1})
            sampler.seed(i, height * width)
            accum += self.renderer.render_iterate_(simple_scene, sensor, sampler, height, width)
            self.renderer.segment_pool.clear()
            self.renderer.camera_first_segments.clear()
        result_new = accum / n_iter

        # reference renderer
        ref_renderer = Renderer()
        result_ref = ref_renderer.render(simple_scene, width, height, spp=n_iter)

        mean_new = np.mean(result_new)
        mean_ref = np.mean(result_ref)

        assert mean_ref > 0, "Reference renderer produced black image"
        rel_diff = abs(mean_new - mean_ref) / mean_ref
        assert rel_diff < 0.20, f"Mean radiance differs by {rel_diff*100:.1f}% from reference"