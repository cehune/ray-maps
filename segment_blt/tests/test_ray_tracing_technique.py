"""
Unit + MMIS-integration tests for the independent ray-tracing technique
(RayTracingSampler) and its wiring into the MMIS denominator.

Two things here are easy to get wrong, so they get dedicated tests:
  * the pdf is UNCONDITIONAL and computed from geometry, p(s)=G(s)/(pi|M|),
    NOT read from the stored sample_parea (which only exists on RT-drawn segs);
  * it enters the denominator ONCE per segment, never once per auxiliary.

Run in your venv:  pytest tests/test_ray_tracing_technique.py -q
"""
import math
import numpy as np
import pytest

mi = pytest.importorskip("mitsuba")

from tests.utils import make_surface_point, make_segment, make_aabb
from segments.primitives import SegmentTechnique
from segments.sampler_types import RayTracingSampler
from segments.cluster import Cluster
from segments.mmis import MMIS
from unittest.mock import MagicMock


def _unit_segment():
    # (0,0,0)->(0,0,1), both normals +z  =>  geom_term = |n_x·ŝ||n_y·ŝ|/len² = 1
    return make_segment(
        make_surface_point(0, 0, 0, 0, 0, 1),
        make_surface_point(0, 0, 1, 0, 0, 1),
    )


class TestRayTracingPdf:
    def test_pdf_is_geom_over_pi_area(self):
        seg = _unit_segment()                                 # geom_term == 1.0
        rt = RayTracingSampler(scene_area=4.0)
        assert float(seg.geom_term) == pytest.approx(1.0)
        assert rt.conditional_pdf(seg) == pytest.approx(1.0 / (math.pi * 4.0))

    def test_unconditional_ignores_auxiliary(self):
        seg, other = _unit_segment(), _unit_segment()
        rt = RayTracingSampler(scene_area=4.0)
        base = rt.conditional_pdf(seg)
        assert rt.conditional_pdf(seg, None) == base          # default arg
        assert rt.conditional_pdf(seg, other) == base         # auxiliary is ignored

    def test_scales_inverse_area(self):
        seg = _unit_segment()
        assert (RayTracingSampler(2.0).conditional_pdf(seg)
                == pytest.approx(2.0 * RayTracingSampler(4.0).conditional_pdf(seg)))

    def test_uses_geometry_not_sample_parea(self):
        """Guards the correction: the denominator pdf must come from geom_term so
        it is defined for camera segments too (which carry no sample_parea)."""
        seg = _unit_segment()
        seg.sample_parea = 999.0                              # deliberately wrong stored value
        rt = RayTracingSampler(scene_area=4.0)
        assert rt.conditional_pdf(seg) == pytest.approx(1.0 / (math.pi * 4.0))


class TestRayTracingInMMIS:
    @pytest.mark.parametrize("n_aux", [0, 2, 5])
    def test_rt_added_once_regardless_of_auxiliary_count(self, n_aux):
        """With only RT registered, the denominator is exactly p_RT(main) — added
        once, independent of how many auxiliaries sit in the cluster. If it were
        added inside the auxiliary loop the weight would shrink with n_aux."""
        mmis = MMIS()
        main_seg = _unit_segment()                            # geom_term == 1.0
        A = 4.0
        rt = RayTracingSampler(scene_area=A)
        samplers = {SegmentTechnique.RAY_TRACING: rt}         # CAMERA deliberately absent

        aux = [MagicMock(technique=SegmentTechnique.CAMERA) for _ in range(n_aux)]
        segments = aux + [main_seg]
        main_idx = len(aux)

        cluster = MagicMock()
        cluster.segments = segments
        cluster.endpoint_to_cluster = np.zeros(2 * len(segments), dtype=np.int32)
        cluster.cluster_ranges = [(0, n_aux)]
        cluster.sorted_indices = np.arange(n_aux, dtype=np.int32)   # aux y-endpoints
        meta = [[i, 1] for i in range(n_aux)]                       # each aux's y-endpoint
        cluster.endpoint_metadata = (np.array(meta, dtype=np.int32)
                                     if meta else np.empty((0, 2), dtype=np.int32))

        weight = mmis.compute_mmis_weight(main_seg, main_idx, cluster, samplers)
        # p_RT = 1/(pi*A)  =>  weight = 1/p_RT = pi*A, for every n_aux
        assert weight == pytest.approx(math.pi * A)
        assert weight == pytest.approx(1.0 / rt.conditional_pdf(main_seg))

    def test_parity_and_denominator_growth(self):
        """Real cluster: (a) non-vec and vec agree with RT registered; (b) adding
        RT strictly lowers interior weights (denominator grew); (c) the
        camera-origin trivial segment stays exactly 1.0 in both paths."""
        from segments.pair_cache import build_pair_cache

        s0 = make_segment(                                    # camera-origin: trivial
            make_surface_point(0.0, 0.0, 0.0, 0, 1, 0, is_camera=True),
            make_surface_point(0.30, 0.30, 0.30, 0, 1, 0),
        )
        interior = [
            make_segment(make_surface_point(*x, 0, 1, 0), make_surface_point(*y, 0, 1, 0))
            for x, y in [
                ((0.30, 0.30, 0.30), (0.60, 0.60, 0.60)),
                ((0.31, 0.30, 0.30), (0.70, 0.50, 0.50)),
                ((0.32, 0.31, 0.29), (0.55, 0.55, 0.65)),
                ((0.29, 0.30, 0.31), (0.50, 0.70, 0.55)),
                ((0.30, 0.31, 0.32), (0.65, 0.55, 0.60)),
                ((0.31, 0.29, 0.30), (0.58, 0.62, 0.58)),
            ]
        ]
        segs = [s0] + interior
        interior_idx = np.arange(1, len(segs))

        def fresh_cluster():
            c = Cluster(c=5)
            c.set_scene_aabb(make_aabb([0, 0, 0], [1, 1, 1]))
            c.set_segments(segs)
            c.cluster(np.random.default_rng(0), voxel_size=0.2)   # deterministic
            return c

        seq = MagicMock()
        seq.technique_type = SegmentTechnique.CAMERA
        seq.conditional_pdf.return_value = 0.4
        seq.shift_invariant_bsdf.return_value = mi.Color3f(0.0)
        rt = RayTracingSampler(scene_area=4.0)
        mmis = MMIS()

        # without RT (baseline denominator = sum of sequential pdfs only)
        c = fresh_cluster()
        mmis.compute_all_mmis_weights(c, {SegmentTechnique.CAMERA: seq})
        w_no_rt = np.array([s.mmis_weight for s in c.segments])

        samplers = {SegmentTechnique.CAMERA: seq, SegmentTechnique.RAY_TRACING: rt}

        # with RT — non-vec
        c = fresh_cluster()
        mmis.compute_all_mmis_weights(c, samplers)
        nonvec = np.array([s.mmis_weight for s in c.segments])

        # with RT — vec (via PairCache.p_rt)
        c = fresh_cluster()
        pc = build_pair_cache(c, samplers)
        assert pc.p_rt.size == len(segs) and pc.p_rt[0] == 0.0   # trivial seg pinned
        mmis.compute_all_mmis_weights_vec(c, pc)
        vec = np.array([s.mmis_weight for s in c.segments])

        # (a) parity
        assert np.allclose(nonvec, vec, atol=1e-9), f"nonvec {nonvec} != vec {vec}"
        # (c) trivial segment unaffected
        assert nonvec[0] == 1.0 and vec[0] == 1.0
        # (b) RT strictly grows the denominator => interior weights strictly smaller
        assert np.all(nonvec[interior_idx] < w_no_rt[interior_idx] - 1e-12), (
            f"RT term should lower interior weights:\n"
            f"  with RT:    {nonvec[interior_idx]}\n"
            f"  without RT: {w_no_rt[interior_idx]}"
        )
