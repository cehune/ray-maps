import pytest
import numpy as np
import math
from unittest.mock import MagicMock
import mitsuba as mi
import drjit as dr

from segments.cluster import Cluster
from segments.primitives import SurfacePoint, Segment, SegmentTechnique
from segments.sampler_types import SequentialSampler
from segments.mmis import MMIS
from segments.propagation import Propagation

from tests.utils import make_endpoint_si
# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def make_scene_aabb(scene):
    aabb = scene.bbox()
    # pad slightly so no endpoint lands on boundary
    pad  = mi.Vector3f(0.1)
    return mi.BoundingBox3f(aabb.min - pad, aabb.max + pad)

def make_surface_point(px, py, pz, nx, ny, nz, is_camera=False, is_light=False):
    si = make_endpoint_si(mi.Point3f(px, py, pz), mi.Vector3f(nx, ny, nz))
    return SurfacePoint(si=si, is_camera=is_camera, is_light=is_light)


def make_surface_point_with_mock_bsdf(px, py, pz, nx, ny, nz, bsdf_pdf_value=1.0, bsdf_eval_value=None):
    if bsdf_eval_value is None:
        # Lambertian: eval = albedo/pi, here albedo=1
        bsdf_eval_value = mi.Color3f(1.0 / math.pi)

    mock_bsdf = MagicMock()
    mock_bsdf.eval_pdf.return_value = bsdf_pdf_value
    mock_bsdf.eval.return_value = bsdf_eval_value

    mock_si = MagicMock(spec=mi.SurfaceInteraction3f)
    mock_si.p = mi.Point3f(px, py, pz)
    mock_si.n = mi.Normal3f(nx, ny, nz)
    mock_si.bsdf.return_value = mock_bsdf

    return SurfacePoint(si=mock_si)


def make_segment(x, y, technique=SegmentTechnique.CAMERA, **kwargs):
    return Segment(x=x, y=y, technique=technique, **kwargs)


def make_minimal_cluster(segments, kernel_radius=0.5):
    n = len(segments)

    c = Cluster(c=5)

    aabb = mi.BoundingBox3f()
    for s in segments:
        aabb.expand(mi.Point3f(float(s.x.p[0]), float(s.x.p[1]), float(s.x.p[2])))
        aabb.expand(mi.Point3f(float(s.y.p[0]), float(s.y.p[1]), float(s.y.p[2])))
    pad = mi.Vector3f(kernel_radius)
    aabb = mi.BoundingBox3f(aabb.min - pad, aabb.max + pad)

    c.set_scene_aabb(aabb)
    c.set_segments(segments)

    seg_indices = np.repeat(np.arange(n, dtype=np.int32), 2)
    which_ends  = np.tile([0, 1], n).astype(np.int32)
    c.endpoint_metadata = np.stack([seg_indices, which_ends], axis=1)

    c.sorted_indices      = np.arange(2 * n, dtype=np.int32)
    c.cluster_ranges      = [(0, 2 * n)]
    c.endpoint_to_cluster = np.zeros(2 * n, dtype=np.int32)

    positions = np.array(
        [[float(s.x.p[0]), float(s.x.p[1]), float(s.x.p[2])] for s in segments] +
        [[float(s.y.p[0]), float(s.y.p[1]), float(s.y.p[2])] for s in segments],
        dtype=np.float64
    )
    normals = np.array(
        [[float(s.x.n[0]), float(s.x.n[1]), float(s.x.n[2])] for s in segments] +
        [[float(s.y.n[0]), float(s.y.n[1]), float(s.y.n[2])] for s in segments],
        dtype=np.float64
    )
    c.cluster_mean_positions = positions.mean(axis=0, keepdims=True)
    c.cluster_mean_normals   = normals.mean(axis=0, keepdims=True)

    return c


def make_samplers():
    return {SegmentTechnique.CAMERA: SequentialSampler()}


def make_mmis():
    return MMIS()

def make_flat_scene():
    """
    Single diffuse plane at y=0, Lambertian albedo=1.
    Enough to get real SurfaceInteraction3f objects with valid BSDFs.
    """
    return mi.load_dict({
        "type": "scene",
        "floor": {
            "type": "rectangle",
            "to_world": mi.ScalarTransform4f.scale([10, 10, 1])
                                            .rotate([1,0,0], 90),
            "bsdf": {
                "type": "diffuse",
                "reflectance": {"type": "rgb", "value": [1, 1, 1]}
            }
        },
        "integrator": {"type": "path"},
        "sensor": {
            "type": "perspective",
            "to_world": mi.ScalarTransform4f.look_at(
                origin=[0, 5, 0], target=[0, 0, 0], up=[0, 0, 1]
            ),
            "film": {
                "type": "hdrfilm",
                "width": 4, "height": 4
            }
        }
    })


def make_real_surface_point(scene, px, py, pz, dx, dy, dz, is_camera=False, is_light=False):
    """
    Cast a ray into the scene and return a SurfacePoint from the real intersection.
    (dx,dy,dz) is the ray direction — should point toward the plane.
    """
    ray = mi.Ray3f(mi.Point3f(px, py, pz), mi.Vector3f(dx, dy, dz))
    si  = scene.ray_intersect(ray)
    assert si.is_valid(), f"Ray from ({px},{py},{pz}) dir ({dx},{dy},{dz}) missed scene"
    sp = SurfacePoint.from_intersection(si)
    sp.is_camera = is_camera
    sp.is_light  = is_light
    return sp

# ── scene fixtures ────────────────────────────────────────────────────────────

def make_two_plane_scene():
    return mi.load_dict({
        "type": "scene",
        "floor": {
            "type": "rectangle",
            "to_world": mi.ScalarTransform4f.rotate([1,0,0], 90).scale([5,5,1]),
            "bsdf": {"type": "diffuse",
                     "reflectance": {"type": "rgb", "value": [1, 1, 1]}}
        },
        "ceiling": {
            "type": "rectangle",
            "to_world": mi.ScalarTransform4f.translate([0,2,0])
                                            .rotate([1,0,0], -90)
                                            .scale([5,5,1]),
            "bsdf": {"type": "diffuse",
                     "reflectance": {"type": "rgb", "value": [1, 1, 1]}}
        },
        "integrator": {"type": "path"},
        "sensor": {
            "type": "perspective",
            "to_world": mi.ScalarTransform4f.look_at(
                origin=[0,1,-5], target=[0,1,0], up=[0,1,0]),
            "film": {"type": "hdrfilm", "width": 4, "height": 4}
        }
    })


def hit(scene, ox, oy, oz, dx, dy, dz):
    """Shoot a ray, assert it hits, return SurfacePoint."""
    ray = mi.Ray3f(mi.Point3f(ox, oy, oz), dr.normalize(mi.Vector3f(dx, dy, dz)))
    si  = scene.ray_intersect(ray)
    assert bool(si.is_valid()), f"ray from ({ox},{oy},{oz}) dir ({dx},{dy},{dz}) missed"
    return SurfacePoint.from_intersection(si)


def camera_point(px, py, pz, nx, ny, nz):
    return SurfacePoint(si=make_endpoint_si(mi.Point3f(px,py,pz),
                                            mi.Vector3f(nx,ny,nz)),
                        is_camera=True)


def seg(x, y, is_light=False, Le=0.0):
    y.is_light = is_light
    if is_light:
        y.Le = mi.Color3f(Le)
    return Segment(x=x, y=y, technique=SegmentTechnique.CAMERA)

class TestHelp:

    @pytest.fixture(scope="class")
    def scene(self):
        return make_two_plane_scene()

    # ── conditional_pdf (mock ok — never calls shift_invariant_bsdf) ──────────

    def test_conditional_pdf_lambertian_known_geometry(self):
        pw  = 1.0 / math.pi
        aux = make_segment(
            x=make_surface_point(0,-1,0, 0,1,0),
            y=make_surface_point_with_mock_bsdf(0,0,0, 0,1,0, bsdf_pdf_value=pw),
        )
        s = make_segment(
            x=make_surface_point(0,0,0, 0,1,0),
            y=make_surface_point(0,1,0, 0,1,0),
        )
        result   = SequentialSampler().conditional_pdf(s, aux)
        expected = (1.0/math.pi) * 1.0 / 1.0**2
        assert abs(result - expected)/expected < 0.01, f"got {result:.6e} expected {expected:.6e}"

    def test_conditional_pdf_45_degree(self):
        # pw is the solid-angle PDF returned by bsdf.eval_pdf — for Lambertian at 45°
        # this already includes the cosine: pw = cos(45°)/π.
        # conditional_pdf = pw * cos_at_seg_y / len_sq
        #                 = (cos45/π) * cos45 / 2   [cos45 at seg.y, len=√2 → len_sq=2]
        cos45 = 1.0/math.sqrt(2)
        pw    = cos45/math.pi
        aux   = make_segment(
            x=make_surface_point(0,-1,0, 0,1,0),
            y=make_surface_point_with_mock_bsdf(0,0,0, 0,1,0, bsdf_pdf_value=pw),
        )
        s = make_segment(
            x=make_surface_point(0,0,0, 0,1,0),
            y=make_surface_point(1,1,0, 0,1,0),
        )
        result   = SequentialSampler().conditional_pdf(s, aux)
        expected = pw * cos45 / (math.sqrt(2)**2)   # = cos²45 / (2π)
        assert abs(result - expected)/expected < 0.01, f"got {result:.6e} expected {expected:.6e}"

    def test_conditional_pdf_zero_for_light_auxiliary(self):
        aux = make_segment(
            x=make_surface_point(0,-1,0, 0,1,0),
            y=make_surface_point(0,0,0, 0,1,0, is_light=True),
        )
        s = make_segment(
            x=make_surface_point(0,0,0, 0,1,0),
            y=make_surface_point(0,1,0, 0,1,0),
        )
        assert SequentialSampler().conditional_pdf(s, aux) == 0.0

    def test_conditional_pdf_zero_for_camera_auxiliary(self):
        aux = make_segment(
            x=make_surface_point(0,-1,0, 0,1,0),
            y=make_surface_point(0,0,0, 0,1,0, is_camera=True),
        )
        s = make_segment(
            x=make_surface_point(0,0,0, 0,1,0),
            y=make_surface_point(0,1,0, 0,1,0),
        )
        assert SequentialSampler().conditional_pdf(s, aux) == 0.0

    # ── G cancellation (mock ok) ──────────────────────────────────────────────

    def test_g_cancellation_single_auxiliary(self):
        pw    = 1.0/math.pi
        aux   = make_segment(
            x=make_surface_point(0,-1,0, 0,1,0),
            y=make_surface_point_with_mock_bsdf(0,0,0, 0,1,0, bsdf_pdf_value=pw),
        )
        s     = make_segment(
            x=make_surface_point(0,0,0, 0,1,0),
            y=make_surface_point(0,1,0, 0,1,0),
        )
        p = SequentialSampler().conditional_pdf(s, aux)
        assert abs(float(s.geom_term)/p - math.pi)/math.pi < 0.01, \
            f"G/p = {float(s.geom_term)/p:.4f}, expected pi={math.pi:.4f}"

    def test_g_cancellation_n_equal_auxiliaries(self):
        pw = 1.0/math.pi
        for n in [1, 3, 10]:
            auxs = [make_segment(
                        x=make_surface_point(0,-1,0, 0,1,0),
                        y=make_surface_point_with_mock_bsdf(0,0,0, 0,1,0, bsdf_pdf_value=pw))
                    for _ in range(n)]
            s    = make_segment(
                        x=make_surface_point(0,0,0, 0,1,0),
                        y=make_surface_point(0,1,0, 0,1,0))
            sampler = SequentialSampler()
            p_sum   = sum(sampler.conditional_pdf(s, a) for a in auxs)
            total   = n * (1.0/p_sum) * float(s.geom_term)
            assert abs(total - math.pi)/math.pi < 0.01, \
                f"n={n}: n*G/p_sum={total:.4f}, expected pi={math.pi:.4f}"

    # ── propagation (real scene required) ────────────────────────────────────

    def test_single_hop_radiance_reaches_predecessor(self, scene):
        """
        camera->p0 + p0->p1 (light).
        With K removed from propagation: result = fr * (G/p) * Le = Le  [fr=1/π, G/p=π].
        Uses mock surfaces with n=(0,0,1) so wo=(0,0,1) is in the upper hemisphere.
        """
        Le = 1.0
        bsdf_pdf = 1.0 / math.pi  # Lambertian at normal incidence

        # p0: predecessor's y-endpoint and light's x-endpoint (same position)
        p0 = make_surface_point_with_mock_bsdf(0, 0, 0, 0, 0, 1, bsdf_pdf_value=bsdf_pdf)
        # p1: light y-endpoint, directly above p0 (distance=1, wo=(0,0,1) above surface)
        p1 = make_surface_point_with_mock_bsdf(0, 0, 1, 0, 0, 1, bsdf_pdf_value=bsdf_pdf)
        p1.is_light = True
        p1.Le = mi.Color3f(Le)

        s_pred  = seg(x=camera_point(0, 0, -1, 0, 0, 1), y=p0)
        s_light = seg(x=p0, y=p1, is_light=True, Le=Le)
        s_light.radiance_in = mi.Color3f(Le)

        kr         = 2.0
        cluster    = make_minimal_cluster([s_pred, s_light], kr)
        samplers   = make_samplers()
        mmis       = make_mmis()
        propagator = Propagation(num_prop_iterations=1)

        mmis.compute_all_mmis_weights(cluster, samplers)
        for s in cluster.segments:
            s.radiance_in = s.Le
        propagator.propagate_all_segments(cluster, samplers)

        # K is removed from propagation; result = fr * (G/p) * Le = Le
        result = float(s_pred.radiance_out[0])
        assert result > 0, "radiance_out is zero"
        assert abs(result - Le) / Le < 0.15, \
            f"got {result:.4e}, expected Le={Le:.4e}"

    def test_single_hop_scales_with_Le(self, scene):
        bsdf_pdf = 1.0 / math.pi

        def run(Le):
            p0 = make_surface_point_with_mock_bsdf(0, 0, 0, 0, 0, 1, bsdf_pdf_value=bsdf_pdf)
            p1 = make_surface_point_with_mock_bsdf(0, 0, 1, 0, 0, 1, bsdf_pdf_value=bsdf_pdf)
            p1.is_light = True
            p1.Le = mi.Color3f(Le)
            s_pred  = seg(x=camera_point(0, 0, -1, 0, 0, 1), y=p0)
            s_light = seg(x=p0, y=p1, is_light=True, Le=Le)
            s_light.radiance_in = mi.Color3f(Le)
            cluster    = make_minimal_cluster([s_pred, s_light], 2.0)
            mmis       = make_mmis()
            propagator = Propagation(num_prop_iterations=1)
            mmis.compute_all_mmis_weights(cluster, make_samplers())
            for s in cluster.segments:
                s.radiance_in = s.Le
            propagator.propagate_all_segments(cluster, make_samplers())
            return float(s_pred.radiance_out[0])

        r1, r5, r20 = run(1.0), run(5.0), run(20.0)
        assert r1 > 0, "single hop gave zero"
        assert abs(r5/r1   -  5.0) <  0.5, f"ratio={r5/r1:.3f}"
        assert abs(r20/r1  - 20.0) <  2.0, f"ratio={r20/r1:.3f}"

    def test_two_hop_attenuation_is_exactly_K_per_hop(self, scene):
        """
        With K removed from propagation, each hop propagates Le without attenuation
        (G/p cancellation). Verify r1 > 0 and r2 > 0 — both hops produce nonzero
        radiance — and that r2/r1 ≈ 1.0 (the G/p cancellation property).
        """
        Le = 1.0
        kr = 0.5
        bsdf_pdf = 1.0 / math.pi

        def make_pt(z, is_light=False):
            p = make_surface_point_with_mock_bsdf(0, 0, z, 0, 0, 1, bsdf_pdf_value=bsdf_pdf)
            if is_light:
                p.is_light = True
                p.Le = mi.Color3f(Le)
            return p

        def run_hops(n_hops):
            if n_hops == 1:
                p0, p1 = make_pt(0), make_pt(1, is_light=True)
                segs = [
                    seg(x=camera_point(0, 0, -1, 0, 0, 1), y=p0),
                    seg(x=p0, y=p1, is_light=True, Le=Le),
                ]
            else:
                p0, p1, p2 = make_pt(0), make_pt(1), make_pt(2, is_light=True)
                segs = [
                    seg(x=camera_point(0, 0, -1, 0, 0, 1), y=p0),
                    seg(x=p0, y=p1),
                    seg(x=p1, y=p2, is_light=True, Le=Le),
                ]
            segs[-1].radiance_in = mi.Color3f(Le)

            cluster    = make_minimal_cluster(segs, kr)
            mmis       = make_mmis()
            # This test checks the raw G/p cancellation algebra; the synthetic
            # two-source cluster has a row gain of exactly 1.6 (not energy-
            # conserving by construction), which the production row_gain_cap
            # would project down to 1.0. Pin the stabilizers off.
            propagator = Propagation(num_prop_iterations=n_hops,
                                     row_gain_cap=None,
                                     merge_min_len_factor=None)
            mmis.compute_all_mmis_weights(cluster, make_samplers())
            for s in cluster.segments:
                s.radiance_in = s.Le
            for _ in range(n_hops):
                propagator.propagate_all_segments(cluster, make_samplers())
                propagator.swap_segment_radiance(cluster)
            return float(segs[0].radiance_in[0])

        r1 = run_hops(1)
        r2 = run_hops(2)

        assert r1 > 0, "one-hop gave zero — propagation is broken"
        assert r2 > 0, "two-hop gave zero — propagation is broken"
        # Both tests use make_minimal_cluster which puts ALL endpoints in one cluster.
        # With the corrected conditional_pdf (distance = dist(aux.y, seg.y)):
        #   1-hop: s_light has 1 auxiliary (s_pred.y, dist=1).  mmis_w = π.
        #          r1 = π * (1/π) * 1 * Le = 1.
        #   2-hop: s_light has 2 auxiliaries: s_pred.y (dist=2 → pdf=1/(4π)) and
        #          s_mid.y (dist=1 → pdf=1/π).  p_sum = 5/(4π), mmis_w = 4π/5.
        #          s_pred also receives DIRECTLY from s_light each hop (single cluster),
        #          plus indirectly through s_mid → r2 = 4/5 + 4/5 = 8/5 = 1.6.
        assert abs(r2/r1 - 1.6) < 0.15, \
            f"r1={r1:.4e} r2={r2:.4e} ratio={r2/r1:.4f}, expected ≈ 1.6"

    def test_mmis_balanced_populations(self, scene):
        """n_aux == n_cont: with balanced populations result must equal Le (G/p=π cancels fr=1/π)."""
        Le, n, kr = 1.0, 3, 0.5
        bsdf_pdf = 1.0 / math.pi

        # n predecessors (camera segs) and n light segs in one cluster
        aux_pts  = [make_surface_point_with_mock_bsdf(i*0.01, 0, 0, 0, 0, 1,
                                                       bsdf_pdf_value=bsdf_pdf) for i in range(n)]
        emit_pts = [make_surface_point_with_mock_bsdf(i*0.01, 0, 1, 0, 0, 1,
                                                       bsdf_pdf_value=bsdf_pdf) for i in range(n)]
        for p in emit_pts:
            p.is_light = True
            p.Le = mi.Color3f(Le)

        aux_segs  = [seg(x=camera_point(i*0.01, 0, -1, 0, 0, 1), y=aux_pts[i])
                     for i in range(n)]
        cont_segs = [seg(x=aux_pts[i], y=emit_pts[i], is_light=True, Le=Le)
                     for i in range(n)]
        for s in cont_segs:
            s.radiance_in = mi.Color3f(Le)

        cluster    = make_minimal_cluster(aux_segs + cont_segs, kr)
        mmis       = make_mmis()
        propagator = Propagation(num_prop_iterations=1)
        mmis.compute_all_mmis_weights(cluster, make_samplers())
        for s in cluster.segments:
            s.radiance_in = s.Le
        propagator.propagate_all_segments(cluster, make_samplers())

        # With K removed: result = fr * (G/p) * Le = Le
        expected = Le
        for i, s in enumerate(aux_segs):
            result = float(s.radiance_out[0])
            assert abs(result - expected) / expected < 0.1, \
                f"aux[{i}]: got {result:.4e} expected {expected:.4e}"
            
    def test_mmis_weight_nonzero_for_interior_segment(self, scene):
        """s_light is an interior segment (x not camera/light). Its mmis_weight must be nonzero."""
        bsdf_pdf = 1.0 / math.pi
        # p0 shared between s_pred.y and s_light.x; p1 is the light y-endpoint above p0
        p0 = make_surface_point_with_mock_bsdf(0, 0, 0, 0, 0, 1, bsdf_pdf_value=bsdf_pdf)
        p1 = make_surface_point_with_mock_bsdf(0, 0, 1, 0, 0, 1, bsdf_pdf_value=bsdf_pdf)
        p1.is_light = True
        p1.Le = mi.Color3f(1.0)

        s_pred  = seg(x=camera_point(0, 0, -1, 0, 0, 1), y=p0)
        s_light = seg(x=p0, y=p1, is_light=True, Le=1.0)

        cluster = make_minimal_cluster([s_pred, s_light], 2.0)
        mmis    = make_mmis()
        mmis.compute_all_mmis_weights(cluster, make_samplers())

        assert s_light.mmis_weight > 0, \
            f"s_light.mmis_weight={s_light.mmis_weight}, should be nonzero"


    def test_mmis_weight_skipped_for_camera_origin(self, scene):
        """s_pred has x.is_camera=True so mmis_weight must be pinned to 1.0."""
        floor_a = hit(scene, 0, 5, 0,  0,-1,0)
        floor_b = hit(scene, 0.01, 5, 0,  0,-1,0)

        s_pred  = seg(x=camera_point(0,10,0, 0,-1,0), y=floor_a)
        s_light = seg(x=floor_a, y=floor_b, is_light=True, Le=1.0)

        cluster = make_minimal_cluster([s_pred, s_light], 2.0)
        mmis    = make_mmis()
        mmis.compute_all_mmis_weights(cluster, make_samplers())

        assert s_pred.mmis_weight == 1.0, \
            f"s_pred.mmis_weight={s_pred.mmis_weight}, should be 1.0"


    def test_cluster_contains_x_endpoint_of_successor(self, scene):
        """
        s_pred.y and s_light.x are the same point.
        s_pred's y-cluster must contain s_light's x-endpoint (which_end==0).
        This is what propagate_segment iterates over.
        """
        floor_a = hit(scene, 0, 5, 0,  0,-1,0)
        floor_b = hit(scene, 0.01, 5, 0,  0,-1,0)
        floor_b.is_light = True

        s_pred  = seg(x=camera_point(0,10,0, 0,-1,0), y=floor_a)
        s_light = seg(x=floor_a, y=floor_b, is_light=True, Le=1.0)

        cluster = make_minimal_cluster([s_pred, s_light], 2.0)

        # s_pred is seg_idx=0, its y-endpoint is flat_idx=1
        y_cluster_idx = cluster.endpoint_to_cluster[0 * 2 + 1]
        start, end    = cluster.cluster_ranges[y_cluster_idx]

        x_endpoints_in_cluster = [
            cluster.endpoint_metadata[cluster.sorted_indices[i]]
            for i in range(start, end)
            if cluster.endpoint_metadata[cluster.sorted_indices[i]][1] == 0
        ]

        # s_light is seg_idx=1, its x-endpoint should appear here
        s_light_x_present = any(seg_idx == 1 for seg_idx, _ in x_endpoints_in_cluster)
        assert s_light_x_present, \
            f"s_light x-endpoint not found in s_pred y-cluster. " \
            f"x-endpoints present: {x_endpoints_in_cluster}"


    def test_propagate_segment_finds_neighbor(self, scene):
        """
        propagate_segment for s_pred must find s_light as a neighbor and
        return nonzero radiance_out. Isolates the inner loop logic.
        Uses mock surfaces with n=(0,0,1) so wo=(0,0,1) is in the upper
        hemisphere and bsdf.eval returns a nonzero value.
        """
        Le       = 1.0
        bsdf_pdf = 1.0 / math.pi
        p0 = make_surface_point_with_mock_bsdf(0, 0, 0, 0, 0, 1, bsdf_pdf_value=bsdf_pdf)
        p1 = make_surface_point_with_mock_bsdf(0, 0, 1, 0, 0, 1, bsdf_pdf_value=bsdf_pdf)
        p1.is_light = True
        p1.Le = mi.Color3f(Le)

        s_pred  = seg(x=camera_point(0, 0, -1, 0, 0, 1), y=p0)
        s_light = seg(x=p0, y=p1, is_light=True, Le=Le)
        s_light.radiance_in = mi.Color3f(Le)
        s_light.mmis_weight = 1.0   # bypass MMIS for this test

        kr         = 2.0
        cluster    = make_minimal_cluster([s_pred, s_light], kr)
        propagator = Propagation(num_prop_iterations=1)

        result = propagator.propagate_segment(s_pred, 0, cluster, make_samplers())
        assert float(dr.max(result)) > 0, \
            f"propagate_segment returned zero even with mmis_weight=1"
    
    def test_mmis_auxiliary_found_in_cluster(self, scene):
        """p_sum must be nonzero — at least one auxiliary must be found."""
        floor_a = hit(scene, 0,    5, 0,  0,-1,0)
        floor_b = hit(scene, 0.01, 5, 0,  0,-1,0)
        floor_b.is_light = True

        s_pred  = seg(x=camera_point(0,10,0, 0,-1,0), y=floor_a)
        s_light = seg(x=floor_a, y=floor_b, is_light=True, Le=1.0)

        cluster = make_minimal_cluster([s_pred, s_light], 2.0)

        # manually walk the same loop as compute_mmis_weight for s_light (seg_idx=1)
        x_cluster_idx = cluster.endpoint_to_cluster[1 * 2 + 0]
        start, end    = cluster.cluster_ranges[x_cluster_idx]

        found = []
        for flat_idx in cluster.sorted_indices[start:end]:
            aux_seg_idx, which_end = cluster.endpoint_metadata[flat_idx]
            t = cluster.segments[aux_seg_idx]
            found.append((aux_seg_idx, which_end, t.technique,
                        t.y.is_camera, t.y.is_light))

        y_endpoints = [(i, we, tech, ic, il)
                    for i, we, tech, ic, il in found if we == 1]

        assert len(y_endpoints) > 0, \
            f"no y-endpoints found in s_light x-cluster. all found: {found}"
        assert any(not ic and not il for _, _, _, ic, il in y_endpoints), \
            f"all y-endpoints are camera or light, conditional_pdf returns 0: {y_endpoints}"
    def test_shift_invariant_bsdf_nonzero(self, scene):
        # Use mock surfaces with n=(0,0,1) so wo=(0,0,1) is in the upper hemisphere.
        # s_pred: camera(0,0,-1) → p0(0,0,0); s_next: p0 → p1(0,0,1).
        # wo = (0,0,1), cos_o = 1 → bsdf.eval / 1 = 1/π ≠ 0.
        p0 = make_surface_point_with_mock_bsdf(0, 0, 0, 0, 0, 1)
        p1 = make_surface_point_with_mock_bsdf(0, 0, 1, 0, 0, 1)

        s_pred = seg(x=camera_point(0, 0, -1, 0, 0, 1), y=p0)
        s_next = seg(x=p0, y=p1)

        result = SequentialSampler.shift_invariant_bsdf(s_pred, s_next)
        assert float(dr.max(result)) > 0, \
            f"shift_invariant_bsdf returned zero: {result}"
    
    def test_propagate_segment_finds_neighbor_alt(self, scene):
        """Second variant: same logic, different label to avoid duplicate name."""
        Le       = 1.0
        bsdf_pdf = 1.0 / math.pi
        p0 = make_surface_point_with_mock_bsdf(0, 0, 0, 0, 0, 1, bsdf_pdf_value=bsdf_pdf)
        p1 = make_surface_point_with_mock_bsdf(0, 0, 1, 0, 0, 1, bsdf_pdf_value=bsdf_pdf)
        p1.is_light = True
        p1.Le = mi.Color3f(Le)

        s_pred  = seg(x=camera_point(0, 0, -1, 0, 0, 1), y=p0)
        s_light = seg(x=p0, y=p1, is_light=True, Le=Le)
        s_light.radiance_in = mi.Color3f(Le)
        s_light.mmis_weight = 1.0

        kr         = 2.0
        cluster    = make_minimal_cluster([s_pred, s_light], kr)
        propagator = Propagation(num_prop_iterations=1)

        result = propagator.propagate_segment(s_pred, 0, cluster, make_samplers())
        assert float(dr.max(result)) > 0, \
            f"propagate_segment returned zero even with mmis_weight=1"
    
    def test_single_pair_energy_conservation_with_factors_printed(self):
        """
        Two-segment setup with everything analytically computable:
        - s_pred = (camera @ z=-1, p0 @ z=0)
        - s_light = (p0 @ z=0, p1 @ z=1) with p1 emitting Le=1
        - All surfaces Lambertian, albedo=1, normals = +z
        - One cluster containing all endpoints

        Expected per-factor:
        mmis_w(s_light) = π   (1 / (1/π))
        shift_invariant_bsdf  = 1/π  (Lambertian eval / cos)
        geom_term(s_light)    = 1    (cos*cos/dist² = 1*1/1)
        radiance_in(s_light)  = Le = 1
        contrib               = π · 1/π · 1 · 1 = 1.0
        """
        Le_value = 1.0
        bsdf_pdf = 1.0 / math.pi
        bsdf_eval = mi.Color3f(1.0 / math.pi)  # Lambertian, albedo=1

        p0 = make_surface_point_with_mock_bsdf(0, 0, 0, 0, 0, 1,
                                            bsdf_pdf_value=bsdf_pdf,
                                            bsdf_eval_value=bsdf_eval)
        p1 = make_surface_point_with_mock_bsdf(0, 0, 1, 0, 0, 1,
                                            bsdf_pdf_value=bsdf_pdf,
                                            bsdf_eval_value=bsdf_eval)

        s_pred  = seg(x=camera_point(0, 0, -1, 0, 0, 1), y=p0)
        s_light = seg(x=p0, y=p1, is_light=True, Le=Le_value)
        s_light.radiance_in = mi.Color3f(Le_value)

        cluster    = make_minimal_cluster([s_pred, s_light], kernel_radius=2.0)
        samplers   = make_samplers()
        mmis       = MMIS()
        propagator = Propagation(num_prop_iterations=1)

        # ---- MMIS ----
        mmis.compute_all_mmis_weights(cluster, samplers)
        mmis_w_pred  = float(s_pred.mmis_weight)
        mmis_w_light = float(s_light.mmis_weight)

        # ---- propagation ----
        for s in cluster.segments:
            s.radiance_in = s.Le
        propagator.propagate_all_segments(cluster, samplers)

        # ---- gather factors ----
        sampler = samplers[SegmentTechnique.CAMERA]
        f_color = sampler.shift_invariant_bsdf(s_pred, s_light)
        f_val   = float(f_color[0])
        geom    = float(s_light.geom_term)
        rad_in  = float(s_light.radiance_in[0])
        result  = float(s_pred.radiance_out[0])
        expected_contrib = mmis_w_light * f_val * geom * rad_in

        # ---- print the breakdown ----
        print("\n── single-pair sanity ────────────────────────────────────")
        print(f"  mmis_w(s_pred)  = {mmis_w_pred:.6f}   (expected 1.0   — trivial)")
        print(f"  mmis_w(s_light) = {mmis_w_light:.6f}   (expected π   = {math.pi:.6f})")
        print(f"  f (BSDF)        = {f_val:.6f}   (expected 1/π = {1/math.pi:.6f})")
        print(f"  geom_term       = {geom:.6f}   (expected 1.0)")
        print(f"  radiance_in     = {rad_in:.6f}   (expected 1.0)")
        print(f"  contrib (calc)  = {expected_contrib:.6f}   (expected 1.0)")
        print(f"  radiance_out    = {result:.6f}   (expected 1.0)")
        print("──────────────────────────────────────────────────────────")

        # ---- assertions ----
        assert math.isclose(mmis_w_pred,  1.0,        rel_tol=1e-6), "s_pred not trivial"
        assert math.isclose(mmis_w_light, math.pi,    rel_tol=1e-4), \
            f"mmis_w(s_light) drift: got {mmis_w_light}, expected π"
        assert math.isclose(f_val,        1/math.pi,  rel_tol=1e-4), \
            f"BSDF eval drift: got {f_val}, expected 1/π"
        assert math.isclose(geom,         1.0,        rel_tol=1e-4), \
            f"geom_term drift: got {geom}, expected 1.0"
        assert math.isclose(result,       1.0,        rel_tol=0.05), \
            f"PROPAGATION DRIFT: got {result}, expected 1.0 — investigate which factor diverges"
            

class TestClustering:

    @pytest.fixture(scope="class")
    def scene(self):
        return make_two_plane_scene()

    def test_consecutive_endpoints_share_cluster(self, scene):
        """
        In a path s0->s1->s2, s0.y and s1.x are the same point.
        After clustering they must land in the same cluster.
        """
        floor  = hit(scene, 0, 5, 0,  0,-1,0)
        ceil   = hit(scene, 0,-1, 0,  0, 1,0)
        floor2 = hit(scene, 0, 5, 0,  0,-1,0)  # second floor hit, same point

        s0 = seg(x=camera_point(0,10,0, 0,-1,0), y=floor)
        s1 = seg(x=floor, y=ceil)
        s2 = seg(x=ceil,  y=floor2, is_light=True, Le=1.0)

        rng     = np.random.default_rng(42)
        cluster = Cluster(c=3)
        cluster.set_scene_aabb(make_scene_aabb(scene))
        cluster.set_segments([s0, s1, s2])
        cluster.cluster(rng)

        # s0.y and s1.x are the same point — must share a cluster
        s0_y_cluster = cluster.endpoint_to_cluster[0 * 2 + 1]  # flat_idx of s0.y
        s1_x_cluster = cluster.endpoint_to_cluster[1 * 2 + 0]  # flat_idx of s1.x
        assert s0_y_cluster == s1_x_cluster, \
            f"s0.y in cluster {s0_y_cluster}, s1.x in cluster {s1_x_cluster} — path broken"

    def test_consecutive_endpoints_share_cluster_s1_s2(self, scene):
        """Same check for s1.y and s2.x."""
        floor  = hit(scene, 0, 5, 0,  0,-1,0)
        ceil   = hit(scene, 0,-1, 0,  0, 1,0)
        floor2 = hit(scene, 0, 5, 0,  0,-1,0)

        s0 = seg(x=camera_point(0,10,0, 0,-1,0), y=floor)
        s1 = seg(x=floor, y=ceil)
        s2 = seg(x=ceil,  y=floor2, is_light=True, Le=1.0)

        rng     = np.random.default_rng(42)
        cluster = Cluster(c=3)
        cluster.set_scene_aabb(make_scene_aabb(scene))
        cluster.set_segments([s0, s1, s2])
        cluster.cluster(rng)

        s1_y_cluster = cluster.endpoint_to_cluster[1 * 2 + 1]
        s2_x_cluster = cluster.endpoint_to_cluster[2 * 2 + 0]
        assert s1_y_cluster == s2_x_cluster, \
            f"s1.y in cluster {s1_y_cluster}, s2.x in cluster {s2_x_cluster} — path broken"

    def test_chain_fully_recoverable(self, scene):
        """
        Given a known path s0->s1->s2, for each segment si, propagate_segment
        must find s_{i+1} as a neighbor by walking the cluster of si.y.
        This is the full connectivity check.
        """
        floor  = hit(scene, 0, 5, 0,  0,-1,0)
        ceil   = hit(scene, 0,-1, 0,  0, 1,0)
        floor2 = hit(scene, 0, 5, 0,  0,-1,0)

        s0 = seg(x=camera_point(0,10,0, 0,-1,0), y=floor)
        s1 = seg(x=floor, y=ceil)
        s2 = seg(x=ceil,  y=floor2, is_light=True, Le=1.0)
        segs = [s0, s1, s2]

        rng     = np.random.default_rng(42)
        cluster = Cluster(c=3)
        cluster.set_scene_aabb(make_scene_aabb(scene))
        cluster.set_segments(segs)
        cluster.cluster(rng)

        for i in range(len(segs) - 1):
            si      = segs[i]
            si_next = segs[i + 1]

            y_cluster_idx = cluster.endpoint_to_cluster[i * 2 + 1]
            start, end    = cluster.cluster_ranges[y_cluster_idx]

            found = False
            for pos in range(start, end):
                flat_idx             = cluster.sorted_indices[pos]
                neighbor_idx, which_end = cluster.endpoint_metadata[flat_idx]
                if which_end == 0 and neighbor_idx == i + 1:
                    found = True
                    break

            assert found, \
                f"s{i}.y cluster does not contain s{i+1}.x — " \
                f"chain broken at link {i}->{i+1}"