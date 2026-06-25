"""
Tests for the to-light BRIDGE technique (paper §S2 / Eq. S8).

The bridge segment (next-event estimation, expressed as its own technique)
must contribute its generation pdf p_L(x) to the MMIS denominator once per
camera-side auxiliary in the kernel — independent of the auxiliary's own
technique dispatch. In the camera-only limit this reproduces the old
`Σ_t p_bsdf + M·p_bridge` denominator exactly (no regression), but now as a
clean, separate technique instead of an additive hack inside SequentialSampler.
"""
from tests.utils import *
from segments.primitives import SegmentTechnique
from segments.sampler_types import BridgeSampler, SequentialSampler
from segments.mmis import MMIS
import numpy as np


def _bridge_segment(x, y_light, parea):
    """A to-light bridge: x = surface vertex, y = light vertex, with p_L stored."""
    seg = make_segment(
        make_surface_point(*x, 0, 1, 0),
        make_surface_point(*y_light, 0, 1, 0, is_light=True),
        technique=SegmentTechnique.BRIDGE,
    )
    seg.bridge_parea = parea
    return seg


def _camera_aux(x, y):
    return make_segment(
        make_surface_point(*x, 0, 1, 0),
        make_surface_point(*y, 0, 1, 0),
        technique=SegmentTechnique.CAMERA,
    )


class TestBridgeSampler:
    def test_returns_parea_for_light_terminal_with_camera_aux(self):
        seg = _bridge_segment((0.3, 0.3, 0.3), (0.3, 0.9, 0.3), 0.7)
        aux = _camera_aux((0.3, 0.3, 0.3), (0.6, 0.6, 0.6))
        assert BridgeSampler().conditional_pdf(seg, aux) == 0.7

    def test_zero_when_segment_not_light_terminal(self):
        # ordinary interior segment — bridge mass must not apply even if set
        seg = _camera_aux((0.3, 0.3, 0.3), (0.6, 0.6, 0.6))
        seg.bridge_parea = 0.7
        aux = _camera_aux((0.3, 0.3, 0.3), (0.55, 0.55, 0.55))
        assert BridgeSampler().conditional_pdf(seg, aux) == 0.0

    def test_zero_when_aux_is_light_terminal(self):
        # a light-terminal auxiliary is not a valid camera-side conditioner
        seg = _bridge_segment((0.3, 0.3, 0.3), (0.3, 0.9, 0.3), 0.7)
        light_aux = make_segment(
            make_surface_point(0.3, 0.3, 0.3, 0, 1, 0),
            make_surface_point(0.3, 0.95, 0.3, 0, 1, 0, is_light=True),
            technique=SegmentTechnique.CAMERA,
        )
        assert BridgeSampler().conditional_pdf(seg, light_aux) == 0.0

    def test_zero_when_aux_is_light_technique(self):
        # a light-subpath auxiliary is not a camera-side conditioner
        seg = _bridge_segment((0.3, 0.3, 0.3), (0.3, 0.9, 0.3), 0.7)
        light_tech_aux = make_segment(
            make_surface_point(0.3, 0.3, 0.3, 0, 1, 0),
            make_surface_point(0.6, 0.6, 0.6, 0, 1, 0),
            technique=SegmentTechnique.LIGHT,
        )
        assert BridgeSampler().conditional_pdf(seg, light_tech_aux) == 0.0

    def test_missing_parea_defaults_zero(self):
        seg = make_segment(
            make_surface_point(0.3, 0.3, 0.3, 0, 1, 0),
            make_surface_point(0.3, 0.9, 0.3, 0, 1, 0, is_light=True),
            technique=SegmentTechnique.BRIDGE,
        )  # no bridge_parea set
        aux = _camera_aux((0.3, 0.3, 0.3), (0.6, 0.6, 0.6))
        assert BridgeSampler().conditional_pdf(seg, aux) == 0.0


class TestBridgeInMMISDenominator:
    """The MMIS weight of a light-terminal segment must include the bridge mass
    once per camera-side auxiliary: p_sum = Σ_t p_bsdf(s'|t) + M·p_L."""

    def _mock_cluster(self, aux_0, aux_1, main):
        mock_cluster = MagicMock()
        mock_cluster.cluster_ranges = [(0, 3)]
        # cluster holds aux_0.y (flat 1), aux_1.y (flat 3), main.x (flat 4)
        mock_cluster.sorted_indices = np.array([1, 3, 4], dtype=np.int32)
        mock_cluster.endpoint_to_cluster = np.zeros(6, dtype=np.int32)
        mock_cluster.endpoint_metadata = np.array([
            [0, 0], [0, 1],   # aux_0 x, y
            [1, 0], [1, 1],   # aux_1 x, y
            [2, 0], [2, 1],   # main  x, y
        ], dtype=np.int32)
        mock_cluster.segments = [aux_0, aux_1, main]
        return mock_cluster

    def test_light_terminal_includes_bridge_mass(self):
        b, P = 0.4, 0.7          # constant bsdf pdf per aux, bridge p_L
        aux_0 = _camera_aux((0.30, 0.30, 0.30), (0.60, 0.60, 0.60))
        aux_1 = _camera_aux((0.31, 0.30, 0.30), (0.70, 0.50, 0.50))
        main  = _bridge_segment((0.30, 0.30, 0.30), (0.30, 0.90, 0.30), P)

        mock_cam = MagicMock()
        mock_cam.conditional_pdf.return_value = b
        samplers = {SegmentTechnique.CAMERA: mock_cam,
                    SegmentTechnique.BRIDGE: BridgeSampler()}

        w = MMIS().compute_mmis_weight(main, 2, self._mock_cluster(aux_0, aux_1, main), samplers)
        # 2 camera auxiliaries: p_sum = 2*b (bsdf) + 2*P (bridge)
        expected = 1.0 / (2 * b + 2 * P)
        assert abs(w - expected) < 1e-9, f"expected {expected}, got {w}"

    def test_non_light_segment_gets_no_bridge_mass(self):
        b = 0.4
        aux_0 = _camera_aux((0.30, 0.30, 0.30), (0.60, 0.60, 0.60))
        aux_1 = _camera_aux((0.31, 0.30, 0.30), (0.70, 0.50, 0.50))
        # ordinary interior main: y NOT light → bridge contributes nothing,
        # even though we set a bogus parea to prove it's gated on y.is_light
        main = _camera_aux((0.30, 0.30, 0.30), (0.65, 0.65, 0.65))
        main.bridge_parea = 999.0

        mock_cam = MagicMock()
        mock_cam.conditional_pdf.return_value = b
        samplers = {SegmentTechnique.CAMERA: mock_cam,
                    SegmentTechnique.BRIDGE: BridgeSampler()}

        w = MMIS().compute_mmis_weight(main, 2, self._mock_cluster(aux_0, aux_1, main), samplers)
        expected = 1.0 / (2 * b)
        assert abs(w - expected) < 1e-9, f"expected {expected}, got {w}"
