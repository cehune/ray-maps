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

def _make_blt_components(scene, add_light_samples, kernel_radius=16, kernel_weight=0.67, num_prop_iterations=4, cluster_c=10, geom_clamp_factor=None, mmis_clamp_factor=None, geom_clamp_mode="hard", alpha=2/3, row_gain_cap=1.0, merge_min_len_factor=1.0, add_bridge_segments = True):
    mmis        = MMIS()
    mmis.mmis_clamp_factor = mmis_clamp_factor
    propagation = Propagation(num_prop_iterations=num_prop_iterations,
                              geom_clamp_factor=geom_clamp_factor,
                              geom_clamp_mode=geom_clamp_mode,
                              row_gain_cap=row_gain_cap,
                              merge_min_len_factor=merge_min_len_factor)
    cluster = Cluster(c=cluster_c, alpha=alpha)
    cluster.set_scene_aabb(scene.bbox())
    # BridgeSampler is always registered: it accounts for the to-light bridge
    # technique in the MMIS denominator (paper §S2/S8). Harmless when no
    # bridge segments exist — its conditional_pdf is then 0 everywhere.
    # Always register Sequential + Bridge; add Light only when light paths are
    # enabled. (The old `if add_light_samples / if add_bridge_segments` pair
    # clobbered LightSampler when both were on, and left `samplers` undefined
    # when both were off — e.g. the --no-bridge camera-only arm.)
    techniques = [SequentialSampler(), BridgeSampler()]
    if add_light_samples:
        techniques.append(LightSampler())
    # Independent ray-tracing technique (Eq. 13): unconditional, p(s)=G(s)/(pi|M|).
    scene_area = sum(float(s.surface_area()) for s in scene.shapes())
    techniques.append(RayTracingSampler(scene_area))
    samplers = Sampler.build_registry(*techniques)
    return cluster, mmis, propagation, samplers

def _sample_segments_ray_tracing(scene, sensor, sampler, height, width,
                                 add_light_segments=False, add_bridge_segments=False,  # OFF for QMC study
                                 cam_segments_per_pixel=10):
    pool = SegmentPoolV2()

    # transport pool: independent ray-traced segments (one LDS point each)
    for _ in range(width * height * cam_segments_per_pixel):
        res = sample_surface_point(scene, sampler)
        if res is None: continue
        point, p_x = res
        out = sample_segment_from_point(scene, point, sampler, SegmentTechnique.RAY_TRACING)
        if out is None: continue
        seg, p_y_given_x = out
        seg.sample_parea = p_x * p_y_given_x 
        pool.add_segment(seg)
    return pool

def _sample_segments(scene, sensor, sampler, height, width, beta = 0.25, add_light_samples = True,
                     add_bridge_segments = True):
    """
    Sample one camera path per pixel.
    Returns (segment_pool, camera_first_segments).
    camera_first_segments[pixel_idx] is segments[0] for that pixel, or None.

    add_bridge_segments: emit one to-light bridge segment per path vertex
    (vertex → sampled emitter point, visibility-tested). These flood every
    voxel with light-terminal Le sources so the cached-direct gather D(v)
    stops being density estimation of rare BSDF emitter hits — the main
    grain mechanism vs a next-event path tracer. These are tagged technique=BRIDGE;
    MIS against BSDF light hits is automatic via BridgeSampler in the MMIS
    denominator (paper §S2/S8).
    """
    segment_pool = SegmentPoolV2()
    camera_first_segments = []
    n_pixels = height * width
    n_bridge = 0

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
            bridge_segs = [] if add_bridge_segments else None
            segments = sample_camera_path(scene, sampler, ray, ray_weight, si,
                                          bridge_out=bridge_segs)
            if bridge_segs:
                segment_pool.add_path(bridge_segs)
                n_bridge += len(bridge_segs)

            if segments:
                # stash the whole path on the first segment for the depth-d
                # split final gather (_next_seg kept for back-compat)
                segments[0]._next_seg = segments[1] if len(segments) > 1 else None
                segments[0]._path = segments
                camera_first_segments.append(segments[0])
                segment_pool.add_path(segments)
            else:
                camera_first_segments.append(None)

    # ── Light paths (β · n_pixels total) ──────────────────────────────
    if add_light_samples:
        n_light_paths = int(beta * n_pixels)
        light_paths_added = 0
        for _ in range(n_light_paths):
            light_segments = sample_light_path(scene, sampler)
            if light_segments:
                segment_pool.add_path(light_segments)
                light_paths_added += 1

    if n_bridge:
        print(f"  [bridge] {n_bridge} bridge segments added "
              f"({n_bridge/max(len(segment_pool.paths),1)*100:.0f}% of pool)")

    return segment_pool, camera_first_segments

def _readout_path(path, depth):
    """
    Depth-d split readout along the pixel's own camera path (radiance INTO
    v_1 along s_1, excluding T0):

        R_k = D(s_k) + T_{k+1} · R_{k+1}      for k < d   (split levels)
        R_d = rad_in(s_d)                                 (cache level)

    with the light-terminal drop at every split level: if s_{k+1} terminates
    on the emitter, R_k = D(s_k) — the path's own emitter hit is already
    inside D(s_k) with its MMIS weight, so a raw Le never multiplies a single
    path weight (no salt-and-pepper).

    depth=1  → pure cache read at the first hit (splotchy: shared estimate).
    depth=2  → previous behavior: cached direct at v1 + own bounce to the
               indirect cache at v2.
    depth=d  → bounces 1..d-1 use smooth cached-direct plus the path's own
               unbiased transport (no kernel, no multi-hop compounding);
               the multi-hop cache only enters at bounce d, whose energy
               share decays with d. Large d ≈ path tracing with cached
               direct lighting — the zero-cache-indirect diagnostic arm.

    If the path dies before depth d (max_depth cutoff), the tail reads the
    full cache at the last segment. T_{k+1} carries the bounce's own 1/p_ω,
    so no mmis weight applies at split levels.
    """
    last = min(depth, len(path)) - 1   # index of the cache-read segment
    has_direct = getattr(path[0], "direct_in", None) is not None

    if not has_direct:
        # non-vec runs have no direct split — plain product readout down to
        # the cache level (old depth-2 non-vec fallback, generalized)
        R = path[last].radiance_in
        for k in range(last, 0, -1):
            R = path[k].throughput * R
        return R

    seg_last = path[last]
    # Distinguish "stopped here BY DESIGN" from "the path DIED here":
    #   last < len-1  → we hit the fg_depth cap with the path still alive;
    #                   rad_in(s_last) is the legitimate merged estimate of
    #                   all transport beyond this vertex. Read full cache.
    #   last == len-1 → the path terminated (max_depth, or a BSDF pdf=0
    #                   backface/grazing death) WITHOUT reaching a light. Its
    #                   continuation is a genuine ZERO MC sample — the sampled
    #                   ray saw black. Reading the full cache instead injects
    #                   the voxel-average INDIRECT radiance there: pure
    #                   survivorship bias, and it leaks more with a wider
    #                   kernel (O3: tail bucket 3.3e-2→3.85e-2 as c 10→90,
    #                   the entire fg99 overcount). Seed direct-only; the
    #                   dead continuation contributes its honest zero.
    if last == len(path) - 1 and not (seg_last.y.is_light or seg_last.x.is_light):
        R = seg_last.direct_in
    else:
        R = seg_last.radiance_in        # full cache (alive at the depth cap)
    for k in range(last - 1, -1, -1):
        nxt = path[k + 1]
        if nxt.y.is_light or nxt.x.is_light:
            R = path[k].direct_in       # emitter hit counted inside D
        else:
            R = path[k].direct_in + nxt.throughput * R
    return R


def _final_gather(camera_first_segments, height, width, depth=2):
    """
    Collect per-pixel radiance after propagation: pixel = T0 · R_1 with the
    depth-d split readout (see _readout_path). depth=2 reproduces the old
    split gather exactly; depth=1 the old cache readout.
    """
    accum = np.zeros((height * width, 3))
    for pixel_idx, first_seg in enumerate(camera_first_segments):
        if first_seg is None:
            continue
        path = getattr(first_seg, "_path", None)
        if depth <= 1 or path is None:
            val = first_seg.throughput * first_seg.radiance_in * first_seg.mmis_weight
        else:
            val = first_seg.throughput * _readout_path(path, depth)
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

    # (diagnostic prints removed — _diag now computes silently; re-add here
    #  if you need the S/P/M/coverage breakdown for a specific debug session)
    _ = (label, S, P, M, T, n_light_segs, n_nonzero_Le,
         n_nonzero_mmis, n_nonzero_rad, n_nonzero_out)

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
                    # ray_weight is already baked into segments[0].throughput
                    pixel_color += L

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
                               verbose=False, beta = 0.25, add_light_samples = True):
        segment_pool, camera_first_segments = _sample_segments(
            scene, sensor, sampler, height, width, beta, add_light_samples
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
                      num_prop_iterations=4, verbose=True, beta = 0.25, add_light_samples = True):
        """
        BLT render using the non-vectorized MMIS + propagation path.
        Use as a reference to compare against render_vec().
        """
        sensor = scene.sensors()[0]
        accum  = np.zeros((height, width, 3), dtype=np.float32)
        cluster, mmis, propagation, samplers = _make_blt_components(
            scene, add_light_samples, kernel_radius, kernel_weight, num_prop_iterations
        )

        for i in range(n_iterations):
            # print(f"[nonvec] iter {i}/{n_iterations}"
            #       f" — r={propagation.kernel_radius:.3f}")
            sampler = mi.load_dict({'type': 'independent', 'sample_count': 1})
            sampler.seed(i, width * height)
            accum += self._render_iterate_nonvec(
                scene, sensor, sampler, height, width,
                cluster, mmis, propagation, samplers,
                iteration=i, verbose=verbose, beta=beta, add_light_samples=add_light_samples
            )

        return accum / n_iterations

    # ── 3. BLT vectorized ─────────────────────────────────────────────────────

    def _render_iterate_vec(self, scene, sensor, sampler, height, width,
                            cluster, mmis, propagation, samplers, iteration=0,
                            verbose=False, beta = 0.25, add_light_samples = True,
                            final_gather_depth=2, add_bridge_segments=True):
        
        segment_pool, camera_first_segments = _sample_segments(
            scene, sensor, sampler, height, width, beta, add_light_samples,
            add_bridge_segments=add_bridge_segments
        )

        if not segment_pool.paths:
            return np.zeros((height, width, 3))
        cluster.set_segments(segment_pool.paths)
        cluster.cluster(np.random.default_rng(seed=iteration))
        # Build pair cache once — BSDF + PDF computed here, reused in propagation
        pair_cache = build_pair_cache(
            cluster, samplers,
            merge_min_len_factor=propagation.merge_min_len_factor,
        )

        # MMIS weights (vec)
        mmis.compute_all_mmis_weights_vec(cluster, pair_cache)

        # Propagation (vec)
        propagation.iterate_propogation_vec(cluster, pair_cache)

        if verbose:
            _diag("vec", cluster, pair_cache, camera_first_segments)

        bin = _final_gather(camera_first_segments, height, width,
                            depth=final_gather_depth)
        return bin

    def render_vec(self, scene, height=128, width=128, n_iterations=8,
                   kernel_radius=16, kernel_weight=0.67,
                   num_prop_iterations=4, verbose=True, beta = 0.25, add_light_samples = True,
                   cluster_c=30, progressive=True,
                   geom_clamp_factor=None, mmis_clamp_factor=None,
                   geom_clamp_mode="hard", alpha=0.2, row_gain_cap=1.0,
                   merge_min_len_factor=1.0, final_gather_depth=2,
                   add_bridge_segments=True):
        """
        BLT render using the vectorized (numpy) MMIS + propagation path.

        cluster_c: target endpoints per voxel (Cluster.c). Bigger → wider kernel
            → more pairs per receiver → lower variance, more bias. Default 10.
        progressive: enable Knaus-Zwicker progressive kernel shrinkage —
            voxel_size at outer iter n = voxel_size_0 * n^(-α/2). When True,
            this method bumps cluster._iteration each outer iter so the existing
            _compute_progressive_voxel_size() in Cluster actually takes effect
            (it never did before — counter was never incremented anywhere).
            Default True; set False to A/B against the old behavior.
        """
        sensor = scene.sensors()[0]
        accum  = np.zeros((height, width, 3), dtype=np.float32)
        cluster, mmis, propagation, samplers = _make_blt_components(
            scene, add_light_samples, kernel_radius, kernel_weight, num_prop_iterations,
            cluster_c=cluster_c,
            geom_clamp_factor=geom_clamp_factor, mmis_clamp_factor=mmis_clamp_factor,
            geom_clamp_mode=geom_clamp_mode, alpha=alpha, row_gain_cap=row_gain_cap,
            merge_min_len_factor=merge_min_len_factor,
        )

        for i in range(n_iterations):
            if progressive:
                # MUST be set BEFORE _render_iterate_vec runs cluster.cluster(), since
                # _compute_progressive_voxel_size reads self._iteration. Without this,
                # every outer iter re-uses voxel_size_0 and there's no kernel shrinkage.
                cluster._iteration = i
            sampler = mi.load_dict({'type': 'independent', 'sample_count': 1})
            sampler.seed(i, width * height)
            accum += self._render_iterate_vec(
                scene, sensor, sampler, height, width,
                cluster, mmis, propagation, samplers,
                iteration=i, verbose=verbose, beta=beta, add_light_samples=add_light_samples,
                final_gather_depth=final_gather_depth, add_bridge_segments=add_bridge_segments
            )

        return accum / n_iterations

    # ── 4. Save helpers ───────────────────────────────────────────────────────

    def save_render_by_scene_path(self, scene_path, save_file_path="output.png",
                                  width=128, height=128, n_iterations=8,
                                  mode='vec', beta=0.25, add_light_samples = False):
        """
        mode: 'ref'    → simple path tracer (render)
              'nonvec' → BLT non-vectorized
              'vec'    → BLT vectorized  (default)
        """
        mi.set_variant(self.variant)
        #scene = mi.load_file(scene_path)
        scene_dict = mi.cornell_box()
        width = 200
        height = 200
        scene_dict["sensor"]["film"]["width"] = width
        scene_dict["sensor"]["film"]["height"] = height
        scene_dict["integrator"] = {"type": "path", "max_depth": -1}
        scene = mi.load_dict(scene_dict)
        print("add lgith: ", add_light_samples)
        if mode == 'ref':
            image = self.render(scene, width, height, spp=n_iterations)
        elif mode == 'nonvec':
            image = self.render_nonvec(scene, height, width,
                                       n_iterations=n_iterations, beta=beta,
                                       add_light_samples=add_light_samples)
        else:
            image = self.render_vec(scene, height, width,
                                    n_iterations=n_iterations, verbose=False,
                                    beta=beta, add_light_samples=add_light_samples)
        

        # plt.imsave(save_file_path, np.clip(image ** (1 / 2.2), 0, 1))
        save_with_tonemap(save_file_path, image)

        norm = image / np.percentile(image, 99.5)
        img = np.array(image)
        print(f"mean: {img.mean():.4f}")
        print(f"max:  {img.max():.4f}")
        print(f"% pixels > 1.0: {(img > 1.0).mean()*100:.1f}%")

    

    # print(f"\n── cluster diagnostics [{label}] ─────────────────────────────")
    # print(f"  segments: {S}    prop pairs: {P}    avg pairs/seg: {P/S:.1f}")
    # print(f"  zero-incoming:    {n_zero}/{S} ({100*n_zero/max(S,1):.1f}%)  "
    #       f"med={int(np.median(incoming))}  max={int(incoming.max())}")
    # print(f"  camera-first 0:   {n_zero_cam}/{n_cam} ({100*n_zero_cam/max(n_cam,1):.1f}%)"
    #       f"   ← pixels with NO indirect contribution")
    # stats("mmis_w",       mmis_w,      "1 (or π for diffuse interiors)")
    # stats("geom",         geom,        "1 — drops with distance²")
    # stats("Le",           Le,          "0 except light-terminators")
    # stats("radiance_in",  rad_in,      "1 if at light, lower elsewhere")
    # if P > 0:
    #     stats("f_r (BSDF)",    fr,          "1/π ≈ 0.318 for Lambertian")
    #     stats("weighted_fr",   weighted_fr, "1 if energy-conserving")
    #     stats("contrib/pair",  contrib,     "1 if Le=1 reachable")
    #     gain = (P / S) * weighted_fr.mean()
    #     print(f"  estimated gain/hop = (pairs/seg) · mean(weighted_fr) = {gain:.4f}")
    #     print(f"     ▸ gain ≈ 1.0 means energy-conserving; gain < 0.5 means lossy")
    # print("──────────────────────────────────────────────────────────")



    # print(f"\n── voxel {voxel_idx} [{label}] ──")
    # print(f"  segments in voxel: {len(seg_ids)}    pairs receiving here: {mask.sum()}")
    # for name, arr in [("mmis_w", mmis_w), ("geom", geom),
    #                   ("rad_in", rad_in), ("f_r", fr)]:
    #     nz = arr[arr > 0] if (arr > 0).any() else arr
    #     if len(nz):
    #         print(f"  {name:8s}  med={np.median(nz):.3e}  "
    #               f"min={nz.min():.3e}  max={nz.max():.3e}")
            



def trace_firefly(accum_image, camera_first_segments, height, width, top_k=5):
    """Find the brightest pixels and dump exactly what made them bright."""
    lum = accum_image.sum(axis=2)

    # Pixels that look saturated after gamma — these are the "fireflies" you see
    SATURATION_THRESHOLD = 1.0  # ≈ where gamma 2.2 starts clipping
    saturated = lum > SATURATION_THRESHOLD
    print(f"\nSaturated pixels: {saturated.sum()} / {height*width} "
          f"({100*saturated.sum()/(height*width):.2f}%)")
    print(f"  lum distribution: med={np.median(lum):.3e}  "
          f"p90={np.percentile(lum, 90):.3e}  p99={np.percentile(lum, 99):.3e}  "
          f"max={lum.max():.3e}")

    # Sample a few "in-the-wild" saturated pixels (not the light)
    sat_idxs = np.where(saturated.flatten() & (lum.flatten() < 30))[0]
    if len(sat_idxs) == 0:
        print("  no non-light saturated pixels")
        return
    # print(f"\n── 5 random non-light saturated pixels ──")
    # np.random.shuffle(sat_idxs)
    # for idx in sat_idxs[:5]:
    #     y, x = idx // width, idx % width
    #     first_seg = camera_first_segments[idx]
    #     if first_seg is None: continue
    #     print(f"  pixel ({x},{y}) lum={lum[y,x]:.3e}")
    #     print(f"    rad_in={tuple(float(first_seg.radiance_in[k]) for k in range(3))}")
    #     print(f"    mmis_w={first_seg.mmis_weight:.3e}")
    #     print(f"    geom={first_seg.geom_term:.3e}")
    #     print(f"    y.is_light={first_seg.y.is_light}")

def save_with_tonemap(path, image, gamma=2.2, exposure=None, target_grey=0.18):
    """
    Robust tonemap for HDR debug renders.

    Pipeline:
      1. Exposure: scale so the MEDIAN luminance maps to `target_grey` (0.18 ≈
         middle grey). Median is used instead of 99.5%ile because fireflies
         inflate the tail and make percentile-based exposure lie about the
         bulk brightness — that's exactly the failure mode that made our BLT
         output "look dim" even when its mean matched ref.
      2. Reinhard compression: x → x / (1 + x). Bright pixels saturate softly
         instead of clipping, so fireflies don't punch holes in the image and
         don't push neighboring pixels into 1.0.
      3. Gamma.

    Pass `exposure=<float>` to override auto-exposure with a fixed scale —
    useful when comparing renders A/B (same exposure → directly comparable).

    Compare two renders with matched exposure:
        save_with_tonemap("a.png", img_a, exposure=2.0)
        save_with_tonemap("b.png", img_b, exposure=2.0)
    """
    img = np.asarray(image, dtype=np.float64)

    # luminance for exposure decisions only (apply scaling to all channels)
    lum = img.sum(-1) if img.ndim == 3 else img

    if exposure is None:
        med = float(np.median(lum[lum > 0])) if (lum > 0).any() else 0.0
        # scale so median luma maps to `target_grey` (post-Reinhard, post-gamma the
        # bulk lands near middle grey on screen). Reinhard halves at x=1, so we
        # aim a touch above target_grey pre-compression.
        exposure = (target_grey / med) if med > 0 else 1.0

    scaled = img * exposure

    # per-channel Reinhard — preserves hue better than luminance-based variants
    compressed = scaled / (1.0 + scaled)

    out = np.clip(compressed ** (1.0 / gamma), 0.0, 1.0)
    plt.imsave(path, out.astype(np.float32))

    # one-line summary so you can see what exposure was chosen
    print(f"  tonemap: exposure={exposure:.3e}  "
          f"median_lum={float(np.median(lum)):.3e}  "
          f"max_lum={float(lum.max()):.3e}")