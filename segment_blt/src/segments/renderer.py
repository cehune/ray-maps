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

def _make_blt_components(scene, add_light_samples, kernel_radius=16, kernel_weight=0.67, num_prop_iterations=4, cluster_c=10, apply_kernel_correction=False, geom_clamp_factor=None, mmis_clamp_factor=None):
    mmis        = MMIS()
    mmis.mmis_clamp_factor = mmis_clamp_factor
    propagation = Propagation(num_prop_iterations=num_prop_iterations,
                              apply_kernel_correction=apply_kernel_correction,
                              geom_clamp_factor=geom_clamp_factor)
    cluster = Cluster(c=cluster_c)
    cluster.set_scene_aabb(scene.bbox())
    if add_light_samples:
        samplers = Sampler.build_registry(SequentialSampler(), LightSampler())
    else:
        samplers = Sampler.build_registry(SequentialSampler())
    return cluster, mmis, propagation, samplers


def _sample_segments(scene, sensor, sampler, height, width, beta = 0.25, add_light_samples = True):
    """
    Sample one camera path per pixel.
    Returns (segment_pool, camera_first_segments).
    camera_first_segments[pixel_idx] is segments[0] for that pixel, or None.
    """
    segment_pool = SegmentPoolV2()
    camera_first_segments = []
    n_pixels = height * width

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

    # ── Light paths (β · n_pixels total) ──────────────────────────────
    if add_light_samples:
        n_light_paths = int(beta * n_pixels)
        light_paths_added = 0
        for _ in range(n_light_paths):
            light_segments = sample_light_path(scene, sampler)
            if light_segments:
                segment_pool.add_path(light_segments)
                light_paths_added += 1

        print(f"  sampled {len(camera_first_segments)} camera paths, "
            f"{light_paths_added}/{n_light_paths} light paths "
            f"(β={beta})")

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
                            verbose=False, beta = 0.25, add_light_samples = True):
        
        t0 = time.time()
        segment_pool, camera_first_segments = _sample_segments(
            scene, sensor, sampler, height, width, beta, add_light_samples
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

        if iteration == 0:
            diagnose_cluster_factors(cluster, samplers, pair_cache, label=f"iter{iteration}")
            n_voxels = len(cluster.cluster_ranges)
            for vi in [0, n_voxels // 2, n_voxels - 1]:
                diagnose_voxel_cluster(cluster, pair_cache, vi, label=f"iter{iteration}")

        if verbose:
            _diag("vec", cluster, pair_cache, camera_first_segments)

        t0 = time.time()
        bin = _final_gather(camera_first_segments, height, width)

        trace_firefly(bin, camera_first_segments, height, width, 5)
        print(f"sampling: {time.time()-t0:.1f}s")
        return bin

    def render_vec(self, scene, height=128, width=128, n_iterations=8,
                   kernel_radius=16, kernel_weight=0.67,
                   num_prop_iterations=4, verbose=True, beta = 0.25, add_light_samples = True,
                   cluster_c=10, apply_kernel_correction=False, progressive=True,
                   geom_clamp_factor=None, mmis_clamp_factor=None):
        """
        BLT render using the vectorized (numpy) MMIS + propagation path.

        cluster_c: target endpoints per voxel (Cluster.c). Bigger → wider kernel
            → more pairs per receiver → lower variance, more bias. Default 10.
        apply_kernel_correction: DIAGNOSTIC — multiply per-pair weight by 1/K of
            the receiver's cluster. Tests whether the BLT paper's 1/K factor is
            genuinely missing from the estimator. Default False.
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
            cluster_c=cluster_c, apply_kernel_correction=apply_kernel_correction,
            geom_clamp_factor=geom_clamp_factor, mmis_clamp_factor=mmis_clamp_factor,
        )

        for i in range(n_iterations):
            if progressive:
                # MUST be set BEFORE _render_iterate_vec runs cluster.cluster(), since
                # _compute_progressive_voxel_size reads self._iteration. Without this,
                # every outer iter re-uses voxel_size_0 and there's no kernel shrinkage.
                cluster._iteration = i
            print(f"[vec] iter {i}/{n_iterations}"
                  f"{f'   voxel_size={cluster.voxel_size:.3e}' if i > 0 else ''}")
            sampler = mi.load_dict({'type': 'independent', 'sample_count': 1})
            sampler.seed(i, width * height)
            accum += self._render_iterate_vec(
                scene, sensor, sampler, height, width,
                cluster, mmis, propagation, samplers,
                iteration=i, verbose=verbose, beta=beta, add_light_samples=add_light_samples
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
        # report_bsdf_types(scene)
        # force_white_lambertian(scene)
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

    # ── Legacy entry point (kept for back-compat) ─────────────────────────────

    def render_iterate_loop(self, scene, height=128, width=128, n_iterations=8):
        """Alias for render_vec() with default BLT parameters."""
        return self.render_vec(scene, height, width, n_iterations=n_iterations)
    
def diagnose_cluster_factors(cluster, samplers, pair_cache, *, label=""):
    """
    Print magnitude statistics of every factor in the propagation product:
        contrib_per_pair = mmis_w · f_r · geom · rad_in
    Compares the medians/means to what you'd see in a unit-conserving
    cornell-box-style setup (everything ~O(1)).
    """
    import numpy as np

    segs = cluster.segments
    S = len(segs)
    if S == 0:
        print(f"[{label}] empty cluster — skipped")
        return

    mmis_w = np.array([float(s.mmis_weight) for s in segs])
    geom   = np.array([float(s.geom_term) for s in segs])
    rad_in = np.array([float(s.radiance_in[0]) for s in segs])
    Le     = np.array([float(s.Le[0]) for s in segs])

    P = len(pair_cache.prop_i)
    if P > 0:
        fr = pair_cache.prop_fr[:, 0]                  # red channel
        j  = pair_cache.prop_j
        weighted_fr = fr * mmis_w[j] * geom[j]
        contrib = weighted_fr * rad_in[j]
    else:
        fr = weighted_fr = contrib = np.array([0.0])

    def stats(name, arr, expected_order=None):
        nz = arr[arr > 0] if (arr > 0).any() else arr
        med  = np.median(nz)
        mn   = nz.min()
        mx   = nz.max()
        hint = f"   expected ~O({expected_order})" if expected_order else ""
        print(f"  {name:18s}  med={med:.3e}  min={mn:.3e}  max={mx:.3e}  n={len(nz)}{hint}")

    # zero-coverage diagnostic: how many segments receive NO propagation pairs?
    # camera-first segments with zero coverage = pixels that get only direct hits.
    if P > 0:
        incoming = np.bincount(pair_cache.prop_i, minlength=S)
    else:
        incoming = np.zeros(S, dtype=np.int64)
    is_cam_first = np.array([s.x.is_camera for s in segs], dtype=bool)
    n_zero       = int((incoming == 0).sum())
    n_cam        = int(is_cam_first.sum())
    n_zero_cam   = int(((incoming == 0) & is_cam_first).sum())

    print(f"\n── cluster diagnostics [{label}] ─────────────────────────────")
    print(f"  segments: {S}    prop pairs: {P}    avg pairs/seg: {P/S:.1f}")
    print(f"  zero-incoming:    {n_zero}/{S} ({100*n_zero/max(S,1):.1f}%)  "
          f"med={int(np.median(incoming))}  max={int(incoming.max())}")
    print(f"  camera-first 0:   {n_zero_cam}/{n_cam} ({100*n_zero_cam/max(n_cam,1):.1f}%)"
          f"   ← pixels with NO indirect contribution")
    stats("mmis_w",       mmis_w,      "1 (or π for diffuse interiors)")
    stats("geom",         geom,        "1 — drops with distance²")
    stats("Le",           Le,          "0 except light-terminators")
    stats("radiance_in",  rad_in,      "1 if at light, lower elsewhere")
    if P > 0:
        stats("f_r (BSDF)",    fr,          "1/π ≈ 0.318 for Lambertian")
        stats("weighted_fr",   weighted_fr, "1 if energy-conserving")
        stats("contrib/pair",  contrib,     "1 if Le=1 reachable")
        gain = (P / S) * weighted_fr.mean()
        print(f"  estimated gain/hop = (pairs/seg) · mean(weighted_fr) = {gain:.4f}")
        print(f"     ▸ gain ≈ 1.0 means energy-conserving; gain < 0.5 means lossy")
    print("──────────────────────────────────────────────────────────")


def diagnose_voxel_cluster(cluster, pair_cache, voxel_idx, *, label=""):
    """
    Print the same factor stats restricted to one voxel-cluster (one entry
    of cluster.cluster_ranges).  Useful for comparing a 'central' voxel vs
    a sparse corner voxel — the latter is often the source of grid artifacts.
    """
    import numpy as np

    start, end = cluster.cluster_ranges[voxel_idx]
    flat = cluster.sorted_indices[start:end]
    meta = cluster.endpoint_metadata[flat]
    seg_ids = np.unique(meta[:, 0])   # segments touching this voxel

    if len(seg_ids) == 0:
        print(f"[{label}] voxel {voxel_idx} empty"); return

    mmis_w = np.array([float(cluster.segments[i].mmis_weight) for i in seg_ids])
    geom   = np.array([float(cluster.segments[i].geom_term)   for i in seg_ids])
    rad_in = np.array([float(cluster.segments[i].radiance_in[0]) for i in seg_ids])

    # filter pair_cache to pairs whose receiver (prop_i) is in this voxel
    seg_set = set(seg_ids.tolist())
    mask = np.array([int(i) in seg_set for i in pair_cache.prop_i], dtype=bool)
    fr = pair_cache.prop_fr[mask, 0] if mask.any() else np.array([0.0])

    # print(f"\n── voxel {voxel_idx} [{label}] ──")
    # print(f"  segments in voxel: {len(seg_ids)}    pairs receiving here: {mask.sum()}")
    # for name, arr in [("mmis_w", mmis_w), ("geom", geom),
    #                   ("rad_in", rad_in), ("f_r", fr)]:
    #     nz = arr[arr > 0] if (arr > 0).any() else arr
    #     if len(nz):
    #         print(f"  {name:8s}  med={np.median(nz):.3e}  "
    #               f"min={nz.min():.3e}  max={nz.max():.3e}")
            

def force_white_lambertian(scene):
    """Replace every diffuse BSDF in the scene with albedo=1 Lambertian."""
    white = mi.load_dict({
        'type': 'diffuse',
        'reflectance': {'type': 'rgb', 'value': [1.0, 1.0, 1.0]},
    })
    for shape in scene.shapes():
        bsdf = shape.bsdf()
        if bsdf is not None:
            # Mitsuba doesn't expose a `set_bsdf` directly on shapes, so we
            # mutate the BSDF's reflectance in place via params traversal.
            params = mi.traverse(bsdf)
            for k in params.keys():
                if 'reflectance' in k.lower() and 'value' in k.lower():
                    params[k] = mi.ScalarColor3f(1.0, 1.0, 1.0)
            params.update()

def report_bsdf_types(scene):
    print("\nBSDF inventory:")
    from mitsuba import BSDFFlags as F
    for shape in scene.shapes():
        bsdf = shape.bsdf()
        if bsdf is None:
            continue
        flags = bsdf.flags()
        is_delta = bool(flags & (F.DeltaReflection | F.DeltaTransmission))
        is_glossy = bool(flags & (F.GlossyReflection | F.GlossyTransmission))
        is_diffuse = bool(flags & (F.DiffuseReflection | F.DiffuseTransmission))
        category = []
        if is_diffuse: category.append("diffuse")
        if is_glossy: category.append("glossy")
        if is_delta: category.append("SPECULAR — not filtered by BLT")
        print(f"  {shape.id() or shape}: {', '.join(category) or 'unknown'}")

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
    print(f"\n── 5 random non-light saturated pixels ──")
    np.random.shuffle(sat_idxs)
    for idx in sat_idxs[:5]:
        y, x = idx // width, idx % width
        first_seg = camera_first_segments[idx]
        if first_seg is None: continue
        print(f"  pixel ({x},{y}) lum={lum[y,x]:.3e}")
        print(f"    rad_in={tuple(float(first_seg.radiance_in[k]) for k in range(3))}")
        print(f"    mmis_w={first_seg.mmis_weight:.3e}")
        print(f"    geom={first_seg.geom_term:.3e}")
        print(f"    y.is_light={first_seg.y.is_light}")

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