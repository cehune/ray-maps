"""
O2: single-gather calibration at the SECOND vertex.

The contradiction this resolves (all at the same width):
  O  : one gather at v1 (first hits)            -> ~0.96-0.99  (calibrated-ish)
  Q2 : same-pool reuse compounding              -> ~+10%/hop   (real, not 1.4)
  P  : first ADDED bounce vs PT                 -> ~1.39       (a single hop!)

A single application cannot be both 0.99 and 1.39 unless the gather's bias
depends on WHERE it is evaluated. v1 sits on open walls/floor; v2 (one BSDF
bounce later) concentrates in corners, under boxes, on the ceiling — exactly
where the hot structure shows in the G diff map. This script measures the
same one-gather quantity as O, but at v2:

  BLT:  pixel = T0 · T1 · D(v2)         (zero if the path dies at v1 or
        seg2 is light-terminal — that energy belongs to direct)
  PT :  reference(max_depth=3) - reference(max_depth=2)
        (exactly the 1-bounce-indirect estimand; both refs cached)

Verdict:
  ratio ~= O's v1 ratio        -> the gather is location-neutral; P's heat
                                  must come from elsewhere (readout depth
                                  semantics, compounding at higher N).
  ratio ~= P's marginal (1.3+) -> gather bias is receiver-geometry
                                  conditional: kernel pairing in concave
                                  regions is the target. Expect the
                                  per-pixel ratio map to glow in corners.

Run:  python O2_v2_gather_calibration.py --width 64 --iters 16
"""
from _setup import cornell_scene
from _cache import RenderCache, pt_reference

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import mitsuba as mi

from segments.renderer import _make_blt_components, _sample_segments
from segments.pair_cache import build_pair_cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=16)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--ref-spp", type=int, default=512)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    W = H = args.width

    # ── ground truth: 1-bounce indirect = PT(3) − PT(2), both cached ─────────
    ref3 = pt_reference(W, H, max_depth=3, spp=args.ref_spp, refresh=args.refresh)
    ref2 = pt_reference(W, H, max_depth=2, spp=args.ref_spp, refresh=args.refresh)
    ref = np.maximum(ref3 - ref2, 0.0)
    print(f"  ref 1-bounce-indirect mean = {ref.sum(-1).mean():.4e}")

    # ── BLT: injection + ONE gather, read at v2 through the path's own T1 ────
    scene = cornell_scene(W, H)
    sensor = scene.sensors()[0]
    cluster, mmis, propagation, samplers = _make_blt_components(
        scene, add_light_samples=False, num_prop_iterations=1,
        cluster_c=30, geom_clamp_factor=None, mmis_clamp_factor=None,
    )

    cache = RenderCache(
        params=dict(W=W, H=H, c=30, N=1, beta=0.1, light=False, nee=True,
                    mode="O2_v2_gather"),
        tag="O2_v2", enabled=not args.refresh)

    accum = np.zeros((H, W, 3), dtype=np.float64)
    for i in range(args.iters):
        img = cache.load_iter(i)
        if img is None:
            sampler = mi.load_dict({"type": "independent", "sample_count": 1})
            sampler.seed(i, W * H)
            pool, cam_first = _sample_segments(scene, sensor, sampler, H, W,
                                               beta=0.1, add_light_samples=False)
            cluster.set_segments(pool.paths)
            cluster.cluster(np.random.default_rng(seed=i))
            pc = build_pair_cache(
                cluster, samplers,
                merge_min_len_factor=propagation.merge_min_len_factor)
            mmis.compute_all_mmis_weights_vec(cluster, pc)
            propagation.iterate_propogation_vec(cluster, pc)   # fills direct_in

            flat = np.zeros((H * W, 3))
            for px, fs in enumerate(cam_first):
                if fs is None:
                    continue
                path = getattr(fs, "_path", None)
                if path is None or len(path) < 2:
                    continue
                s2 = path[1]
                if s2.y.is_light or s2.x.is_light:
                    continue                       # direct energy, not ours
                D2 = getattr(s2, "direct_in", None)
                if D2 is None:
                    continue
                v = fs.throughput * (s2.throughput * D2)
                flat[px] = [float(v.x), float(v.y), float(v.z)]
            img = flat.reshape(H, W, 3)
            cache.save_iter(i, img, stats={"mean": float(img.mean())})
        accum += img
        running = accum / (i + 1)
        mr = running.sum(-1).mean() / max(ref.sum(-1).mean(), 1e-12)
        print(f"iter {i+1:3d}/{args.iters}   v2-gather mean_ratio = {mr:.4f}")

    blt = accum / args.iters
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ratio = np.where(ref.sum(-1) > 1e-6, blt.sum(-1) / np.maximum(ref.sum(-1), 1e-12), np.nan)
    for a, im, t in zip(
            ax, [ref, blt, None],
            ["PT 1-bounce indirect (d3−d2)", "BLT T0·T1·D(v2)", "BLT/PT ratio"]):
        if t.endswith("ratio"):
            m = a.imshow(ratio, cmap="RdBu_r", vmin=0.5, vmax=1.5)
            plt.colorbar(m, ax=a, fraction=0.046)
        else:
            a.imshow(np.clip((im / max(float(np.percentile(im, 99)), 1e-9)) ** (1 / 2.2), 0, 1))
        a.set_title(t); a.axis("off")
    out = os.path.join(os.path.dirname(__file__), "O2_v2_gather.png")
    plt.tight_layout(); plt.savefig(out, dpi=110); plt.close()
    print(f"\nsaved: {out}")
    print("ratio map: red regions = where the single gather runs hot — if it "
          "glows in corners/ceiling, the heat is receiver-geometry conditional.")


if __name__ == "__main__":
    main()
