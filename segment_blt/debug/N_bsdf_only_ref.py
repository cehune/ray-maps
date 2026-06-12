"""
N. BSDF-only baseline — the verdict experiment for the ~0.967 mean ratio.

Renders the scene with plain unidirectional BSDF-only path tracing
(evaluate_path_base — the SAME sampling strategy BLT is built on, but with
no clustering, no MMIS, no propagation, no caps), then scores it against
the NEE-converged reference exactly like G does.

Interpretation:
  * BSDF-only ALSO sits at ~0.96-0.97 with a dark band near the light
        -> the deficit is the unidirectional sampling strategy itself
           (heavy-tailed near small emitters), NOT the BLT estimator.
           Cure = reintroduce the light-segment technique.
  * BSDF-only converges to ~1.0 while BLT sits at 0.967
        -> the deficit IS in the BLT pipeline; bring the cap/exclusion
           prints and we keep hunting.

Run:  python N_bsdf_only_ref.py --spp 48
"""
from _setup import cornell_scene

import argparse
import os
import numpy as np
import mitsuba as mi
from _cache import RenderCache
from segments.renderer import Renderer
from segments.segment_gen_sampler import sample_camera_path, evaluate_path_base

DEFAULT_REF = "/Users/mckalechung/Documents/proj/ray-maps/baseline/spp_renders-128x128-d-1/reference.exr"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=DEFAULT_REF)
    ap.add_argument("--spp", type=int, default=48,
                    help="match this to your G --max-iter for an equal-effort comparison")
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--height", type=int, default=128)
    args = ap.parse_args()

    W, H = args.width, args.height
    ref = np.array(mi.Bitmap(args.ref))[..., :3].astype(np.float64)
    if ref.shape[:2] != (H, W):
        raise SystemExit(f"ref is {ref.shape[:2]}, requested {(H, W)}")

    scene = cornell_scene(W, H)
    sensor = scene.sensors()[0]

    cache = RenderCache(params=dict(W=W, H=H, mode="N_bsdf_only"),
                        tag="N_bsdf_only")
    accum = np.zeros((H, W, 3), dtype=np.float64)
    for s in range(args.spp):
        cached = cache.load_iter(s)
        if cached is not None:
            accum += cached
            running = accum / (s + 1)
            mr = running.sum(-1).mean() / max(ref.sum(-1).mean(), 1e-12)
            rmse = float(np.sqrt(np.mean((running - ref) ** 2)))
            print(f"spp {s+1:3d}/{args.spp}   mean_ratio={mr:.4f}   RMSE={rmse:.4e}")
            continue
        sampler = mi.load_dict({"type": "independent", "sample_count": 1})
        sampler.seed(s, W * H)
        img = np.zeros((H, W, 3), dtype=np.float64)
        for y in range(H):
            for x in range(W):
                pos = mi.Point2f(x + sampler.next_1d(), y + sampler.next_1d())
                film = mi.Point2f(pos.x / W, pos.y / H)
                ray, rw = sensor.sample_ray(
                    time=0.0, sample1=sampler.next_1d(), sample2=film,
                    sample3=mi.Point2f(sampler.next_1d(), sampler.next_1d()))
                si = scene.ray_intersect(ray)
                segs = sample_camera_path(scene, sampler, ray, rw, si)
                L = evaluate_path_base(segs)
                img[y, x] = [float(L[0]), float(L[1]), float(L[2])]
        cache.save_iter(s, img, stats={"mean": float(img.mean())})
        accum += img
        running = accum / (s + 1)
        mr = running.sum(-1).mean() / max(ref.sum(-1).mean(), 1e-12)
        rmse = float(np.sqrt(np.mean((running - ref) ** 2)))
        print(f"spp {s+1:3d}/{args.spp}   mean_ratio={mr:.4f}   RMSE={rmse:.4e}")

    out = os.path.join(os.path.dirname(__file__), f"N_bsdf_only_{args.spp}spp.exr")
    mi.Bitmap(np.ascontiguousarray((accum / args.spp).astype(np.float32))).write(out)
    print(f"\nsaved {out}")
    print("compare visually:  python F_ab_vs_reference_all.py --blt-path " + out)


if __name__ == "__main__":
    main()
