"""
R: lamp-scene shadow diagnostic.

Question: BLT loses the cast/background shadows. Is it (a) cross-visibility
LEAK in the merged direct gather (shared light samples bleeding across the
shadow boundary — softens with bigger kernel), or (b) just kernel SCALE
(voxels too big for the shadow features in this scene)?

Renders, at each --c, the BLT direct gather (T0·D(v1), the merged direct)
and compares to PT max_depth=2 (true direct, hard shadows). If the shadow is
crisp in PT-direct but washed out in BLT-direct, and gets WORSE with larger
c, it's the leak — and the fix is own-NEE-for-direct, not a kernel knob.

Saves a PNG per c with [PT direct | BLT direct | diff].

Run:  python R_lamp_shadows.py --width 128 --iters 6 --c 10 30 90
"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import mitsuba as mi

import _setup  # noqa: F401  (variant + sys.path)
from segments.renderer import (_make_blt_components, _sample_segments,
                               _final_gather, _resolve_scene)
from segments.pair_cache import build_pair_cache


def tonemap(img):
    lum = img.sum(-1)
    med = float(np.median(lum[lum > 0])) if (lum > 0).any() else 1.0
    e = 0.18 / med if med > 0 else 1.0
    return np.clip((img * e / (1 + img * e)) ** (1 / 2.2), 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--ref-spp", type=int, default=128)
    ap.add_argument("--c", type=int, nargs="+", default=[10, 30, 90])
    ap.add_argument("--scene", default="lamp",
                    help="sample NAME (resolved under ray-maps/samples/<name>/"
                         "scene.xml) or a direct path to a scene .xml — same "
                         "resolution rule as main.py / save_render_by_scene_path")
    ap.add_argument("--full", action="store_true",
                    help="render the FULL image (direct+indirect, fg2) vs full "
                         "PT (max_depth=-1) instead of direct-only vs PT(2). "
                         "This is the one that shows whether INDIRECT loses the "
                         "background shadows; the default direct-only mode only "
                         "tests the direct term.")
    ap.add_argument("--prop-iters", type=int, default=8)
    args = ap.parse_args()
    W = H = args.width
    full = args.full

    scene_path = _resolve_scene(args.scene)
    print(f"scene: {scene_path}  [{'FULL direct+indirect' if full else 'DIRECT-only'}]")

    # ── PT reference: full transport (max_depth=-1) or direct (max_depth=2) ──
    scene = mi.load_file(scene_path, resx=W, resy=H)
    ref_depth = -1 if full else 2
    ref_scene = mi.load_file(scene_path, resx=W, resy=H,
                             integrator="path", max_depth=ref_depth)
    print(f"PT reference (max_depth={ref_depth}, {args.ref_spp} spp)...")
    ref = np.array(mi.render(ref_scene, spp=args.ref_spp))[..., :3].astype(np.float64)

    sensor = scene.sensors()[0]
    for c in args.c:
        cluster, mmis, prop, samplers = _make_blt_components(
            scene, add_light_samples=False,
            num_prop_iterations=(args.prop_iters if full else 1), cluster_c=c)
        acc = np.zeros((H, W, 3))
        for i in range(args.iters):
            smp = mi.load_dict({"type": "independent", "sample_count": 1})
            smp.seed(i, W * H)
            pool, cam = _sample_segments(scene, sensor, smp, H, W, beta=0.0,
                                         add_light_samples=False)
            cluster.set_segments(pool.paths)
            cluster.cluster(np.random.default_rng(i))
            pc = build_pair_cache(cluster, samplers,
                                  merge_min_len_factor=prop.merge_min_len_factor)
            mmis.compute_all_mmis_weights_vec(cluster, pc)
            prop.iterate_propogation_vec(cluster, pc)
            if full:
                # complete image: direct + indirect via the fg2 split readout
                acc += _final_gather(cam, H, W, depth=2)
            else:
                # direct-only: T0·D(v1), the merged direct gather
                flat = np.zeros((H * W, 3))
                for px, fs in enumerate(cam):
                    if fs is None:
                        continue
                    if fs.y.is_light or fs.x.is_light:
                        v = fs.throughput * fs.radiance_in
                    else:
                        D = getattr(fs, "direct_in", None)
                        if D is None:
                            continue
                        v = fs.throughput * D
                    flat[px] = [float(v.x), float(v.y), float(v.z)]
                acc += flat.reshape(H, W, 3)
        blt = acc / args.iters
        mr = blt.sum(-1).mean() / max(ref.sum(-1).mean(), 1e-12)
        kind = "full" if full else "direct"
        print(f"  c={c:>3}: voxel={float(cluster.voxel_size):.3e}  "
              f"{kind} mean_ratio={mr:.4f}")

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        d = blt.sum(-1) - ref.sum(-1)
        ptlabel = "PT full (d=-1)" if full else "PT direct (d=2)"
        bltlabel = f"BLT {'full' if full else 'direct'} (c={c})"
        for a, im, t in zip(ax, [ref, blt, None],
                            [ptlabel, bltlabel, "diff"]):
            if t == "diff":
                m = a.imshow(d, cmap="RdBu_r",
                             vmin=-np.abs(d).max(), vmax=np.abs(d).max())
                plt.colorbar(m, ax=a, fraction=0.046)
            else:
                a.imshow(tonemap(im))
            a.set_title(t); a.axis("off")
        out = os.path.join(os.path.dirname(__file__), f"R_lamp_{'full' if full else 'direct'}_c{c}.png")
        plt.tight_layout(); plt.savefig(out, dpi=110); plt.close()
        print(f"    saved {out}")


if __name__ == "__main__":
    main()
