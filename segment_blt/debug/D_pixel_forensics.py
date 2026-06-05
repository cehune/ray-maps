"""
D. Single-pixel forensics.

Pick one pixel, walk the entire pipeline by hand, dump:
  • the camera segment's throughput / rad_in / mmis_w / final contribution
  • the TOP-K propagation pairs flowing INTO this pixel's first segment,
    sorted by luminance contribution
  • for each contributor j: its position, whether y is light, the BSDF
    value, geom, mmis_w, rad_in at j

What to look for:
  • If no incoming pairs → kernel coverage problem at this point.
  • If top contributors have rad_in ≈ 0 → indirect chain didn't reach them.
  • If top contributors come from segments on the wrong surface (wall lit
    from a floor segment 2m away with absurd geom) → cluster keys are
    leaking across surface boundaries.
  • If lum_contrib is dominated by ONE huge pair → that pair is your firefly.

CLI:
  python D_pixel_forensics.py --pixel 100 100 --top 10 --prop 4 --width 200
"""
from _setup import cornell_scene

import argparse
import numpy as np
import mitsuba as mi
from segments.renderer import _make_blt_components, _sample_segments
from segments.pair_cache import build_pair_cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pixel", nargs=2, type=int, default=[100, 100],
                    metavar=("X", "Y"))
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--width", type=int, default=200)
    ap.add_argument("--prop", type=int, default=4)
    args = ap.parse_args()

    W = H = args.width
    px, py = args.pixel
    if not (0 <= px < W and 0 <= py < H):
        raise SystemExit(f"pixel ({px},{py}) out of bounds for {W}×{H}")
    target_pixel = py * W + px
    print(f"target pixel = ({px},{py})  flat_idx={target_pixel}")

    scene = cornell_scene(W, H)
    cluster, mmis, propagation, samplers = _make_blt_components(
        scene, add_light_samples=False, num_prop_iterations=args.prop,
    )
    sensor = scene.sensors()[0]
    sampler = mi.load_dict({"type": "independent", "sample_count": 1})
    sampler.seed(0, W * H)

    print("sampling segments ...")
    segment_pool, camera_first_segments = _sample_segments(
        scene, sensor, sampler, H, W, beta=0.1, add_light_samples=False,
    )
    print("clustering ...")
    cluster.set_segments(segment_pool.paths)
    cluster.cluster(np.random.default_rng(seed=0))
    print("building pair cache ...")
    pair_cache = build_pair_cache(cluster, samplers)
    print("computing mmis weights ...")
    mmis.compute_all_mmis_weights_vec(cluster, pair_cache)
    print("propagating ...")
    propagation.iterate_propogation_vec(cluster, pair_cache)

    first_seg = camera_first_segments[target_pixel]
    if first_seg is None:
        print(f"\nfirst_seg is None — ray escaped at ({px},{py}). Pick another pixel.")
        return

    # locate first_seg in cluster.segments (identity)
    target_idx = next((i for i, s in enumerate(cluster.segments) if s is first_seg), None)
    if target_idx is None:
        print("BUG: first_seg not in cluster.segments")
        return

    p = first_seg.y.p
    n = first_seg.y.n
    tp = first_seg.throughput
    rin = first_seg.radiance_in
    val = tp * rin * first_seg.mmis_weight

    print(f"\n── target segment (cluster idx={target_idx}) ─────────────")
    print(f"  y position    = ({float(p.x):+.3f}, {float(p.y):+.3f}, {float(p.z):+.3f})")
    print(f"  y normal      = ({float(n.x):+.3f}, {float(n.y):+.3f}, {float(n.z):+.3f})")
    print(f"  y.is_light    = {first_seg.y.is_light}")
    print(f"  throughput    = ({float(tp[0]):.4e}, {float(tp[1]):.4e}, {float(tp[2]):.4e})")
    print(f"  radiance_in   = ({float(rin[0]):.4e}, {float(rin[1]):.4e}, {float(rin[2]):.4e})")
    print(f"  mmis_weight   = {first_seg.mmis_weight:.4e}")
    print(f"  final contrib = ({float(val[0]):.4e}, {float(val[1]):.4e}, {float(val[2]):.4e})")
    lum_pixel = 0.2126*float(val[0]) + 0.7152*float(val[1]) + 0.0722*float(val[2])
    print(f"  pixel luma    = {lum_pixel:.4e}")

    mask = pair_cache.prop_i == target_idx
    n_pairs = int(mask.sum())
    print(f"\nincoming pairs into target seg: {n_pairs}")
    if n_pairs == 0:
        print("  → NO PAIRS. This pixel's rad_in is 0 by construction.")
        print("    Either: cluster at y is too small (<=5 endpoints → vec skips),")
        print("    or no other segments have x near this y.")
        return

    js = pair_cache.prop_j[mask]
    frs = pair_cache.prop_fr[mask]
    segs = cluster.segments
    mmis_arr = np.array([segs[int(j)].mmis_weight for j in js], dtype=np.float64)
    geom_arr = np.array([float(segs[int(j)].geom_term) for j in js], dtype=np.float64)
    rad_arr = np.array([[float(segs[int(j)].radiance_in[c]) for c in range(3)] for j in js])
    is_light_arr = np.array([bool(segs[int(j)].y.is_light or segs[int(j)].x.is_light) for j in js])

    weighted = frs * (mmis_arr * geom_arr)[:, None]
    contrib = weighted * rad_arr
    luma_per_pair = (contrib * np.array([0.2126, 0.7152, 0.0722])).sum(-1)

    order = np.argsort(-luma_per_pair)[:args.top]
    print(f"\ntop {args.top} contributors into this pixel "
          f"(sum_top/sum_all = {luma_per_pair[order].sum() / max(luma_per_pair.sum(), 1e-12):.3f}):")
    hdr = (f"{'rk':>3} {'j':>6} {'luma':>10} {'fr_R':>9} {'mmis':>9} "
           f"{'geom':>9} {'radR':>9} {'pos_j':>22} {'lit':>4}")
    print(hdr)
    print("-" * len(hdr))
    for r, k in enumerate(order):
        j = int(js[k])
        sj = segs[j]
        sx = sj.x.p
        pos = f"({float(sx.x):+.2f},{float(sx.y):+.2f},{float(sx.z):+.2f})"
        print(f"{r:>3} {j:>6} {luma_per_pair[k]:>10.3e} {frs[k,0]:>9.3e} "
              f"{mmis_arr[k]:>9.3e} {geom_arr[k]:>9.3e} {rad_arr[k,0]:>9.3e} "
              f"{pos:>22} {('Y' if is_light_arr[k] else '.'):>4}")

    print(f"\nstats over all {n_pairs} incoming pairs:")
    print(f"  fr_R   med={np.median(frs[:,0]):.3e}  max={frs[:,0].max():.3e}")
    print(f"  mmis   med={np.median(mmis_arr):.3e}  max={mmis_arr.max():.3e}")
    print(f"  geom   med={np.median(geom_arr):.3e}  max={geom_arr.max():.3e}")
    print(f"  rad_R  med={np.median(rad_arr[:,0]):.3e}  max={rad_arr[:,0].max():.3e}")
    print(f"  pairs with rad_R==0: {(rad_arr[:,0]==0).sum()}/{n_pairs} "
          f"({100*(rad_arr[:,0]==0).mean():.1f}%)")


if __name__ == "__main__":
    main()
