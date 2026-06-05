"""
Read an EXR (or PNG/HDR/anything mi.Bitmap accepts) and print luminance
stats in the same shape as renderer.trace_firefly / B_bounce_ladder, so
you can directly compare to your BLT output's mean_lum / p99.

Usage:
    python read_exr.py <path/to/image.exr>
    python read_exr.py path/to/reference.exr --normalize-by 18.4
"""
from _setup import cornell_scene  # noqa: F401  (just for the mi.set_variant side effect)

import argparse
import numpy as np
import mitsuba as mi


def main():
    ap = argparse.ArgumentParser()
    default_path = "/Users/celine/Documents/projects/ray-maps/baseline/spp_renders-200x200-d-1/reference.exr"
    ap.add_argument("--path", default=default_path, help="EXR / HDR / PNG / etc. (any mi.Bitmap format)")
    ap.add_argument("--normalize-by", type=float, default=None,
                    help="optional divisor applied to all pixels before stats")
    args = ap.parse_args()

    img = np.array(mi.Bitmap(args.path))
    print(f"loaded: {args.path}")
    print(f"  shape = {img.shape}   dtype = {img.dtype}")

    # strip alpha if present
    if img.ndim == 3 and img.shape[-1] >= 3:
        rgb = img[..., :3].astype(np.float64)
    else:
        raise SystemExit(f"unexpected image shape {img.shape}; need HxWx{{3,4}}")

    if args.normalize_by is not None:
        rgb = rgb / args.normalize_by
        print(f"  (normalized by {args.normalize_by})")

    # Σ-RGB luminance, matches renderer.trace_firefly's `lum = image.sum(axis=2)`
    lum_sum = rgb.sum(-1)
    # perceptual luma (BT.709), for cross-check
    lum_per = (rgb * np.array([0.2126, 0.7152, 0.0722])).sum(-1)

    def stats(name, arr):
        print(f"  {name:14s} mean={arr.mean():.4e}  med={np.median(arr):.4e}  "
              f"p90={np.percentile(arr, 90):.4e}  p99={np.percentile(arr, 99):.4e}  "
              f"max={arr.max():.4e}")

    print("\n── stats ──")
    stats("Σ-RGB lum",      lum_sum)
    stats("perceptual luma", lum_per)
    for c, name in enumerate("RGB"):
        stats(f"{name} channel",      rgb[..., c])

    nz = (lum_sum > 0).mean()
    over1 = (lum_sum > 1.0).mean()
    print(f"\n  nonzero pixels: {100*nz:.2f}%")
    print(f"  pixels >1.0   : {100*over1:.2f}%  (would clip in default tonemap)")


if __name__ == "__main__":
    main()
