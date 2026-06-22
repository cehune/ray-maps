"""
Generate a reference EXR at any resolution.

Mirrors baseline/baseline.py but parameterized — saves to a path that
matches the existing convention so the other debug scripts can find it:

    /Users/celine/Documents/projects/ray-maps/baseline/spp_renders-WxH-d-1/reference.exr

Usage:
    # default: 400×400, 4096 spp (a few minutes on a modern CPU)
    python debug/make_reference.py

    # higher res:
    python debug/make_reference.py --width 512 --spp 4096

    # quick test ref at low spp (don't use for real A/B):
    python debug/make_reference.py --width 256 --spp 512

    # overwrite an existing one:
    python debug/make_reference.py --width 400 --force
"""
from _setup import cornell_scene

import argparse
import os
import time
import numpy as np
import mitsuba as mi


DEFAULT_BASE = "/Users/celine/Documents/projects/ray-maps/baseline"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=400)
    ap.add_argument("--height", type=int, default=None,
                    help="defaults to --width (square image)")
    ap.add_argument("--spp", type=int, default=4096)
    ap.add_argument("--base-dir", default=DEFAULT_BASE)
    ap.add_argument("--force", action="store_true",
                    help="overwrite if the file already exists")
    args = ap.parse_args()

    H = args.height or args.width
    W = args.width
    folder = os.path.join(args.base_dir, f"spp_renders-{W}x{H}-d-1")
    os.makedirs(folder, exist_ok=True)
    out = os.path.join(folder, "reference.exr")

    if os.path.exists(out) and not args.force:
        print(f"already exists, skipping: {out}")
        print(f"(pass --force to overwrite)")
        # still print stats for convenience
        arr = np.array(mi.Bitmap(out))[..., :3].astype(np.float64)
        print(f"  shape = {arr.shape}   mean Σ-RGB = {arr.sum(-1).mean():.4e}")
        return

    print(f"rendering reference at {W}×{H} with spp={args.spp}")
    print(f"  total pixel-samples: {W*H*args.spp:,}")

    scene = cornell_scene(W, H)
    t0 = time.time()
    img = mi.render(scene, spp=args.spp, seed=0)
    dt = time.time() - t0
    arr = np.array(img)

    mi.util.write_bitmap(out, arr)
    mi.util.write_bitmap(out.replace(".exr", ".png"), arr ** (1 / 2.2))

    print(f"  render time: {dt:.1f}s")
    print(f"  saved EXR : {out}")
    print(f"  saved PNG : {out.replace('.exr', '.png')}")
    print(f"  mean Σ-RGB lum = {arr[..., :3].sum(-1).mean():.4e}")

    print("\nTo point the other debug scripts at this reference:")
    print(f"    --ref {out}")


if __name__ == "__main__":
    main()
