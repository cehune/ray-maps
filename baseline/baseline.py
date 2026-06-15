import argparse
from pathlib import Path

import numpy as np
import mitsuba as mi

# --- Paths: anchor on THIS file, never on the cwd ---
SCRIPT_DIR = Path(__file__).resolve().parent              # .../ray-maps/baseline
SAMPLES_DIR = (SCRIPT_DIR / ".." / "samples").resolve()   # .../ray-maps/samples


def parse_args():
    p = argparse.ArgumentParser(
        description="SPP sweep + log-log RMSE slope for a Mitsuba 3 sample scene."
    )
    p.add_argument(
        "sample",
        help="Sample directory name under ../samples (e.g. 'lamp', 'cbox'). "
             "Resolves to ../samples/<sample>/scene.xml.",
    )
    p.add_argument(
        "--width", type=int, default=None,
        help="Target film width. Height auto-scales to preserve the scene's "
             "authored aspect ratio. Requires $resx/$resy in the scene XML. "
             "Omit to render at the authored resolution.",
    )
    p.add_argument("--ref-spp", type=int, default=16384,
                   help="SPP for the converged reference (default: 16384).")
    return p.parse_args()


def main():
    args = parse_args()
    mi.set_variant("scalar_rgb")

    scene_path = SAMPLES_DIR / args.sample / "scene.xml"
    if not scene_path.exists():
        raise FileNotFoundError(f"No scene at {scene_path}")

    # Load once at the authored resolution just to read the aspect ratio.
    # This is a parse, not a render -- cheap.
    scene = mi.load_file(str(scene_path))
    fs = scene.sensors()[0].film().crop_size()
    w0, h0 = int(fs[0]), int(fs[1])
    aspect = w0 / h0

    if args.width is not None:
        width = args.width
        height = round(width / aspect)               # height auto-scaled
        scene = mi.load_file(str(scene_path), resx=width, resy=height)

        # Confirm the override bound to a $resx/$resy in the XML; otherwise the
        # film built at the wrong size and the RMSE-vs-reference is meaningless.
        fs = scene.sensors()[0].film().crop_size()
        aw, ah = int(fs[0]), int(fs[1])
        if (aw, ah) != (width, height):
            raise RuntimeError(
                f"Requested {width}x{height} but film built at {aw}x{ah}. "
                f"Check that {scene_path.name} uses $resx / $resy."
            )
    else:
        width, height = w0, h0

    scale = width / w0
    print(f"Sample        : {args.sample}")
    print(f"Authored res  : {w0}x{h0}")
    print(f"Rendering at  : {width}x{height}  ({scale:.4g}x authored)")

    # Folder is keyed on sample name + resolution, so different scales never
    # collide and a cached reference can't be reused at the wrong size.
    out_dir = SCRIPT_DIR / f"spp_renders-{args.sample}-{width}x{height}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Reference: load if cached, else render and cache ---
    ref_exr = out_dir / "reference.exr"
    if ref_exr.exists():
        reference = np.array(mi.Bitmap(str(ref_exr)))
        print(f"Reference     : loaded from disk ({width}x{height})")
    else:
        print(f"Reference     : rendering {width}x{height} @ {args.ref_spp} spp...")
        reference = np.array(mi.render(scene, spp=args.ref_spp, seed=0))
        mi.util.write_bitmap(str(ref_exr), reference)
        mi.util.write_bitmap(str(ref_exr.with_suffix(".png")), reference ** (1 / 2.2))
        print("Reference     : saved.")

    def rmse(img, ref):
        return float(np.sqrt(np.mean((np.asarray(img) - np.asarray(ref)) ** 2)))

    # --- PT-vs-itself sanity sweep ---
    spps = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    pt_errs = []
    print(f"PT sweep      : {width}x{height}, spp = {spps}")
    for i, spp in enumerate(spps):
        print(f"  spp={spp:>4}  @ {width}x{height}")
        img = mi.render(scene, spp=spp, seed=1000 + i)
        pt_errs.append(rmse(img, reference))                 # linear, pre-tonemap
        mi.util.write_bitmap(str(out_dir / f"{spp}.exr"), img)            # linear HDR, for analysis
        mi.util.write_bitmap(str(out_dir / f"{spp}.png"),
                             img ** (1 / 2.2))                # gamma, for viewing only

    # RMSE ~ N^{-1/2} for an unbiased estimator => slope ~ -0.5 in log-log.
    slope, _ = np.polyfit(np.log(spps), np.log(pt_errs), 1)
    print(f"Rendered at   : {width}x{height}")
    print(f"PT sanity slope = {slope:.3f}  (expect ~ -0.5)")


if __name__ == "__main__":
    main()