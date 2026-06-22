"""
A. QMC convergence study (Hachisuka-style) against a fed-in reference EXR.

What this answers
-----------------
Does QMC actually beat plain Monte-Carlo for our unidirectional + NEE path
tracer, and *how many leading path dimensions* is it worth applying Sobol' to
before the benefit flattens out?

How it works
------------
The reference is a converged EXR you hand in (`--ref`) — e.g. the matched
vec-integrator render at baseline/spp_renders-cbox-128x128-vec-nee/. We do NOT
re-render it. The image resolution is read straight off the EXR so the test
renders always match it (a size mismatch would floor RMSE at a constant no
amount of sampling could fix).

Then we sweep, on the existing scalar `Renderer` + the existing QMC samplers:
  * spp        : shared powers-of-two grid (`--spp`) — the convergence axis.
  * dimensions : `SobolSampler(max_dim=K)` for each K in `--dims`. K = "the
                 first K path dimensions use Sobol', the tail is independent."
                 This is the knob, and it already exists — no new sampler.
Plus an `independent` line (plain MC, the slope-(-1/2) control). Every config
shares the identical spp grid and resolution, so any gap is sampler-attributable.

Two views (your question: sweep both, or one-spp-at-a-time across dims?)
-----------------------------------------------------------------------
Render the full (K, spp) grid once — cached — and read BOTH off it for free:
  LEFT  : log-log RMSE vs spp, one line per K (+ independent). The rigorous
          convergence claim. Read slope (asymptotic rate) AND vertical offset
          (the constant) — a lower line at equal slope is still a real QMC win.
  RIGHT : RMSE vs #Sobol-dims K at a few fixed spp. "Where does adding QMC stop
          paying off." Hachisuka's thesis made empirical. You CAN'T read a rate
          off this, so it's the secondary figure, not the headline.

Matching the reference (important)
----------------------------------
`--max-depth` must equal the depth the reference was rendered at, or RMSE floors
at a constant bias (different light transport). Default 8 matches the vec
integrator's default (pathtracer_vec.py). `--rr-depth` need not match — RR is
unbiased, it only changes variance, not the converged image.

Usage
-----
    # default: matched vec+NEE 128 reference, dims 1..64, spp 1..64
    ./venv/bin/python A_convergence.py        # run from quasi_monte_carlo/debug

    # the clean slope-separation case from the design discussion: low bounce, no RR
    ./venv/bin/python A_convergence.py --max-depth 2 --rr-depth 9999

    # smoother lines: average RMSE over independent realizations
    ./venv/bin/python A_convergence.py --trials 4

Everything is cached in debug/_cache (keyed by every image-affecting knob + a
hash of the sampler/integrator source), so reruns are instant and editing a
sampler auto-invalidates its stale renders.
"""
import _setup  # noqa: F401  (sys.path + scalar variant; must precede qmc imports)

import argparse
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import mitsuba as mi

from baseline import load_scene_box_rfilter
from qmc.renderer import Renderer
from qmc.setup import build_sampler
from _cache import RenderCache

REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
# MUST be a BOX-rfilter reference. The scalar Renderer accumulates one sample
# per pixel (a box filter) — comparing it to a tent/gaussian reference, whose
# bright emitters splat onto neighbouring pixels, leaves a fixed per-pixel
# difference that never averages out and floors RMSE (every slope collapses to
# ~-0.12, independent included). spp_renders_basic-* are force-boxed by
# baseline_basic.py; spp_renders-cbox-128x128-vec-nee is TENT — do not use it.
DEFAULT_REF = os.path.join(REPO, "baseline",
                           "spp_renders_basic-cbox-128x128-vec-nee",
                           "reference.exr")


def rmse(img, ref):
    """Linear-space RMSE — identical metric to baseline.py."""
    return float(np.sqrt(np.mean((np.asarray(img) - np.asarray(ref)) ** 2)))


def load_ref(path):
    arr = np.array(mi.Bitmap(path))
    if arr.ndim != 3 or arr.shape[-1] < 3:
        raise SystemExit(f"unexpected ref shape {arr.shape} at {path}")
    rgb = arr[..., :3].astype(np.float64)
    H, W = rgb.shape[:2]
    return rgb, W, H


def resolve_scene(scene_arg):
    """Accept a sample name ('cbox') or a direct path to scene.xml."""
    if scene_arg.endswith(".xml"):
        return scene_arg
    p = os.path.join(REPO, "samples", scene_arg, "scene.xml")
    if not os.path.exists(p):
        raise SystemExit(f"no scene at {p} (give a sample name or a .xml path)")
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", default=DEFAULT_REF, help="converged reference EXR")
    ap.add_argument("--scene", default="cbox",
                    help="sample name under samples/ or path to scene.xml")
    ap.add_argument("--dims", type=int, nargs="+",
                    default=[2, 4, 6, 10, 16, 22, 34, 64],
                    help="#Sobol' dimensions (max_dim) per config — the sweep. "
                         "Defaults land on fixed-block boundaries (camera=4, then "
                         "+6/bounce), so each cuts WHOLE bounces: 2=jitter, "
                         "4=camera, 6=+b1 NEE, 10/16/22=+b1/b2/b3, 34=+b5, 64=+b10.")
    ap.add_argument("--spp", type=int, nargs="+",
                    default=[1, 2, 4, 8, 16, 32, 64, 128],
                    help="shared powers-of-two spp grid (the convergence axis)")
    ap.add_argument("--scramble", default="owen", choices=["none", "xor", "owen"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trials", type=int, default=1,
                    help="independent realizations per point; RMSE is averaged "
                         "over them for smoother lines")
    ap.add_argument("--max-depth", type=int, default=8,
                    help="MUST match the reference's depth (else RMSE floors)")
    ap.add_argument("--rr-depth", type=int, default=3,
                    help="RR start depth; unbiased, need not match the reference")
    ap.add_argument("--no-independent", action="store_true",
                    help="drop the plain-MC control line")
    ap.add_argument("--padded", action="store_true",
                    help="also plot a padded_sobol line (all dims, shuffled)")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--out", default=None, help="path for the convergence plot")
    ap.add_argument("--out2", default=None, help="path for the contact-sheet of renders")
    args = ap.parse_args()

    ref, W, H = load_ref(args.ref)
    scene_path = resolve_scene(args.scene)
    scene_name = os.path.basename(os.path.dirname(scene_path)) \
        if scene_path.endswith("scene.xml") else os.path.basename(scene_path)

    scene = load_scene_box_rfilter(scene_path, resx=W, resy=H)
    fw, fh = (int(v) for v in scene.sensors()[0].film().crop_size())
    if (fw, fh) != (W, H):
        raise SystemExit(
            f"reference is {W}x{H} but the scene built at {fw}x{fh} — check "
            f"$resx/$resy in {scene_path}, or pass a matching --ref.")
    renderer = Renderer(scene, max_depth=args.max_depth, rr_depth=args.rr_depth)

    print(f"reference : {args.ref}")
    print(f"            {W}x{H}  mean luma = {ref.sum(-1).mean():.4e}")
    print(f"scene     : {scene_name}   max_depth={args.max_depth} rr_depth={args.rr_depth}")
    print(f"spp grid  : {args.spp}")
    print(f"dims      : {args.dims}   scramble={args.scramble}  trials={args.trials}")

    # ── configs: (label, sampler-name, max_dim) ───────────────────────────────
    configs = []
    if not args.no_independent:
        configs.append(("independent", "independent", None))
    for K in args.dims:
        configs.append((f"sobol d{K}", "sobol", K))
    if args.padded:
        configs.append(("padded_sobol", "padded_sobol", None))

    # ── render / load-from-cache the whole (config, spp, trial) grid ──────────
    # curves[label] = array of mean-over-trials RMSE, aligned with args.spp
    curves = {}
    grid = {}  # (label, spp) -> trial-0 image, kept for the contact sheet
    t_start = time.time()
    for label, name, K in configs:
        errs = []
        for spp in args.spp:
            trial_errs = []
            for t in range(args.trials):
                seed = args.seed + 1009 * t
                params = dict(scene=scene_name, W=W, H=H, sampler=name,
                              max_dim=K, scramble=(args.scramble if name != "independent" else None),
                              seed=seed, spp=spp,
                              max_depth=args.max_depth, rr_depth=args.rr_depth)
                cache = RenderCache(params, tag="A_convergence",
                                    enabled=not args.no_cache)
                img = cache.load()
                if img is None:
                    sampler = build_sampler(name, spp=spp, max_dim=(K or 64),
                                            scramble_method=args.scramble, seed=seed)
                    img = renderer.render(sampler, spp)
                    cache.save(img, stats={"rmse": rmse(img, ref),
                                           "mean": float(np.asarray(img).mean())})
                    tag = "rendered"
                else:
                    tag = "cached"
                if t == 0:
                    grid[(label, spp)] = img
                trial_errs.append(rmse(img, ref))
            e = float(np.mean(trial_errs))
            errs.append(e)
            print(f"  [{tag:8}] {label:14} spp={spp:>4}  rmse={e:.4e}"
                  f"   ({time.time()-t_start:.0f}s)")
        curves[label] = np.array(errs)

    spp = np.array(args.spp, dtype=float)

    # ── fitted slopes ─────────────────────────────────────────────────────────
    print("\nlog-log RMSE-vs-spp slopes (expect ~ -0.5 for plain MC):")
    slopes = {}
    for label, _, _ in configs:
        s, _b = np.polyfit(np.log(spp), np.log(curves[label]), 1)
        slopes[label] = s
        print(f"  {label:14} slope = {s:+.3f}")

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6))

    # LEFT: convergence — one line per config
    for label, name, K in configs:
        style = "k--o" if name == "independent" else "o-"
        axL.loglog(spp, curves[label], style if name == "independent" else "o-",
                   label=f"{label}  slope={slopes[label]:+.2f}  "
                         f"RMSE={f"{round(curves[label][-1], 5)}"}",
                   alpha=0.9 if name != "independent" else 1.0)
    anchor = curves[configs[0][0]][0]
    axL.loglog(spp, anchor * (spp / spp[0]) ** (-0.5), "k:", alpha=0.4, label="N^(-1/2)")
    axL.loglog(spp, anchor * (spp / spp[0]) ** (-1.0), "k-.", alpha=0.3, label="N^(-1)")
    axL.set_xlabel("spp (N)")
    axL.set_ylabel("RMSE vs reference")
    axL.set_title(f"convergence {scene_name} {W}x{H}, "
                  f"d{args.max_depth} rr{args.rr_depth}")
    axL.grid(True, which="both", alpha=0.3)
    axL.legend(fontsize=8)

    # RIGHT: RMSE vs #Sobol-dims, one line per spp ("where benefit lives")
    sob = [(label, K) for (label, name, K) in configs if name == "sobol"]
    if sob:
        Ks = np.array([K for _, K in sob], dtype=float)
        pick = list(args.spp)                       # every spp, not just a few
        for n in pick:
            j = args.spp.index(n)
            axR.semilogx(Ks, [curves[label][j] for label, _ in sob],
                         "o-", base=2, label=f"spp={n}, RMSE={round(curves[label][j], 5)}")
        if not args.no_independent:
            for n in pick:
                j = args.spp.index(n)
                axR.axhline(curves["independent"][j], ls="--", alpha=0.2)
        axR.set_xlabel("# Sobol' dimensions (max_dim)")
        axR.set_ylabel("RMSE vs reference")
        axR.set_title("rmse per each sobol vs spp")
        axR.grid(True, which="both", alpha=0.3)
        axR.legend(fontsize=8)

    plt.tight_layout()
    out = args.out or os.path.join(
        os.path.dirname(__file__),
        f"A_convergence_{scene_name}_{W}x{H}_d{args.max_depth}_rr{args.rr_depth}"
        f"_{args.scramble}_t{args.trials}.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"\nsaved: {out}")

    # ── second output: contact sheet of every render, each fully labelled ─────
    # Exposure matched to the reference median (same rule as the diff diagnostic)
    # so brightness is comparable across the whole grid and to the reference.
    ref_lum = ref.sum(-1)
    med = float(np.median(ref_lum[ref_lum > 0])) if (ref_lum > 0).any() else 1.0
    exposure = (0.18 / med) if med > 0 else 1.0

    pos = ref_lum[ref_lum > 0]
    key = float(np.exp(np.mean(np.log(pos)))) if pos.size else 1.0   # log-average, not median
    exposure = 0.18 / key if key > 0 else 1.0

    def _tonemap(im):
        s = np.asarray(im, dtype=np.float64) * exposure
        return np.clip((s / (1.0 + s)) ** (1 / 2.2), 0, 1)

    nrows, ncols = len(configs), len(args.spp) + 1   # +1 = reference column
    figG, axG = plt.subplots(nrows, ncols,
                             figsize=(1.7 * ncols, 2.0 * nrows), squeeze=False)
    def _short(name, K):
        if name == "independent":
            return "indep"
        if name == "padded_sobol":
            return "padded"
        return f"d{K}"

    for r, (label, name, K) in enumerate(configs):
        for c, n in enumerate(args.spp):
            ax = axG[r][c]
            ax.imshow(_tonemap(grid[(label, n)]))
            # tile title is just the config id (d{K}); spp labels the column once
            # on the top row; scramble/depth/rr/scene are shared -> suptitle.
            ax.set_title(f"spp{n}\n{_short(name, K)}" if r == 0
                         else _short(name, K), fontsize=7)
            ax.axis("off")
        axr = axG[r][ncols - 1]
        axr.imshow(_tonemap(ref))
        axr.set_title("reference" if r == 0 else "", fontsize=7)
        axr.axis("off")
    figG.suptitle(
        f"all renders. settings: {scene_name} {W}x{H} · scramble={args.scramble} · "
        f"max_depth={args.max_depth} · rr_depth={args.rr_depth} · "
        f"trials={args.trials} · ref={os.path.basename(os.path.dirname(args.ref))}",
        fontsize=11)
    figG.tight_layout(rect=[0, 0, 1, 0.97])
    out_g = args.out2 or os.path.join(
        os.path.dirname(__file__),
        f"A_images_{scene_name}_{W}x{H}_d{args.max_depth}_rr{args.rr_depth}"
        f"_{args.scramble}_t{args.trials}.png")
    figG.savefig(out_g, dpi=110)
    plt.close(figG)
    print(f"saved: {out_g}")


if __name__ == "__main__":
    main()
