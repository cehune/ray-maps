"""
A. vec vs nonvec parity check.

Runs the SAME scene + seed through both BLT pipelines. If the algorithm is
sampler-agnostic, they should agree. They won't agree perfectly — known
divergence sources:

  1. vec MMIS clamps weights (cap = 20×median); nonvec does not.
  2. vec propagation clamps geom_term (cap = 20×median); nonvec does not.
  3. vec pair_cache SKIPS clusters with ≤5 endpoints (pair_cache.py:94);
     nonvec iterates every cluster.

If the diff is dominated by a handful of outlier pixels → divergence is
from the clamps (1, 2). If diff is broadly spread → it's (3) or a real bug
in the vectorized accumulation.
"""
from _setup import cornell_scene

import os
import numpy as np
import matplotlib.pyplot as plt
import mitsuba as mi
from segments.renderer import Renderer


def main():
    W = H = 64  # small for speed; both pipelines are O(S²)-ish
    scene = cornell_scene(W, H)
    renderer = Renderer()

    print("running nonvec ...")
    img_nv = renderer.render_nonvec(
        scene, height=H, width=W, n_iterations=1,
        verbose=False, beta=0.1, add_light_samples=False,
    )
    print("running vec ...")
    img_v = renderer.render_vec(
        scene, height=H, width=W, n_iterations=1,
        verbose=False, beta=0.1, add_light_samples=False,
    )

    diff = img_v - img_nv
    abs_diff = np.abs(diff)
    print(f"\n── results ─────────────────────────────")
    print(f"  vec  mean lum = {img_v.sum(-1).mean():.4e}")
    print(f"  nonv mean lum = {img_nv.sum(-1).mean():.4e}")
    print(f"  ratio vec/nv  = {img_v.sum(-1).mean()/max(img_nv.sum(-1).mean(),1e-12):.4f}")
    print(f"  |diff|  max   = {abs_diff.max():.4e}")
    print(f"  |diff|  median= {np.median(abs_diff):.4e}")
    print(f"  |diff| > 1e-4 : {(abs_diff > 1e-4).mean()*100:.2f}% of values")
    print(f"  |diff| > 1e-2 : {(abs_diff > 1e-2).mean()*100:.2f}% of values")

    for c, name in enumerate("RGB"):
        print(f"  {name}: max|d|={abs_diff[..., c].max():.3e}  "
              f"v_mean={img_v[..., c].mean():.3e}  "
              f"nv_mean={img_nv[..., c].mean():.3e}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(np.clip(img_nv ** (1/2.2), 0, 1)); axes[0].set_title("nonvec")
    axes[1].imshow(np.clip(img_v ** (1/2.2), 0, 1)); axes[1].set_title("vec")
    im = axes[2].imshow(abs_diff.sum(-1), cmap='hot')
    axes[2].set_title("Σ|diff|")
    plt.colorbar(im, ax=axes[2], fraction=0.046)
    out = os.path.join(os.path.dirname(__file__), "A_parity.png")
    plt.tight_layout()
    plt.savefig(out, dpi=100); plt.close()
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
