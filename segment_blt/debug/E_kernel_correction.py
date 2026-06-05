"""
E. Test the missing 1/K hypothesis.

Renders the same scene twice — once with the standard estimator, once with
each per-pair contribution multiplied by 1/K(receiver_cluster). Reports
mean luminance of both, plus the ratio.

Interpretation:
  • ratio jumps from 0.43 → ~ref_mean  → the 1/K factor was genuinely missing.
  • ratio overshoots wildly (≫ ref)    → the factor IS needed but K isn't
        the right quantity (might need π * K or scene-area normalization).
  • ratio barely moves                  → the factor really is implicit in MMIS.

Compare the "vec ON" mean to your reference EXR's mean luminance. Quickest
way to measure that without a new script:

    import numpy as np, mitsuba as mi
    ref = np.array(mi.Bitmap("path/to/reference.exr"))[..., :3]
    print("ref mean lum:", ref.sum(-1).mean())

You said spp=48 is fine for the ref — that's plenty for getting the mean
to ~1% accuracy, way more than we need to compare against a biased estimator.
"""
from _setup import cornell_scene

import os
import numpy as np
import matplotlib.pyplot as plt
import mitsuba as mi
from segments.renderer import Renderer


def main():
    W = H = 128
    N = 8  # plateau-region prop iterations (we're testing the fixed point, not truncation)
    scene = cornell_scene(W, H)
    renderer = Renderer()

    print("── OFF (standard estimator) ──")
    img_off = renderer.render_vec(
        scene, height=H, width=W, n_iterations=1,
        num_prop_iterations=N, cluster_c=30,
        apply_kernel_correction=False,
        verbose=False, beta=0.1, add_light_samples=False,
    )
    mean_off = float(img_off.sum(-1).mean())
    p99_off  = float(np.percentile(img_off.sum(-1), 99))

    print("\n── ON (× 1/K per pair) ──")
    img_on = renderer.render_vec(
        scene, height=H, width=W, n_iterations=1,
        num_prop_iterations=N, cluster_c=30,
        apply_kernel_correction=True,
        verbose=False, beta=0.1, add_light_samples=False,
    )
    mean_on = float(img_on.sum(-1).mean())
    p99_on  = float(np.percentile(img_on.sum(-1), 99))

    print("\n── results ─────────────────────────────")
    print(f"  OFF: mean_lum={mean_off:.4e}  p99={p99_off:.4e}")
    print(f"  ON : mean_lum={mean_on:.4e}  p99={p99_on:.4e}")
    if mean_off > 0:
        print(f"  ratio ON/OFF: {mean_on / mean_off:.3f}×")
    

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(np.clip(img_off ** (1/2.2), 0, 1))
    axes[0].set_title(f"OFF  mean={mean_off:.2e}")
    axes[0].axis("off")
    axes[1].imshow(np.clip(img_on ** (1/2.2), 0, 1))
    axes[1].set_title(f"ON   mean={mean_on:.2e}")
    axes[1].axis("off")
    diff = img_on.sum(-1) - img_off.sum(-1)
    im = axes[2].imshow(diff, cmap="RdBu_r", vmin=-abs(diff).max(), vmax=abs(diff).max())
    axes[2].set_title("Σ(ON - OFF)")
    plt.colorbar(im, ax=axes[2], fraction=0.046)
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "E_kernel_correction.png")
    plt.savefig(out, dpi=100); plt.close()
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
