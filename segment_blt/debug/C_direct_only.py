"""
C. Direct-only render: num_prop_iterations=0.

With zero propagation, only segments whose y (or x) is_light hold nonzero
rad_in (= Le). All other segments stay at 0. The output should be:

  • Black almost everywhere.
  • The visible light disk (camera segments whose first hit IS the emitter)
    at full brightness ≈ throughput × Le.

If the light disk is dim, wrong-color, or missing → the direct path is
broken (throughput accumulation, Le assignment, or mmis_w trivial-marking).
No indirect pipeline can fix that.
"""
from _setup import cornell_scene

import os
import numpy as np
import matplotlib.pyplot as plt
import mitsuba as mi
from segments.renderer import Renderer


def main():
    W = H = 200
    scene = cornell_scene(W, H)
    renderer = Renderer()

    img = renderer.render_vec(
        scene, height=H, width=W, n_iterations=1,
        num_prop_iterations=0,
        verbose=False, beta=0.1, add_light_samples=False,
    )

    lum = img.sum(-1)
    nz = lum > 0
    print("── direct-only ─────────────────────────────")
    print(f"  nonzero pixels : {nz.sum()} / {W*H}  ({100*nz.mean():.2f}%)")
    if nz.any():
        print(f"  lum (nz)       : med={np.median(lum[nz]):.4e}  "
              f"min={lum[nz].min():.4e}  max={lum[nz].max():.4e}")
        # quick sanity vs expected: Le ≈ 18 → luma ≈ 0.2126*Le_R + 0.7152*Le_G + ...
        # for a white emitter (R=G=B=18.4), luma ≈ 18.4. Cornell light may be tinted.
        print(f"  expected order : luma(Le) ≈ 18 for Cornell ceiling emitter")
        print(f"  if max(lum) ≫ 18 → throughput is being double-counted")
        print(f"  if max(lum) ≪ 18 → throughput < 1 (check sensor ray_weight)")
    else:
        print("  ALL pixels zero — first-hit-on-light path is dead.")

    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    ax[0].imshow(np.clip(img ** (1/2.2), 0, 1))
    ax[0].set_title("direct-only (tonemapped)")
    ax[0].axis('off')
    im = ax[1].imshow(np.log10(lum + 1e-6), cmap='viridis')
    ax[1].set_title("log10(luminance)")
    plt.colorbar(im, ax=ax[1], fraction=0.046)
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "C_direct_only.png")
    plt.savefig(out, dpi=100); plt.close()
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
