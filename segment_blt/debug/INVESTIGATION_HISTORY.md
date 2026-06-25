# Segment-BLT diagnosing investigation — memory bank

Record of the one-off diagnostic scripts written during the `diagnosing` work to
chase three coupled symptoms: **splotchy output**, **fireflies without clamping**,
and **never converging** (RMSE plateau, mean_ratio ≈ 0.97). The scripts themselves
were pruned for the clean merge; this file preserves *what each tested, why it was
valid, and what it concluded* so the reasoning isn't lost.

The lettered scripts that survived (`G_convergence`, `J_alpha_sweep`,
`F_ab_vs_reference_all`, `K_ball_kernel`, `M_lookup_compare`, `N_radius_sweep`) are
the reusable tooling; everything below was a single-question probe.

---

## Headline conclusions (what the whole investigation established)

1. **The 1/K kernel correction is a mistake.** The uniform-kernel density `K = 1/area`
   cancels between the propagation numerator and the MMIS `Σp` denominator, so it must
   *not* be applied — doing so double-counts the area and halves the image. (`E`)
   → the entire `apply_kernel_correction` stack was deleted.

2. **The residual error was a *bias floor*, not variance** — flat RMSE vs radius and a
   shallow log-log slope. Decomposed via the `RMSE² = floor² + var/N` fit. (`L_r_sweep`, `J`)

3. **A big chunk of the "never converges" was a reconstruction-filter artifact.** The
   reference used a tent/gaussian rfilter (which blooms the bright emitter ~1px); the
   BLT box-accumulates one sample/pixel. The mismatch put a fixed dark ring around the
   light into the RMSE floor. Forcing a **box rfilter** on the reference removed it and
   restored a healthy ≈ −0.42 convergence slope. (see `baseline.load_scene_box_rfilter`)

4. **The ~3% mean deficit is two stacked things, both real, neither the clamp:**
   `merge_min_len_factor` (sub-kernel exclusion → classical links) leaks ≈ 1.7%, plus a
   residual ≈ 2.8% **voxel-coverage** undercount at surface edges/corners. (`O`–`Q2`, `K_exact`)
   `row_gain_cap` was ruled out (removes ~0.01%).

5. **The camera film-mapping `(x+u)/W` is exactly correct** — verified against Mitsuba's
   own pixel sampling (a ½-pixel shift blows the emitter-edge error up 14,000×). The faint
   line below the emitter is edge variance + merge boundary bias, not a registration bug.

6. **NEE was architecturally wrong** — folded into the CAMERA technique via an additive
   `p_nee`. Reframed as a first-class **BRIDGE** technique (paper §S2/S8), output-identical
   in the camera-only limit but correct and extensible.

7. **Scene source-of-truth split:** the BLT rendered `mi.cornell_box()` while baseline
   referenced `samples/cbox/scene.xml` — *different* camera/geometry/light. Consolidated
   both on the XML scene so mean_ratio is meaningful.

---

## Phase 1 — Pipeline sanity (does the machinery work at all?)

- **`C_direct_only`** — `num_prop_iterations=0`. With zero propagation only light-terminal
  segments hold `rad_in=Le`; output should be black except a full-brightness light disk.
  Validates throughput accumulation, `Le` assignment, and trivial mmis-weight marking
  *before* trusting any indirect result.

- **`A_parity_vec_vs_nonvec`** — same scene+seed through both BLT pipelines. They should
  agree up to known divergences: vec clamps weights/geom (nonvec doesn't) and vec's
  pair-cache skipped clusters with ≤5 endpoints. Outlier-dominated diff ⇒ clamps;
  broadly-spread diff ⇒ the skip or a real vectorized-accumulation bug. (The ≤5 skip was
  later removed.)

- **`B_bounce_ladder`** — sweep `num_prop_iterations ∈ {0,1,2,4,8,16}` (+ optional `cluster_c`
  sweep). Checks propagation saturates to `gain/(1−gain)`. Found a clean plateau ≈ 1.31×
  the single-bounce value by N≈8 (bounces are *not* the convergence problem). Introduced
  the "zero-incoming" / "camera-first 0%" coverage instrumentation.

- **`D_pixel_forensics`** — pick one pixel, dump the top-K propagation pairs flowing into
  it (BSDF, geom, mmis_w, rad_in, whether the source is on the wrong surface). The tool for
  catching coverage holes, cross-surface key leaks, and single-pair fireflies by eye.

## Phase 2 — The bias hunt (mean_ratio ≈ 0.97)

- **`E_kernel_correction`** — A/B the standard estimator vs ×`1/K(receiver_cluster)`.
  **Verdict: 1/K is implicit in MMIS** (ratio overshoots, not corrects) → deleted. (Headline #1)

- **`N_bsdf_only_ref`** — plain unidirectional BSDF-only PT (same sampling BLT is built on,
  no clustering/MMIS/propagation), scored like `G`. Isolates whether the ~0.967 deficit is the
  *sampling strategy* (heavy-tailed near small emitters) or the *BLT estimator*. Cure for the
  former = reintroduce the light-segment / bridge technique.

- **`O_direct_calibration`** — one application of the gather operator: `T0·D(v1)` vs Mitsuba
  `max_depth=2`. Isolates whether the ~+3-4% overshoot is per-gather geometry or Le injection
  scale. Direct came out calibrated (~0.99) → the excess is in *multi-hop* transport.

- **`O2_v2_gather_calibration`** — the same single-gather quantity but at the *second* vertex
  (`T0·T1·D(v2)` vs `PT(3)−PT(2)`). Tests whether gather bias is **receiver-geometry
  conditional** — v2 concentrates in corners/under boxes where the hot structure lives.

- **`O3_readout_ladder`** — bucket the depth-99 split readout by level, assert
  `Σ buckets == readout`, and compare each cumulative level against the PT depth ladder.
  Removes interpretation: pinpoints *which* readout level injects the heat. (Found levels
  1..16 calibrated; the heat is in the deep/last-vertex bucket.)

- **`O4_last_vertex`** — why `D` at the *last* vertex of max-depth-truncated paths read ~300×
  hot. Groups `lum(direct_in)` by receiver depth and decomposes the depth-17 gather pair-by-pair
  (own-bridge / other-bridge / bsdf-hit / classical) to expose a broken weight directly.

- **`P_bounce_calibration`** — BLT cumulative energy per bounce order vs a fresh Mitsuba depth
  ladder (`pixel(N)` ↔ `max_depth=N+2`). Localizes the +3.4% by bounce: flat-then-jump ⇒ one
  bounce broken; steady creep ⇒ per-hop inflation.

- **`Q_reuse_correlation`** — is the hot indirect **hop-reuse correlation** (the same operator
  `M` applied k times shares realized denominators/pairs, so `E[Mᵏ·Le] ≠ E[M]ᵏ·Le`) or per-hop
  calibration bias? Arm A reuses one cache for k hops; arm B re-jitters clustering each hop.

- **`Q2_hop_pools`** — corrected version of Q: arm B gives each hop a genuinely independent
  operator via **disjoint round-robin path-pools** (`pool_mask`), split *by path* so each
  child's parent stays in `p_sum` (the `w·G ≤ π` bound survives). The scale-free ~1.3×/hop
  signature is exactly what same-pool compounding `(1+relvar)ᵏ` would leave.

## Phase 3 — Firefly mechanism & the clamp

- **`M_firefly_mechanism`** — validated *why* the clamp was needed, three claims on one
  iteration (no clamps): (1) **guard asymmetry** — `conditional_pdf` zeroed the parent pdf
  term while the numerator kept `f·G>0` (unbounded `w·G`); (2) **parent dominance** — for
  `len ≪ voxel_size`, `p_sum` is dominated by the parent so `w·G ≈ π·cos` transfers full ρ·L
  with no 1/N averaging; (3) **loop gain** — corner voxels of m short segments form a subgraph
  with per-hop gain `m·ρ>1`, compounding `(m·ρ)ᴺ` across the reused prop iterations (measured
  as the operator's spectral radius via power iteration). → motivated `merge_min_len_factor` +
  `row_gain_cap`.

- **`H_clamp_sweep`** — geom/MMIS clamp factor sweep, final RMSE & mean_ratio vs reference.
  Mapped the bias/variance tradeoff: tight clamp = low-variance/dark; loose = unbiased/firefly-y.

- **`I_soft_vs_hard_clip`** — hard `min(geom,cap)` vs soft Reinhard `geom·cap/(cap+geom)` at
  matched factor. Soft keeps mean_ratio closer to 1.0 (less tail energy lost) at slightly
  higher early variance.

## Phase 4 — Kernel / lookup geometry (the splotch & the floor)

- **`K_exact_vs_voxel`** — is voxel-hash clustering a faithful "all pairs within r"? An
  axis-aligned voxel is neither a ball nor query-centered → false negatives (close pts in
  different voxels) and false positives (opposite corners up to r√3). Swaps *only* the
  enumeration (`build_pair_cache` vs `build_pair_cache_exact`) to decide whether the data
  structure is a confound before trusting any kernel diagnostic. (`build_pair_cache_exact`
  was removed with this script.)

- **`L_r_sweep_bias_floor`** — exact-radius r-sweep looking for the **bias floor**.
  `RMSE(r,N) ≈ C_bias·r^β + C_var·r^(−d/2)·N^(−1/2)`: a U-shape, minimum at the sweet spot.
  The diagnostic: if RMSE plateaus at a positive floor no r-shrink/sample-budget can lower,
  the **weights/PDFs are wrong** (MMIS denominator or kernel normalization). That floor *is*
  "splotchy and never converges." Uses exact pairs to remove the data structure as a confound.

- **`L_disc_kernel`** — restrict neighbors to the receiver's **tangent plane** (radius +
  height-tol + normal-cone) so cross-surface neighbors near corners (wall pt ↔ floor pt) don't
  contaminate the density estimate. `K = 1/(πr²)` inside the disc; the canonical photon-mapping
  kernel. A candidate cure for the corner-coverage residual.
