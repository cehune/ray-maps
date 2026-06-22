# Segment-BLT — Findings, Changes & Tests

*Reconstructed from the debugging chain and verified against the code actually in this folder. Nothing below is taken on faith from the chat log — every claim was re-checked against `src/segments/` and the test suite (85 passed / 9 asset-missing, run in a fresh Mitsuba 3.8 sandbox).*

---

## 1. What this code is

A reimplementation of **Bidirectional Light Transport by segment merging** (the "segment BLT" paper). Camera (and optionally light) subpaths are broken into independent **segments**; segments are bucketed into voxel **clusters**; within a cluster, a segment can *merge* with a neighbor's endpoint to transport radiance, weighted by **MMIS** (multiple multiple-importance sampling). The estimate is built by iterating a propagation operator over the pool, then reading radiance back at each pixel's first hit.

The working scene throughout is the **Cornell box, camera-only**, compared against an unbounded Mitsuba path tracer. The whole investigation answers one question:

> *"The clamp shouldn't exist — something is wrong with MMIS or the estimator that causes fireflies, and the debug tests never converge."*

The verdict: the MMIS algebra was **faithful to the paper**. The fireflies and non-convergence were a stack of separate, smaller defects that two clamps had been papering over. Both clamps are now retired.

---

## 2. The arc (why it went splotchy → grainy → finally converged)

The headline symptom changed shape three times, and each shape was a *different* disease:

1. **Splotches** — every pixel read the cache at its first hit, so all pixels in one voxel shared a single correlated estimate. Fixed by a **split, depth‑$d$ final gather**.
2. **Grain** — after the split, the image went from blotchy to sparse-event noisy. Root cause was **radiance starvation**: with no next-event estimation, radiance only entered the pool when a camera path hit the emitter by BSDF luck (~4% of clusters had any Le source). Fixed by **NEE segments**.
3. **A hot bias floor (~1.07)** that NEE then *exposed* (it had been hidden under the grain). Root cause was a one-line **readout bug**: the gather read the full voxel-averaged cache at the last vertex of *dead* paths. Fixed by the **dead-end rule**.

After the dead-end fix the mean ratio flipped from $+7\%$ hot to $\approx 0.97$ cold — the small, **asymptotically-vanishing** density-estimation bias the paper itself documents. That residual is a *tuning* question (kernel schedule), not a bug.

---

## 3. Bugs found and fixed (the canonical list)

Each is verified present in the code, with the file/mechanism noted.

### 3.1 Guard asymmetry — *firefly class* (`sampler_types.py`, `primitives.py`, `mmis.py`)
`conditional_pdf` was zeroing pdf terms at **looser** epsilons than `Segment` admission used (e.g. `len_sq < 1e-10` vs a min length of `1e-8`, plus a gratuitous grazing-cosine guard, plus a `p_sum > 1e-5` floor that allowed weights up to $10^5$). Whenever the numerator keeps $f\cdot G$ but the denominator loses its cancelling term, the contribution is unbounded.

**Fix:** a single source of truth `MIN_SEG_LEN = 1e-8` (`primitives.py`), used identically in `Segment.__post_init__` and in both samplers' `conditional_pdf`; the grazing-cosine guard removed ($p_\omega$ already $\to 0$ smoothly at grazing); strict-positive `p_sum` in `compute_all_mmis_weights_vec`. *Measured inactive in the pool — fixed on principle, and locked by `test_no_guard_kills`.*

### 3.2 Corner divergence — *the actual fireflies* (`pair_cache.py`, `propagation.py`)
Sub-kernel segments (length < one voxel) were the real source. The virtual-perturbation pdf prices $x'$ as $\text{Uniform(voxel)}$ while the sample sits exactly at the parent's endpoint, so the singular $G\sim 1/\ell^2$ is overvalued by $\sim(\text{voxel}/\ell)^2$. Corner voxels full of short segments formed a **feedback loop** through the reused segment pool: measured spectral radius $\rho(M)=1.303$, gain $8.3\times$ over 8 hops, eigenmode localized at a box–floor corner.

**Fix:** **sub-kernel exclusion** (`merge_min_len_factor`, default `1.0`): segments shorter than `factor × voxel_size` are excluded as **merge sources** (but still receive radiance and still act as MMIS auxiliaries). Result: $\rho(M)=0.50$, eigenmode delocalized, fireflies gone. The bias from exclusion $\to 0$ as the progressive kernel shrinks. *Locked by `test_subkernel_sources_excluded` and `test_spectral_radius_below_one`.*

### 3.3 Energy deficit from the exclusion → *classical links* (`pair_cache.py`, `propagation.py`)
Excluding short segments left a dark ring around the emitter and dark box–floor contact (any two surfaces within one voxel). **Classical links** route an excluded segment's radiance to its **exact path parent** (identified by `id(child.x) == id(parent.y)`) with weight `= child.throughput` $= f\cos\theta/p_\omega \le$ albedo — i.e. plain path tracing for the near field, whose own pdf already cancels the $1/\ell^2$ singularity. NEE segments are excluded from this routing (their near field is the BSDF child's job; routing both would double-count direct light). Direct calibration after the fix: $0.993$ vs a fresh PT $d{=}2$ ground truth. *Locked by `test_classical_links` (252 links verified, exact parentage, weights ≤ albedo, zero overlap with merge edges).*

### 3.4 Energy-conservation projection — *replaces the geom clamp* (`propagation.py`)
Per-receiver merged row gain $\sum_j \text{lum}(w_j f_j G_j)$ estimates that cluster's directional-hemispherical reflectance, which is physically $\le$ albedo. Rows above `row_gain_cap` (default `1.0`) are scaled down. This bounds the operator's spectral radius below 1 (divergence impossible) and is **dormant** in a healthy pool — no free parameter, no clamp threshold. *An earlier version subtracted classical-link gain from the budget; that wrongly shaved ~1%/iteration and was removed.* *Locked by `test_row_gain_cap` (max pre-cap gain 1.011, 2/1456 rows over cap).*

### 3.5 Small-cluster cutoff (`pair_cache.py`)
`build_pair_cache` had a `<= 5` cluster cutoff that silently deleted **all** pairs in small clusters — energy loss that grew as the kernel shrank. Now `< 2` (the only genuinely degenerate case).

### 3.6 Hidden progressive shrinkage (`cluster.py`, `renderer.py`)
`Cluster.cluster()` self-incremented `_iteration`, so even "non-progressive" runs shrank the kernel — the cause of the slow downward mean-ratio drift. The caller now **owns the schedule**: `render_vec` sets `cluster._iteration = i` only when `progressive=True`. *Locked by `test_progressive_schedule_owned_by_caller`.* (Note this cuts both ways: it later turned out progressive shrinkage was the reason an α-sweep "did nothing" — see §6.)

### 3.7 Splotches → split, depth-$d$ final gather (`renderer.py`, `propagation.py`)
Replaced the single shared-cache readout with a **depth-$d$ split** (`_readout_path`):
$$ R_k = D(s_k) + T_{k+1}\,R_{k+1}\quad(k<d),\qquad R_d = \text{rad\_in}(s_d) $$
with a **light-terminal drop** at every split level (a path's own emitter hit is already inside $D$ with its MMIS weight, so a raw $L_e$ never multiplies a single path weight). `direct_in` $D(v)$ is computed in `iterate_propogation_vec` as the gather restricted to light-terminal sources (shared NEE samples → smooth direct); indirect goes through the pixel's **own** bounce (decorrelates pixels). $d=2$ reproduces the old gather exactly; $d=99$ is the diagnostic "PT-with-cached-direct, zero indirect cache" arm. Convergence slope improved $-0.21 \to -0.34\ldots-0.42$. *Locked by `test_direct_split_readout` and `test_depth_d_readout` (depth-2 parity exact to `1e-12`).*

### 3.8 NEE segments — *the grain fix* (`segment_gen_sampler.py`, `sampler_types.py`, `renderer.py`)
The actual gap vs Mitsuba: **no next-event estimation anywhere.** Added one emitter sample per path vertex (`make_nee_segment`), visibility-tested, added to the pool as a light-terminal **camera-technique** segment. MIS against BSDF-found light hits is free: every light-terminal segment stores its NEE area pdf `nee_parea` $=p_\omega\cos/d^2$, and `conditional_pdf` adds it per-auxiliary, so the denominator becomes the balance heuristic
$$ p_\text{sum} = \sum_t\Big(p_\omega^{\,t}\cos/d^2 + p_A\Big). $$
Measured at the same iteration count: Le sources in pool $13\to 75$, clusters with $\ge1$ Le source $4\%\to 61\%$, per-pixel std down $5.2\times$ (a $\sim 27\times$ variance cut), calibration held at 1.024. **Faithfulness note:** this is the paper's own §S2 bridge sampler ("in the spirit of NEE"), a sanctioned interim — *not* the headline config. The faithful endgame is the light-technique repair (§7). NEE is what makes camera-only mode testable at all; the previous BSDF-only-injection state appears nowhere in the paper.

### 3.9 Dead-end readout rule — *the +7% hot floor* (`renderer.py`)
The hot bias that survived everything. At a path's last vertex, `_readout_path` was reading the **full voxel-averaged cache** (`radiance_in`) even when the path had **died** (hit `max_depth`, or threw a BSDF ray into a backface). A dead path's continuation is a genuine **zero** MC sample — the ray saw black — but the cache read injected the voxel's average indirect radiance there. Pure survivorship bias, and it scaled with kernel size (O3 tail bucket $3.3\text{e-}2\to3.85\text{e-}2$ as $c:10\to90$), which is exactly the entire fg99 overcount.

**Fix:** distinguish "stopped here **by design** (fg-depth cap, path still alive)" — read the full cache, it's the legitimate merged estimate — from "the path **died** here" — seed `direct_in` only. The gate is `last == len(path) - 1 and not (seg.y.is_light or seg.x.is_light)`. After the fix the fg=2 mean ratio went $1.04\to1.004$ and fg=99 $1.072\to1.0147$. *Locked by the corrected `test_depth_d_readout` inline mirror, which now distinguishes "died at $v_k$" from "capped at $v_k$, alive."*

---

## 4. Estimator invariants (the regression locks)

`tests/test_estimator_invariants.py` pins each fix so it can't silently regress. Independently re-run here, measured values:

| # | Invariant | Test | Measured |
|---|-----------|------|----------|
| 1 | Guard symmetry — reachable segments keep their parent term | `test_no_guard_kills` | 0 kills |
| 1 | $w_j\,G_j \le \pi$ (parent-term bound) | `test_parent_term_bound` | max $w\cdot G = 3.1416$ — **bound achieved exactly** |
| 2 | Sub-kernel sources excluded | `test_subkernel_sources_excluded` | 0 short sources |
| 2 | $\rho(M) < 1$ (no divergence) | `test_spectral_radius_below_one` | $\rho = 0.439$ |
| 3 | Classical links: exact parentage, $w\le$ albedo, no merge overlap | `test_classical_links` | 252 verified |
| 4 | Row-gain cap dormant | `test_row_gain_cap` | max 1.011, 2/1456 over cap |
| 6 | Progressive schedule owned by caller | `test_progressive_schedule_owned_by_caller` | pass |
| 7 | Split gather: $D \le$ total, depth-1/2 agree | `test_direct_split_readout` | ratio 1.087 |
| 7 | Depth-$d$ readout: $d{=}2$ parity exact, deeper bands | `test_depth_d_readout` | $d=2,3,4,99$ all within band |

The $\pi$-bound is the keystone. For any diffuse segment $j$ whose path-parent $t^*$ is in the pool,
$$ q(j\mid t^*) = \frac{|n_x\cdot\hat s_j|}{\pi}\cdot\frac{|n_y\cdot\hat s_j|}{\ell_j^2} = \frac{G(j)}{\pi}, $$
and the parent shares $j.x$'s `SurfacePoint` (same position → same jitter → same voxel), so $p_\text{sum}\ge G(j)/\pi$ and therefore $w_j G_j \le \pi$. Hitting **3.1416 exactly** means the parent-term cancellation is intact — any firefly from a deleted denominator term would push this above $\pi$.

**Full suite state (verified):** `85 passed, 9 failed`. All 9 failures are the same `../samples/cbox/cbox.xml` lookup (`test_sampling.py`, `test_segments.py`) — an asset that lives in `ray-maps/samples/` on your machine but isn't in this sandbox. They are environment failures, not logic failures.

---

## 5. Diagnostic infrastructure (the `debug/` scripts)

The investigation was driven by purpose-built instruments rather than argument. Key ones:

- **`O_direct_calibration.py`** — injects + one gather vs PT $d{=}2$. Direct is blur-class and *heals* with resolution ($0.964@64^2 \to 0.991@128^2$).
- **`O2_v2_gather_calibration.py`** — calibrates the gather at the **second** vertex. Read $0.956$, structureless ratio map → the gather operator is **innocent** (location-neutral).
- **`O3_readout_ladder.py`** — the decisive instrument: buckets the $d{=}99$ readout **per level**, each calibrated against its own PT slice, with a per-pixel assert that buckets sum to the production readout. This is what isolated the dead-end tail bucket and (after a factor-3 normalization bug was caught) reconciled every instrument to the same number.
- **`O4_last_vertex.py`** — last-vertex decomposition.
- **`P_bounce_calibration.py`** — cumulative BLT/PT ladder by bounce. (Its per-bounce "marginal 1.39" turned out to be a ladder artifact; the cumulative numbers stand.)
- **`Q_reuse_correlation.py` / `Q2_hop_pools.py`** — testing whether per-hop heat compounds through pool reuse. Q's first design was flawed (arm B reused the same segments); Q2's **disjoint by-path hop-pools** (via `pool_mask` in `build_pair_cache`) confirmed $\sim10\%$/hop reuse compounding.
- **`M_firefly_mechanism.py`** — guard kills, parent share, $\rho(M)$ power iteration pre/post cap.
- **`N_bsdf_only_ref.py`** — equal-effort BSDF-only PT baseline.
- **`G_convergence.py`** — the main convergence harness; `--fg-depth`, `--alpha`, `--no-progressive`, RMSE/mean-ratio + diff panels.

**`debug/_cache.py`** is the reuse engine: per-iteration EXRs keyed on `(params, code-fingerprint)` where the fingerprint is a **content hash of `src/segments/*.py`** — touch the estimator and every entry built on old physics auto-invalidates. `pt_reference()` caches PT ground truths keyed on params only (the reference doesn't depend on estimator code). `--export KEY` averages an entry's iterations into one EXR you can feed straight to `F_ab_vs_reference_all.py`. This is what made the long sweeps painless and is itself one of the changes worth keeping.

---

## 6. Theories proposed and *falsified* by measurement

Worth recording because they shape what's left:

- **Guard-kills as the active firefly source** — count was 0 in the pool.
- **Per-gather geometry-conditional heat** — O/O2/O3 all read the single gather as calibrated; O2's ratio map is structureless.
- **P's "1.39 marginal bounce"** — a ladder-construction artifact; retired.
- **Reuse-correlation as the *whole* story** — Q2 confirmed it exists at $\sim10\%$/hop, but it only accounts for the small cumulative $\sim1.03$, not the $+7\%$ floor (that was the dead-end bug).
- **A dim/wrong reference EXR** — audited: `baseline/fresh = 0.9997`. Reference exonerated.
- **α (shrink rate) moving the cold floor** — it didn't (0.25/0.4/0.67 all $\approx0.97$); the effect **saturates**, which pointed at coverage loss, not blur rate.
- **Bounce truncation as the cold source** — bumping $N$ saturated by $N{=}16$ at $\sim+0.5\%$ and slightly *hot*, not the cold seen.

The surviving explanation for the residual $\approx0.97$ cold: **coverage loss** as the progressive kernel shrinks (some receivers' kernels go empty and read zero) plus a sliver from sub-kernel exclusion that classical links don't fully recover — a documented, asymptotically-vanishing approximation, not a bug.

---

## 7. Current state & open items

**Current (cornell, camera-only):** no fireflies, $\rho\approx0.5$, no drift, no splotches. Grain solved by NEE ($\sim27\times$ variance). Hot bias solved by the dead-end rule. Remaining floor is a small ($\sim$few %) **cold** density-estimation bias that shrinks with samples under the progressive schedule.

**Open items (in fidelity/leverage order):**

1. **The cold floor is a kernel-schedule tradeoff, not a bug.** Fixed kernel: $\approx+1.5\%$ hot but a hard floor. Progressive: $\approx-2.6\%$ cold but asymptotically unbiased. The decisive experiment is a **long run of both** ($\sim$512 iters, cache makes it incremental): progressive should climb toward 1.0 with slope $\approx-0.33$ while fixed plateaus — the Fig S2 crossover. This tells you which to ship.
2. **Light-technique repair** — the paper's actual bidirectional estimator, currently disabled. MMIS pairs are enumerated only via the $x$-side cluster, but the light technique generates segments from the $y$ side (the paper's **dual-cluster** point, Fig 3), so the true light predecessor never enters $p_\text{sum}$; emitter pos/dir pdfs are unused; light-terminal weight is pinned to 1.0. Must be repaired before re-enabling `add_light_samples`. Prerequisite for any scene harder than a Cornell box.
3. **Hop-pooled production propagation** — promote the `pool_mask` machinery from diagnostic into `iterate_propogation_vec` to kill the $\sim10\%$/hop reuse compounding. Roughly cost-neutral; hold until the long run says how much floor it actually buys.
4. **Documented approximations** — kernel-smoothing bias (progressive handles), residual reuse correlation (small), sub-kernel exclusion sliver.

---

## 8. File-by-file change map

| File | What changed |
|------|--------------|
| `primitives.py` | `MIN_SEG_LEN = 1e-8` single source of truth; `is_nee`/`nee_parea`/`direct_in` segment fields used downstream |
| `sampler_types.py` | guard-symmetric `conditional_pdf`; grazing guard removed; `nee_parea` added to denominator (MIS) |
| `mmis.py` | strict-positive `p_sum` (no $10^5$ floor); vectorized weight from pair cache |
| `pair_cache.py` | sub-kernel exclusion (`merge_min_len_factor`); classical links (`cls_*`); `<2` cutoff (was `<=5`); `pool_mask` for hop-pools; NEE/light-terminal receiver filtering |
| `propagation.py` | row-gain cap (energy projection); classical-link transport; `direct_in` split-gather output; geom clamp now off by default |
| `segment_gen_sampler.py` | `make_nee_segment` + per-vertex NEE injection via `nee_out`; `_area_pdf_of_light_hit` (avoids the MI 3.8 segfaulting ctor) |
| `renderer.py` | depth-$d$ split readout (`_readout_path`) **with the dead-end rule**; NEE threading (`add_nee_segments`); caller-owned progressive schedule; `final_gather_depth` |
| `cluster.py` | no longer self-increments `_iteration` (caller owns the schedule) |
| `tests/test_estimator_invariants.py` | new — the 9 invariant locks above |
| `tests/test_help.py` | patched for the corrected behavior |
| `debug/_cache.py` + O2/O3/O4/P/Q2/N + G | the diagnostic + caching infrastructure |

**Housekeeping noted in the chain:** `pre_sync_backup_20260612.tgz` in the project root holds every file overwritten during the Downloads→project sync (safe to delete once committed). Old `M_lookup_compare.py` / `N_radius_sweep.py` coexist with the new `M_`/`N_` scripts (delete if you want the letters unambiguous). **Nothing in this chain is committed** — git lives in the parent `ray-maps/` dir; this is the natural checkpoint.
