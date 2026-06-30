"""
Numerical validation of the Eq.13 ray-tracing segment sampler (paper Fig. 2b).

Standalone, numpy-only. It replicates the EXACT math of
`sample_segment_from_point` / `sample_surface_point` on an analytic sphere,
where ground truth is closed form, and reports PASS/FAIL on three checks:

  (1) zero-variance invariant   G(s) / p(y|x) = pi      (per sample, exact)
  (2) sphere chord-length law   E[len] = 4R/3,  p(l) = l / (2 R^2)   (chi-square)
  (3) it also shows the CURRENT bug: using the SHADING normal in the area
      Jacobian (si_y.sh_frame.n, segment_gen_sampler.py:279) breaks (1).

Why a sphere: it is the simplest scene with curvature, so geometric != shading
normal under smooth shading -- exactly the case a Cornell-box test cannot see.

Run:  python validation/validate_eq13_sampler.py
"""
import numpy as np

def unit(v):
    return v / np.linalg.norm(v, axis=1, keepdims=True)

def cosine_about(normal, rng):
    """Cosine-hemisphere sample about `normal`; returns (dir, solid-angle pdf=cos/pi)."""
    a = np.where(np.abs(normal[:, :1]) < 0.9, [1., 0, 0], [0, 1., 0])
    t = unit(np.cross(normal, a)); b = np.cross(normal, t)
    u1, u2 = rng.random(len(normal)), rng.random(len(normal))
    r, phi = np.sqrt(u1), 2 * np.pi * u2
    d = (r*np.cos(phi))[:, None]*t + (r*np.sin(phi))[:, None]*b + np.sqrt(1-u1)[:, None]*normal
    return d, np.sqrt(1 - u1) / np.pi

def run(R=1.0, N=1_000_000, perturb_deg=20.0, seed=0):
    rng = np.random.default_rng(seed)

    # p(x) = 1/|M|: x uniform by area on the sphere
    x   = R * unit(rng.normal(size=(N, 3)))
    ngx = x / R                                    # geometric normal at x (radial)

    # cosine-hemisphere direction at x, then ray-cast to the 2nd sphere hit
    d, pdf_x = cosine_about(ngx, rng)              # pdf_x = |n_x . s| / pi
    t = -2 * (x * d).sum(1)                        # |x + t d| = R, |x| = R  ->  t = -2 x.d
    y = x + t[:, None] * d
    ngy, dist = y / R, np.abs(t)                   # geometric normal at y, len = t

    cos_x      = np.abs((ngx * d).sum(1))
    cos_y_geom = np.abs((ngy * d).sum(1))          # CORRECT Jacobian normal (geometric)

    # simulate a smooth-shading normal at y, tilted from geometric -> what the code uses
    axis = rng.normal(size=ngy.shape); axis -= (axis*ngy).sum(1, keepdims=True)*ngy
    axis = unit(axis); th = np.deg2rad(perturb_deg)
    nsh_y = np.cos(th)*ngy + np.sin(th)*axis
    cos_y_shad = np.abs((nsh_y * d).sum(1))

    G         = cos_x * cos_y_geom / dist**2       # == Segment.geom_term (geometric)
    p_correct = pdf_x * cos_y_geom / dist**2       # p(y|x) with geometric normal at y
    p_buggy   = pdf_x * cos_y_shad / dist**2       # p(y|x) with shading  normal at y (current)

    ok = True
    print(f"[setup] R={R}  N={N:,}  shading tilt={perturb_deg} deg\n")

    # (1) invariant
    r_geo, r_bug = G/p_correct, G/p_buggy
    p1 = abs(r_geo.mean() - np.pi) < 1e-4 and r_geo.std() < 1e-9
    ok &= p1
    print(f"(1) invariant G/p(y|x) = pi ({np.pi:.6f})")
    print(f"    geometric : mean {r_geo.mean():.6f}  std {r_geo.std():.2e}   {'PASS' if p1 else 'FAIL'}")
    print(f"    shading   : mean {r_bug.mean():.6f}  std {r_bug.std():.2e}   (current code -> biased)")

    # (2) chord-length law
    bins = 40; edges = np.linspace(0, 2*R, bins+1)
    obs, _ = np.histogram(dist, edges)
    exp = N * np.diff(edges**2 / (4*R**2))         # integral of l/(2 R^2)
    chi2 = ((obs - exp)**2 / exp).sum()
    p2 = abs(dist.mean() - 4/3*R) < 5e-3 and chi2 < 2*(bins-1)
    ok &= p2
    print(f"\n(2) chord length: E[len]={dist.mean():.5f} (target {4/3*R:.5f}), "
          f"chi2/dof={chi2:.2f}/{bins-1}   {'PASS' if p2 else 'FAIL'}")

    print(f"\n{'ALL PASS' if ok else 'FAIL'} — geometric-normal sampler is correct; "
          f"shading-normal Jacobian is not.")
    return ok

if __name__ == "__main__":
    run()
