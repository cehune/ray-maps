"""
_cache.py — shared per-iteration render cache for debug scripts.

Any debug script that renders BLT iterations can persist them here and any
OTHER debug script can reuse them, as long as the render is determined by the
same (params, code) pair:

  * params : every knob that affects the image — width, c, N, beta, alpha,
             progressive, clamps, seeds are implicit (sampler.seed(i, W*H)).
  * code   : a content hash of src/segments/*.py. Touch the estimator and
             every cache entry built on the old physics is automatically
             invalidated — debug conclusions must never come from stale code.

Layout (debug/_cache/<key>/):
    meta.json        params, code fingerprint, tag, per-iteration stats
    iter_0000.exr    float32 EXR of each iteration's contribution image

Typical use:

    from _cache import RenderCache
    cache = RenderCache(params=dict(W=128, H=128, c=30, N=8, alpha=0.25,
                                    progressive=True, geom_clamp=None,
                                    beta=0.1, light=False, mode="vec"),
                        tag="G_convergence")
    for i in range(n_iter):
        img = cache.load_iter(i)
        if img is None:
            img = <render iteration i>
            cache.save_iter(i, img, stats={"mean": float(img.mean())},
                            shared={"voxel_size_0": cluster._voxel_size_0})
        accum += img

`shared` entries are merged into meta once and can be read back via
cache.shared() — use this for cross-iteration state (e.g. voxel_size_0) so a
resumed run reproduces the progressive kernel schedule exactly.

CLI:  python _cache.py            list entries
      python _cache.py --purge    delete stale entries (code mismatch)
      python _cache.py --purge-all
"""
import hashlib
import json
import os
import shutil

import numpy as np

DEBUG_DIR  = os.path.dirname(os.path.abspath(__file__))
CACHE_ROOT = os.path.join(DEBUG_DIR, "_cache")
SRC_DIR    = os.path.normpath(os.path.join(DEBUG_DIR, "..", "src", "segments"))

_CODE_FILES = [
    "primitives.py", "segment_gen_sampler.py", "sampler_types.py",
    "cluster.py", "mmis.py", "pair_cache.py", "propagation.py", "renderer.py",
]


def code_fingerprint() -> str:
    """Content hash of the estimator source — stale physics never validates."""
    h = hashlib.sha256()
    for name in _CODE_FILES:
        p = os.path.join(SRC_DIR, name)
        if os.path.exists(p):
            with open(p, "rb") as f:
                h.update(name.encode())
                h.update(f.read())
    return h.hexdigest()[:16]


def _canonical(params: dict) -> str:
    return json.dumps(params, sort_keys=True, default=str)


class RenderCache:
    def __init__(self, params: dict, tag: str = "render", enabled: bool = True):
        self.params = dict(params)
        self.tag = tag
        self.enabled = enabled
        self.fingerprint = code_fingerprint()
        blob = json.dumps({"params": self.params, "code": self.fingerprint},
                          sort_keys=True, default=str)
        self.key = hashlib.sha256(blob.encode()).hexdigest()[:20]
        self.dir = os.path.join(CACHE_ROOT, self.key)
        self._meta = None
        if enabled:
            os.makedirs(self.dir, exist_ok=True)
            self._meta = self._load_meta()
            n = self.n_cached()
            print(f"[cache] key={self.key} tag={tag} "
                  f"({n} iteration(s) cached, code={self.fingerprint})")

    # ── meta ──────────────────────────────────────────────────────────────────
    def _meta_path(self):
        return os.path.join(self.dir, "meta.json")

    def _load_meta(self) -> dict:
        p = self._meta_path()
        if os.path.exists(p):
            try:
                with open(p) as f:
                    m = json.load(f)
                if m.get("code") == self.fingerprint and \
                   _canonical(m.get("params", {})) == _canonical(self.params):
                    return m
            except (json.JSONDecodeError, OSError):
                pass
        return {"params": self.params, "code": self.fingerprint,
                "tag": self.tag, "iters": {}, "shared": {}}

    def _save_meta(self):
        tmp = self._meta_path() + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self._meta, f, indent=1, sort_keys=True, default=str)
        os.replace(tmp, self._meta_path())

    # ── iterations ────────────────────────────────────────────────────────────
    def _iter_path(self, i: int) -> str:
        return os.path.join(self.dir, f"iter_{i:04d}.exr")

    def load_iter(self, i: int):
        """Return the cached (H, W, 3) float image for iteration i, or None."""
        if not self.enabled:
            return None
        p = self._iter_path(i)
        if not os.path.exists(p) or str(i) not in self._meta["iters"]:
            return None
        import mitsuba as mi
        img = np.array(mi.Bitmap(p))[..., :3].astype(np.float64)
        print(f"[cache] iter {i}: loaded from cache")
        return img

    def save_iter(self, i: int, img, stats: dict | None = None,
                  shared: dict | None = None):
        if not self.enabled:
            return
        import mitsuba as mi
        arr = np.ascontiguousarray(np.asarray(img, dtype=np.float32))
        mi.Bitmap(arr).write(self._iter_path(i))
        self._meta["iters"][str(i)] = stats or {}
        if shared:
            self._meta["shared"].update(
                {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                 for k, v in shared.items()})
        self._save_meta()

    def iter_stats(self, i: int) -> dict | None:
        return None if not self.enabled else self._meta["iters"].get(str(i))

    def shared(self) -> dict:
        return {} if not self.enabled else dict(self._meta.get("shared", {}))

    def n_cached(self) -> int:
        """Number of CONTIGUOUS cached iterations starting at 0."""
        if not self.enabled:
            return 0
        n = 0
        while str(n) in self._meta["iters"] and os.path.exists(self._iter_path(n)):
            n += 1
        return n


# ── PT reference cache ────────────────────────────────────────────────────────
REF_DIR = os.path.join(CACHE_ROOT, "_refs")


def pt_reference(W: int, H: int, max_depth: int, spp: int,
                 refresh: bool = False) -> np.ndarray:
    """Cached mitsuba path-integrator render (the ground-truth side of O/P/N
    style comparisons). Keyed by params ONLY — deliberately NOT by the
    src/segments code hash, because the PT reference doesn't depend on our
    estimator code and re-rendering 256-512 spp on every code edit is exactly
    the cost this cache exists to kill.

    Returns (H, W, 3) float64.
    """
    import mitsuba as mi
    os.makedirs(REF_DIR, exist_ok=True)
    p = os.path.join(REF_DIR, f"ptref_w{W}_h{H}_d{max_depth}_spp{spp}.exr")
    if not refresh and os.path.exists(p):
        print(f"[cache] PT ref d={max_depth} spp={spp}: loaded ({os.path.basename(p)})")
        return np.array(mi.Bitmap(p))[..., :3].astype(np.float64)
    d = mi.cornell_box()
    d["sensor"]["film"]["width"] = W
    d["sensor"]["film"]["height"] = H
    d["integrator"] = {"type": "path", "max_depth": max_depth}
    print(f"[cache] PT ref d={max_depth} spp={spp}: rendering...")
    img = np.array(mi.render(mi.load_dict(d), spp=spp))[..., :3].astype(np.float64)
    mi.Bitmap(np.ascontiguousarray(img.astype(np.float32))).write(p)
    return img


# ── CLI: list / purge ─────────────────────────────────────────────────────────
def _list_entries():
    if not os.path.isdir(CACHE_ROOT):
        print("cache empty")
        return []
    fp = code_fingerprint()
    rows = []
    for key in sorted(os.listdir(CACHE_ROOT)):
        mp = os.path.join(CACHE_ROOT, key, "meta.json")
        if not os.path.exists(mp):
            continue
        try:
            with open(mp) as f:
                m = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        stale = m.get("code") != fp
        rows.append((key, m.get("tag", "?"), len(m.get("iters", {})), stale, m))
    return rows


def export_mean(key: str, out: str | None = None, n: int | None = None):
    """Average the contiguous cached iterations of `key` into one EXR (+ a
    tonemapped PNG preview). The EXR can be fed straight to
    F_ab_vs_reference_all.py --blt-path for an A/B against the reference."""
    import mitsuba as mi
    mi.set_variant("scalar_rgb")
    d = os.path.join(CACHE_ROOT, key)
    if not os.path.isdir(d):
        raise SystemExit(f"no cache entry {key}")
    imgs = []
    i = 0
    while os.path.exists(os.path.join(d, f"iter_{i:04d}.exr")):
        if n is not None and i >= n:
            break
        imgs.append(np.array(mi.Bitmap(os.path.join(d, f"iter_{i:04d}.exr")))[..., :3]
                    .astype(np.float64))
        i += 1
    if not imgs:
        raise SystemExit(f"entry {key} has no cached iterations")
    mean = np.mean(imgs, axis=0)

    out = out or os.path.join(DEBUG_DIR, f"cache_{key[:8]}_mean{len(imgs)}.exr")
    mi.Bitmap(np.ascontiguousarray(mean.astype(np.float32))).write(out)

    # tonemapped preview (same Reinhard pipeline as the F/K scripts)
    lum = mean.sum(-1)
    med = float(np.median(lum[lum > 0])) if (lum > 0).any() else 1.0
    exposure = 0.18 / med if med > 0 else 1.0
    scaled = mean * exposure
    png = np.clip((scaled / (1.0 + scaled)) ** (1 / 2.2), 0, 1).astype(np.float32)
    png_path = out.rsplit(".", 1)[0] + ".png"
    import matplotlib.pyplot as plt
    plt.imsave(png_path, png)
    print(f"averaged {len(imgs)} iterations\n  exr: {out}\n  png: {png_path}")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--purge", action="store_true", help="delete stale (code-mismatch) entries")
    ap.add_argument("--purge-all", action="store_true")
    ap.add_argument("--export", metavar="KEY",
                    help="average a cache entry's iterations into one EXR + PNG")
    ap.add_argument("--out", default=None, help="output path for --export")
    ap.add_argument("--n", type=int, default=None,
                    help="only average the first N iterations (e.g. compare 8 vs 48)")
    args = ap.parse_args()

    if args.export:
        export_mean(args.export, args.out, args.n)
        raise SystemExit(0)

    rows = _list_entries()
    for key, tag, n, stale, m in rows:
        mark = "STALE" if stale else "ok   "
        p = m.get("params", {})
        brief = ", ".join(f"{k}={p[k]}" for k in sorted(p)[:6])
        print(f"  {mark} {key}  [{tag}]  {n} iters   {brief}")
    if args.purge or args.purge_all:
        n_del = 0
        for key, tag, n, stale, m in rows:
            if args.purge_all or stale:
                shutil.rmtree(os.path.join(CACHE_ROOT, key), ignore_errors=True)
                n_del += 1
        print(f"deleted {n_del} entr{'y' if n_del == 1 else 'ies'}")
