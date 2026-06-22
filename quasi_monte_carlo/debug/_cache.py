"""
_cache.py — per-render cache for the QMC debug scripts.

Adapted from segment_blt/debug/_cache.py. A render is determined by the
(params, code) pair:

  * params : every knob that affects the image — scene, W, H, sampler, max_dim
             (= #Sobol dimensions), scramble, seed, spp, max_depth, rr_depth.
  * code   : a content hash of the QMC sampler + integrator source. Touch the
             sampler/integrator and every entry built on the old code is
             automatically invalidated — a convergence conclusion must never be
             read off a stale render.

We cache ONE image per config (stored as iter 0), so a config that already has
its EXR on disk is never re-rendered — that's the whole point: feed a reference
EXR, sweep, and never pay for the same (sampler, dim, spp, seed) twice.

Layout (debug/_cache/<key>/):
    meta.json        params, code fingerprint, tag, stats
    iter_0000.exr    float32 EXR of the render

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
SRC_DIR    = os.path.normpath(os.path.join(DEBUG_DIR, "..", "src", "qmc"))

# Files whose contents change the rendered image. Sampler internals + the
# scalar driver/integrator. Direction-number table loader included because it
# defines the Sobol' points themselves.
_CODE_FILES = [
    "renderer.py",
    "pathtracer.py",
    os.path.join("sampler", "sobol_core.py"),
    os.path.join("sampler", "sobol_sampler.py"),
    os.path.join("sampler", "padded_sobol_sampler.py"),
    os.path.join("sampler", "independent_sampler.py"),
    os.path.join("sampler", "scramble.py"),
    os.path.join("sampler", "direction_numbers.py"),
    os.path.join("sampler", "constants.py"),
    "setup.py",
]


def code_fingerprint() -> str:
    """Content hash of the sampler/integrator source — stale code never validates."""
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
                "tag": self.tag, "iters": {}}

    def _save_meta(self):
        tmp = self._meta_path() + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self._meta, f, indent=1, sort_keys=True, default=str)
        os.replace(tmp, self._meta_path())

    # ── the single cached render (stored as iter 0) ─────────────────────────────
    def _iter_path(self, i: int) -> str:
        return os.path.join(self.dir, f"iter_{i:04d}.exr")

    def load(self):
        """Return the cached (H, W, 3) float64 image, or None."""
        if not self.enabled:
            return None
        p = self._iter_path(0)
        if not os.path.exists(p) or "0" not in self._meta["iters"]:
            return None
        import mitsuba as mi
        return np.array(mi.Bitmap(p))[..., :3].astype(np.float64)

    def save(self, img, stats: dict | None = None):
        if not self.enabled:
            return
        import mitsuba as mi
        arr = np.ascontiguousarray(np.asarray(img, dtype=np.float32))
        mi.Bitmap(arr).write(self._iter_path(0))
        # sibling PNG for eyeballing — same simple gamma encode baseline.py uses
        # for its per-spp previews, so test renders are directly comparable to
        # baseline/spp_renders*/<spp>.png.
        png = np.clip(np.maximum(arr, 0.0) ** (1 / 2.2), 0.0, 1.0)
        mi.Bitmap(mi.Bitmap(png).convert(
            mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8, srgb_gamma=False)
        ).write(self._iter_path(0).replace(".exr", ".png"))
        self._meta["iters"]["0"] = stats or {}
        self._save_meta()


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


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--purge", action="store_true", help="delete stale (code-mismatch) entries")
    ap.add_argument("--purge-all", action="store_true")
    args = ap.parse_args()

    rows = _list_entries()
    for key, tag, n, stale, m in rows:
        mark = "STALE" if stale else "ok   "
        p = m.get("params", {})
        brief = ", ".join(f"{k}={p[k]}" for k in sorted(p)[:8])
        print(f"  {mark} {key}  [{tag}]   {brief}")
    if args.purge or args.purge_all:
        n_del = 0
        for key, tag, n, stale, m in rows:
            if args.purge_all or stale:
                shutil.rmtree(os.path.join(CACHE_ROOT, key), ignore_errors=True)
                n_del += 1
        print(f"deleted {n_del} entr{'y' if n_del == 1 else 'ies'}")
