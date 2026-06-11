"""
Render cache for the debug harness.

Problem this solves: the render-producing scripts (J, G, K, L, M, N, ...) re-run
the full BLT integration every time — even when you only want to re-plot, tweak a
metric, or feed the result into a comparison script like F. This module persists
each finished render as a linear float EXR plus a small JSON sidecar, so
downstream scripts can LOAD instead of RE-RENDER.

Layout (gitignored):
    debug/_renders/<tag>.exr    — linear HxWx3 float image (reference-independent)
    debug/_renders/<tag>.json   — config + metric curves (reference-dependent)

A `tag` must encode every knob that changes the *image* (resolution, iters, c, N,
alpha, clamp, lookup, ...). It must NOT depend on the reference: the EXR is the raw
render, so any metric vs any ref can be recomputed from it. Metric curves that DO
depend on a ref live in the JSON sidecar (with the ref's name recorded).

Usage:
    from _cache import save_render, load_render

    tag = f"J_w{W}_i{n}_c{c}_N{N}_a{alpha}"
    hit = load_render(tag)                  # None on miss
    if hit is not None:
        img, meta = hit["image"], hit["meta"]
    else:
        img = ...render...
        save_render(tag, img, meta={...})

The EXR is exactly what F's `--blt-path` expects, so a J/K/L/M/N render can be fed
straight into F without re-rendering:
    python F_ab_vs_reference_all.py --blt-path _renders/<tag>.exr
"""
import os
import json
import numpy as np
import mitsuba as mi

CACHE_DIR = os.path.join(os.path.dirname(__file__), "_renders")


def cache_path(tag: str, ext: str = "exr") -> str:
    return os.path.join(CACHE_DIR, f"{tag}.{ext}")


def exists(tag: str) -> bool:
    return os.path.exists(cache_path(tag, "exr"))


def save_render(tag: str, img, meta: dict | None = None) -> str:
    """Write img (HxWx3 float) to <tag>.exr and optional meta dict to <tag>.json."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    img = np.ascontiguousarray(np.asarray(img, dtype=np.float32))
    # NB: mi.util.write_bitmap is asynchronous — the file may not be flushed
    # before a subsequent read. mi.Bitmap(img).write() is synchronous, so a
    # save immediately followed by load_render in the same process is safe.
    mi.Bitmap(img).write(cache_path(tag, "exr"))
    if meta is not None:
        with open(cache_path(tag, "json"), "w") as f:
            json.dump(meta, f, indent=2)
    return cache_path(tag, "exr")


def load_render(tag: str) -> dict | None:
    """
    Return {'image': HxWx3 float64, 'meta': dict|None} or None if no EXR is cached.
    `tag` may also be a direct path to an .exr (so callers can accept either).
    """
    path = tag if tag.endswith(".exr") else cache_path(tag, "exr")
    if not os.path.exists(path):
        return None
    img = np.array(mi.Bitmap(path))[..., :3].astype(np.float64)
    meta = None
    meta_path = path[:-4] + ".json" if tag.endswith(".exr") else cache_path(tag, "json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    return {"image": img, "meta": meta}
