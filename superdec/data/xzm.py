# Vendored verbatim from the main-repo platform (src/vt3dlearn/data/artifact_io.py) so the standalone
# reads the SAME .xz surface clouds the platform trains on (XZM-1 text archive: one obj_k_surface_pts
# (4096,3) block per object). The platform vendored all of superdec as Task 8; this is the same reader
# round-tripping back to the standalone (the dataloader-side backport).

"""Vendored, byte-compatible reader for videoto3d xzm-1 archives (generator-side numeric GT).

Pinned to the wire format of ``hermes-video-to-3d/videoto3d/artifact_io.py`` (generator 0.2.0,
commit "Artifact format overhaul"). This is a READER-ONLY copy: the generator repo's
``artifact_io.save_array_archive`` (single lzma stream over plain decimal text -- the
Symmetry_Dataset ``save_benchmark_shape(file_fmt='xz')`` pattern, i.e. ``np.savetxt`` inside
``lzma.open(path,'wb')``) is the sole writer; the learning platform never writes .xzm files, so
no writer copy is vendored here (two writers would be a drift trap by design).

WIRE FORMAT (``xzm-1``), post-decompression -- UTF-8 text, LF line endings::

    <XZM-1>
    @key depth_m dtype=float32 shape=32,32 fmt=%.9g
    -1.5,0.0,nan,...          # CSV rows, delimiter ',', raveled C-order == one array row per line
    ...
    @key inst dtype=uint8 shape=32,32 fmt=%d
    0,0,1,...
    @key names type=str n=5
    obj0_box                   # exactly `n` following lines, verbatim (generated artifact only)

Rules (identical to the generator side -- do NOT edit one without the other):
* Header line EXACTLY ``<XZM-1>`` first. Every section starts with ONE line beginning ``@key ``.
* Numeric sections: ``dtype=`` a numpy dtype (float32/float64/int*/uint*), ``shape=`` comma ints
  (literal ``x`` for a 0-d scalar), ``fmt=`` the C format used for every cell. Zero-sized section
  emits no data rows.
* Float formats are round-trip exact by construction (f4 -> %.9g, f8 -> %.17g); ints exact via %d;
  NaN/inf round-trip as their token.
* One data row per first-axis index (C-order ravel of the tail); 1 text row/array row.
* ``type=str`` sections carry verbatim line rows; key names match [A-Za-z0-9_-]+.
* ONE lzma stream (FORMAT_XZ) over the whole document.
* Loud on any malformed header / truncated section / unknown tag (ArtifactIOError, never guess).

The generator-side writer re-reads every save in place before publishing (element-equal check); this
reader is the platform's ONLY entry point to generator archives (see loader helpers in
``data/gt_archives.py`` -- never call np.load on a generator artifact from anywhere else in this
package).
"""
from __future__ import annotations

import lzma
import os
import re as _re
from functools import lru_cache
from typing import Any, Dict

import numpy as np  # type: ignore[import-not-found]

FORMAT_TAG = "<XZM-1>"
_KEY_RE = _re.compile(r"^[A-Za-z0-9_-]+$")

#: C text formats per supported numeric dtype (pinned; MUST match the generator's table exactly).
_FLOAT_FMT = {np.dtype("float32"): "%.9g", np.dtype("float64"): "%.17g"}


class ArtifactIOError(ValueError):
    """Malformed or incomplete .xzm stream (loud at conversion/read time -- never mid-training downstream)."""


def _raise(msg: str) -> ArtifactIOError:
    return ArtifactIOError(msg)


def _fmt_for_dt(dt: np.dtype) -> str:
    if dt.kind == "f":
        if dt.itemsize in (4, 8):
            return _FLOAT_FMT[dt]
        raise _raise(f"unsupported float width {dt} (only float32/float64 supported in xzm archives)")
    if dt.kind in "iu":
        return "%d"
    raise _raise(f"dtype {dt} outside the supported xzm numeric set (want f4/f8/i*/u*)")


def _norm_numeric_dtype(dt: Any) -> np.dtype:
    try:
        out = np.dtype(dt)
    except Exception as e:  # noqa: BLE001
        raise _raise(f"unparseable dtype {dt!r}") from e
    if out.kind not in "fiu":
        raise _raise(f"dtype {out!r} is not a supported xzm numeric dtype (want f4/f8/i*/u* or str sections)")
    return out


# --------------------------------------------------------------------------- #
# PATH RESOLUTION (xzm-primary; legacy .npz -> loud migration pointer, never read)      #
# --------------------------------------------------------------------------- #

def generator_archive_path(bundle_dir: str, subdir: str, stem: str) -> str:
    """Resolve ONE generator GT archive of the platform to its current on-disk path.

    Standard ``.xz`` is the canonical extension (videoto3d >= 0.3.0); the retired ``.xzm`` spelling on
    pre-0.3.0 bundles is byte-identical and still accepted until the rename migration runs. If only a
    legacy .npz exists (pre-0.2.0), the error names the videoto3d migration script -- a mixed-era fleet
    must fail LOUD at cache-build time, never drift mid-epoch."""
    for ext in (".xz", ".xzm"):
        cand = os.path.join(str(bundle_dir), subdir, stem + ext)
        if os.path.isfile(cand):
            return cand
    npz_p = os.path.join(str(bundle_dir), subdir, stem + ".npz")
    if os.path.isfile(npz_p):
        raise ArtifactIOError(
            f"{npz_p} is a LEGACY pre-0.2.0 generator artifact (.npz). Migrate the fleet first -- "
            f"hermes-video-to-3d/scripts/convert_legacy_npz_to_xzm.py <fleet_dir> -- then delete this "
            f"bundle's vt3learn_cache/ and re-run (the cache stamps the archive generation it saw).")
    raise FileNotFoundError(f"no generator archive {stem}{{.xz,.xzm}} under {os.path.join(str(bundle_dir), subdir)}")


# --------------------------------------------------------------------------- #
# READ                                                                               #
# --------------------------------------------------------------------------- #

def load_array_archive(path) -> Dict[str, np.ndarray]:
    """Parse ONE .xzm stream back into a dict of named arrays (exact inverse of the generator's
    ``save_array_archive``).

    Numeric sections come back with the SAME dtype recorded in their section header (round-trip
    exact: f4 via ``%.9g``, f64 via ``%.17g``, ints exact; NaN bg depth round-trips as NaN).
    ``type=str`` sections come back as object arrays holding the verbatim rows. Raises
    :class:`ArtifactIOError` on any malformed header, truncated section or unknown tag -- readers
    must LOUD-fail, never guess."""
    with lzma.open(str(path), "rb") as f:
        text = f.read().decode("utf-8")
    lines = text.split("\n")
    # The pinned writer ends the document with exactly ONE trailing newline; split() turns that into a
    # final "" element that is NOT a data row. Drop it so a truncated stream can't be rescued by the phantom blank.
    if lines and lines[-1] == "":
        lines = lines[:-1]
    if not lines or lines[0] != FORMAT_TAG:
        raise _raise(f"not an xzm-1 archive: first line must be {FORMAT_TAG!r}; got {lines[0] if lines else '<empty file>'!r}")

    out: Dict[str, np.ndarray] = {}
    i, n_lines = 1, len(lines)
    while i < n_lines:
        line = lines[i]
        if line == "":
            i += 1                                         # trailing blank line(s) after the last data row
            continue
        if not line.startswith("@key "):
            raise _raise(f"malformed xzm: expected a '@key ' section header, got {line!r}")
        head = line[len("@key "):]
        toks = head.split(" ")
        name = toks[0]
        if not _KEY_RE.match(name):
            raise _raise(f"malformed xzm key token {name!r}")
        meta = {}
        for t in toks[1:]:
            if "=" not in t:
                raise _raise(f"malformed xzm header token {t!r} in key {name!r}")
            kk, vv = t.split("=", 1)
            meta[kk] = vv
        i += 1

        if meta.get("type") == "str":
            n_rows = int(meta.get("n", "-1"))
            if n_rows < 0:
                raise _raise(f"xzm str section {name!r}: missing/invalid n=<int>")
            rows = lines[i:i + n_rows]
            if len(rows) != n_rows:
                raise _raise(f"xzm str section {name!r}: truncated (needs {n_rows} rows, {len(rows)} left)")
            out[name] = np.array(rows, dtype=object)
            i += n_rows
            continue

        dt = _norm_numeric_dtype(meta.get("dtype", ""))
        fmt = meta.get("fmt")
        if fmt != _fmt_for_dt(dt):
            raise _raise(f"xzm section {name!r}: fmt {fmt!r} does not match its dtype {dt} (format pin broken)")
        shape_txt = meta.get("shape", "x").strip()     # 'x' means 0-d scalar -- kept non-empty so presence is explicit
        if shape_txt == "x":
            shape = ()
        else:
            try:
                shape = tuple(int(x) for x in shape_txt.split(","))
            except ValueError as e:
                raise _raise(f"xzm section {name!r}: unparsable shape {shape_txt!r}") from e
        n_vals = int(np.prod(shape)) if shape else 1
        n_rows = shape[0] if shape else 1                 # one text row per array row (0d -> single row)

        data_rows = lines[i:i + n_rows]
        i += n_rows
        if len(data_rows) < n_rows:
            raise _raise(f"xzm section {name!r}: truncated numeric data ({len(data_rows)}/{n_rows} rows)")
        if n_vals > 0:
            vals = [v for row in data_rows for v in row.split(",")]
            if len(vals) != n_vals:
                raise _raise(f"xzm section {name!r}: expected {n_vals} cells, got {len(vals)}")
            arr_vals = np.array(vals, dtype=dt)
            out[name] = arr_vals.reshape(shape)
        else:
            out[name] = np.zeros(shape, dtype=dt)
    return out


# --------------------------------------------------------------------------- #
# MEMOIZED READ (hot per-frame path)                                           #
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=1024)
def _load_array_archive_cached(path: str, mtime_ns: int, size: int) -> tuple:
    return tuple((k, np.ascontiguousarray(v)) for k, v in load_array_archive(path).items())


def load_array_archive_cached(path) -> Dict[str, np.ndarray]:
    """``load_array_archive`` with a process-local memo keyed on (path, mtime, size) -- the hot per-sample read
    path (dataset.__getitem__ touches one pixel archive per sample; a DDP worker re-treads the same clip's frames
    for epochs).

    Contract: returned arrays are SHARED read-only by reference. The platform never mutates generator GT in place
    (every consumer copies on dtype change / resize / tensor conversion -- see dataset._frame_gt_d); do NOT assign
    into them. A changed file (regeneration) has a new (mtime,size) key, so stale content can never alias; the
    memo is bounded (LRU 1024 files, ~few GB worst case at legacy 704x384 resolution, far less at 32x32)."""
    st = os.stat(str(path))
    d = _load_array_archive_cached(str(path), int(st.st_mtime_ns), int(st.st_size))
    return {k: v for k, v in d}
