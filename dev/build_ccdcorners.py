#!/usr/bin/env python
"""Build a y6a1.ccdcorners-style FITS from per-CCD pixel boxes + pixmappy WCS.

Driven by pixmappy's delveExposures.hdf5: for every exposure therein, transforms
each solved CCD's 4 corners + center (from the per-ccd pixel box) through that
exposure/ccd's DelveMaps WCS into sky coords. One output row per (expnum, ccdnum),
matching y6a1.ccdcorners.fits.gz: expnum, detpos, ccdnum, band, mjd_mid, ra[5], dec[5].
So the result pairs one-to-one with the exposure table it was built from.

Box mins are floored and maxs ceiled, so corners bound the measured detections.
Work is split across processes over exposure chunks (WCS solve is the cost).
"""

import argparse
import os
from multiprocessing import Pool

import fitsio
import h5py
import numpy as np
import pixmappy as pm

DEFAULT_EXP = "/home/vwetzell/gitrepos/pixmappy/pixmappy/data/delveExposures.hdf5"

ROW_DTYPE = [
    ("expnum", "i4"),
    ("detpos", "U3"),
    ("ccdnum", "i2"),
    ("band", "U1"),
    ("mjd_mid", "f8"),
    ("ra", "f8", (5,)),
    ("dec", "f8", (5,)),
]

# per-worker globals, set by _init: one DelveMaps per worker (its init reads a
# 115MB table, so building one per chunk would thrash). getDelveWCS caches every
# exposure's maps into the collection forever, so we snapshot the base maps and
# prune back to it each chunk -> bounded memory, no re-reads.
_MAPS = _BASE = _PX = _PY = _COLOR = None
_MAP_DICTS = ("root", "wcs", "realizedMap", "realizedWCS")


def _init(px, py, color):
    global _MAPS, _BASE, _PX, _PY, _COLOR
    _MAPS = pm.DelveMaps()
    # exptab loads as strided views into one structured array; searchsorted on a
    # non-contiguous column copies it (and touches a cache line per element,
    # ~17MB traffic) on EVERY getDelveWCS call -> saturates memory bandwidth
    # across workers. Make the binary-searched column contiguous once.
    _MAPS.exptab["expnum"] = np.ascontiguousarray(_MAPS.exptab["expnum"])
    _BASE = {k: set(getattr(_MAPS, k)) for k in _MAP_DICTS}
    _PX, _PY, _COLOR = px, py, color


def _prune():
    for k, base in _BASE.items():
        d = getattr(_MAPS, k)
        for key in list(d):
            if key not in base:
                del d[key]


def _work(chunk):
    """chunk = (expnums, bands, mjds, boxed_ccds).

    Try every ccd we have a box for; a missing WCS (not all exposures solve every
    ccd) just skips that ccd. nGaia is NOT used to filter: some ccds have a valid
    WCS with nGaia==0, and we'd wrongly drop those.
    """
    expnums, bands, mjds, ccds = chunk
    maps = _MAPS
    _prune()  # drop the previous chunk's accumulated maps
    # Match exptab's expnum dtype: pixmappy does np.searchsorted(exptab['expnum'],
    # expnum) twice per (exp, ccd); with a python int that promotes + re-casts the
    # whole 271k-row column EVERY call (~0.3 ms + memory bandwidth), with a
    # same-dtype scalar it's a plain binary search (~1 us).
    expnums = expnums.astype(maps.exptab["expnum"].dtype)
    rows, n_skip = [], 0
    for i in range(len(expnums)):
        expnum = expnums[i]
        band = bands[i].decode() if isinstance(bands[i], bytes) else str(bands[i])
        mjd = float(mjds[i])
        for ccd in ccds:
            ccd = int(ccd)
            try:
                wcs = maps.getDelveWCS(expnum, ccd)
                ra, dec = wcs.toSky(_PX[ccd], _PY[ccd], c=_COLOR)
            except Exception:
                n_skip += 1
                continue
            rows.append((expnum, pm.ccdnum2detpos[ccd], ccd, band, mjd, ra, dec))
    return np.array(rows, dtype=ROW_DTYPE), n_skip


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("box", help="per-ccd box FITS from ccd_edge_extent.py --out")
    p.add_argument("out", help="output FITS")
    p.add_argument("--exposures", default=DEFAULT_EXP, help="delveExposures.hdf5")
    p.add_argument("--max", type=int, default=None, help="max exposures")
    p.add_argument(
        "--color",
        type=float,
        default=pm.REF_COLOR,
        help="color fed to toSky (default: neutral REF_COLOR)",
    )
    # Scales ~linearly to physical cores now that the per-call strided-column
    # copies are gone; SMT adds nothing (24 procs ~= 16 on a 16-core box).
    p.add_argument("--nproc", type=int, default=min(16, os.cpu_count()))
    p.add_argument(
        "--chunk",
        type=int,
        default=50,
        help="exposures between map-collection prunes (bounds memory and "
        "the O(chunk) set-build inside PixelMapCollection.update)",
    )
    args = p.parse_args()

    box = fitsio.read(args.box)
    px, py = {}, {}
    for r in box:
        xlo, xhi = np.floor(r["xmin"]), np.ceil(r["xmax"])  # bound the detections
        ylo, yhi = np.floor(r["ymin"]), np.ceil(r["ymax"])
        xc, yc = 0.5 * (xlo + xhi), 0.5 * (ylo + yhi)
        # winding: (lo,lo)->(hi,lo)->(hi,hi)->(lo,hi), center last
        px[int(r["ccdnum"])] = np.array([xlo, xhi, xhi, xlo, xc])
        py[int(r["ccdnum"])] = np.array([ylo, ylo, yhi, yhi, yc])
    ccds = np.array(sorted(px))  # ccdnums we have a box for

    with h5py.File(args.exposures, "r") as f:
        t = f["__astropy_table__"][: args.max]
    expnums, bands, mjds = t["expnum"], t["band"], t["mjdmid"]

    # small fixed-size chunks: bounds each fresh DelveMaps + load-balances
    idx = np.arange(len(expnums))
    chunks = [
        (expnums[s], bands[s], mjds[s], ccds)
        for s in np.array_split(idx, max(len(idx) // args.chunk, 1))
    ]

    parts, n_skip = [], 0
    with Pool(args.nproc, initializer=_init, initargs=(px, py, args.color)) as pool:
        for done, (arr, ns) in enumerate(pool.imap_unordered(_work, chunks), 1):
            parts.append(arr)
            n_skip += ns
            print(
                f"  {done}/{len(chunks)} chunks, {sum(len(a) for a in parts)} rows",
                flush=True,
            )

    out = np.concatenate(parts)
    assert np.isfinite(out["ra"]).all() and np.isfinite(out["dec"]).all()
    fitsio.write(args.out, out, clobber=True)
    print(
        f"{len(t)} exposures -> {len(out)} rows ({n_skip} ccd skipped), wrote {args.out}"
    )


if __name__ == "__main__":
    main()
