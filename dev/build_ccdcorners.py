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
import argparse, os
from multiprocessing import Pool
import numpy as np
import h5py, fitsio
import pixmappy as pm
import glob

DEFAULT_EXP = "/home2/vwetzell/gitrepos/pixmappy/pixmappy/data/delveExposures.hdf5"

ROW_DTYPE = [("expnum", "i4"), ("detpos", "U3"), ("ccdnum", "i2"),
             ("band", "U1"), ("mjd_mid", "f8"), ("ra", "f8", (5,)), ("dec", "f8", (5,))]

# per-worker globals, set by _init (avoids re-pickling the box every chunk)
_MAPS = _PX = _PY = _COLOR = None


def _init(px, py, color):
    global _MAPS, _PX, _PY, _COLOR
    _MAPS, _PX, _PY, _COLOR = pm.DelveMaps(), px, py, color


def _work(chunk):
    """chunk = (expnums, bands, mjds, boxed_ccds).

    Try every ccd we have a box for; a missing WCS (not all exposures solve every
    ccd) just skips that ccd. nGaia is NOT used to filter: some ccds have a valid
    WCS with nGaia==0, and we'd wrongly drop those.
    """
    expnums, bands, mjds, ccds = chunk
    rows, n_skip = [], 0
    for i in range(len(expnums)):
        expnum = int(expnums[i])
        band = bands[i].decode() if isinstance(bands[i], bytes) else str(bands[i])
        mjd = float(mjds[i])
        for ccd in ccds:
            ccd = int(ccd)
            try:
                wcs = _MAPS.getDelveWCS(expnum, ccd)
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
    p.add_argument("--color", type=float, default=pm.REF_COLOR,
                   help="color fed to toSky (default: neutral REF_COLOR)")
    p.add_argument("--nproc", type=int, default=os.cpu_count())
    args = p.parse_args()

    box = fitsio.read(args.box)
    px, py = {}, {}
    for r in box:
        xlo, xhi = np.floor(r["xmin"]), np.ceil(r["xmax"])   # bound the detections
        ylo, yhi = np.floor(r["ymin"]), np.ceil(r["ymax"])
        xc, yc = 0.5 * (xlo + xhi), 0.5 * (ylo + yhi)
        # winding: (lo,lo)->(hi,lo)->(hi,hi)->(lo,hi), center last
        px[int(r["ccdnum"])] = np.array([xlo, xhi, xhi, xlo, xc])
        py[int(r["ccdnum"])] = np.array([ylo, ylo, yhi, yhi, yc])
    ccds = np.array(sorted(px))  # ccdnums we have a box for

    with h5py.File(args.exposures, "r") as f:
        t = f["__astropy_table__"][: args.max]
    expnums, bands, mjds = t["expnum"], t["band"], t["mjdmid"]

    # contiguous chunks, a few per worker for load balance
    nchunk = max(args.nproc * 8, 1)
    idx = np.array_split(np.arange(len(expnums)), nchunk)
    chunks = [(expnums[s], bands[s], mjds[s], ccds) for s in idx if len(s)]

    parts, n_skip = [], 0
    with Pool(args.nproc, initializer=_init, initargs=(px, py, args.color)) as pool:
        for done, (arr, ns) in enumerate(pool.imap_unordered(_work, chunks), 1):
            parts.append(arr); n_skip += ns
            print(f"  {done}/{len(chunks)} chunks, {sum(len(a) for a in parts)} rows", flush=True)

    out = np.concatenate(parts)
    assert np.isfinite(out["ra"]).all() and np.isfinite(out["dec"]).all()
    fitsio.write(args.out, out, clobber=True)
    print(f"{len(t)} exposures -> {len(out)} rows ({n_skip} ccd skipped), wrote {args.out}")


if __name__ == "__main__":
    main()
