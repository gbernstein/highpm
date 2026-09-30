"""
Extract a compact bright-star reference from the DES Y6 Gold / DELVE DR3 gold skims,
for per-exposure photometric zeropoints (scripts/measure_exposure_depth.py).

One output file per nside-32 (RING) gold pixel: point sources (EXT_MASH 0-1,
FLAGS_GOLD == 0) with PSF_MAG_APER_8 between BRIGHT_MIN and BRIGHT_MAX in at least
one of g/r/i/z, plus whether each came from Y6 Gold (inside DES) or DELVE DR3.
Both are on the DES photometric system.
"""

import glob
import os
import re
from multiprocessing import Pool

import fitsio
import numpy as np

GOLD_DIR = "/home2/dchgomes/Turbulence_GPR/DES_DELVE/coadd_skims"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "gold_bright")
BRIGHT_MIN, BRIGHT_MAX = 16.0, 21.5
BANDS = "GRIZ"


def extract(path):
    pix = int(re.search(r"dr3_gold_(\d+)\.fits", path).group(1))
    cols = ["SOURCE", "ALPHAWIN_J2000", "DELTAWIN_J2000", "EXT_MASH", "FLAGS_GOLD"]
    cols += [f"PSF_MAG_APER_8_{b}" for b in BANDS]
    d = fitsio.read(path, columns=cols)
    mags = np.column_stack([d[f"PSF_MAG_APER_8_{b}"] for b in BANDS])
    keep = ((d["EXT_MASH"] >= 0) & (d["EXT_MASH"] <= 1) & (d["FLAGS_GOLD"] == 0)
            & np.any((mags > BRIGHT_MIN) & (mags < BRIGHT_MAX), axis=1))
    out = np.empty(keep.sum(), dtype=[("ra", "f8"), ("dec", "f8"), ("y6", "?")]
                   + [(b.lower(), "f4") for b in BANDS])
    out["ra"], out["dec"] = d["ALPHAWIN_J2000"][keep], d["DELTAWIN_J2000"][keep]
    out["y6"] = np.char.strip(d["SOURCE"][keep].astype(str)) == "Y6_GOLD"
    for i, b in enumerate(BANDS):
        out[b.lower()] = mags[keep, i]
    fitsio.write(os.path.join(OUT_DIR, f"bright_{pix:05d}.fits"), out, clobber=True)
    y6 = np.char.strip(d["SOURCE"].astype(str)) == "Y6_GOLD"
    return pix, len(d), int(y6.sum()), len(out)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(GOLD_DIR, "dr3_gold_*.fits")))
    with Pool(32) as pool:
        rows = pool.map(extract, paths, chunksize=8)
    index = np.array(rows, dtype=[("pix", "i4"), ("n_gold", "i8"), ("n_y6", "i8"), ("n_bright", "i8")])
    fitsio.write(os.path.join(OUT_DIR, "index.fits"), index, clobber=True)
    full_y6 = (index["n_y6"] == index["n_gold"]).sum()
    print(f"wrote {len(index)} pixels to {OUT_DIR}: {index['n_bright'].sum()} bright stars, "
          f"{full_y6} pixels entirely Y6 Gold")


if __name__ == "__main__":
    main()
