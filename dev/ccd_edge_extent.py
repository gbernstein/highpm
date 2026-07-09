#!/usr/bin/env python
"""Median per-CCD pixel extent of detections across a directory of exposure catalogs.

For each ccdnum, measures the detection pixel min/max in every exposure, then
takes the median over exposures -> one stable active-area box per CCD. Those
boxes are the pixel corners later fed through pixmappy to build sky corners
(y6a1.ccdcorners-style). Streams one exposure at a time, so it scales to
thousands of exposures.
"""
import argparse, glob, os
import numpy as np
import fitsio


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", help="dir of *_cat.fits exposure catalogs")
    p.add_argument("--pattern", default="*_cat.fits")
    p.add_argument("--max", type=int, default=None, help="max exposures to read")
    p.add_argument("--xcol", default="XWIN_IMAGE")
    p.add_argument("--ycol", default="YWIN_IMAGE")
    p.add_argument("--out", help="write per-ccd box to this FITS file")
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.directory, args.pattern)))[: args.max]
    if not files:
        p.error(f"no files matching {args.pattern} in {args.directory}")

    cols = ["CCDNUM", args.xcol, args.ycol]
    # per (exposure, ccd) extent, tagged by ccdnum
    ccdnums, xmin, xmax, ymin, ymax = [], [], [], [], []
    ndet = 0
    for i, f in enumerate(files):
        d = fitsio.read(f, columns=cols)
        ndet += d.size
        ccd = d["CCDNUM"]
        for c in np.unique(ccd):  # each file is one exposure -> ccd alone is the group
            m = ccd == c
            ccdnums.append(int(c))
            xmin.append(d[args.xcol][m].min()); xmax.append(d[args.xcol][m].max())
            ymin.append(d[args.ycol][m].min()); ymax.append(d[args.ycol][m].max())
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(files)} exposures read", flush=True)

    ccdnums = np.array(ccdnums)
    xmin, xmax = np.array(xmin), np.array(xmax)
    ymin, ymax = np.array(ymin), np.array(ymax)

    # median over exposures, per ccdnum
    uccd = np.unique(ccdnums)
    box = np.zeros(uccd.size, dtype=[("ccdnum", "i2"), ("xmin", "f8"),
                                     ("xmax", "f8"), ("ymin", "f8"),
                                     ("ymax", "f8"), ("nexp", "i4")])
    for j, c in enumerate(uccd):
        m = ccdnums == c
        box[j] = (c, np.median(xmin[m]), np.median(xmax[m]),
                  np.median(ymin[m]), np.median(ymax[m]), int(m.sum()))

    print(f"{len(files)} exposures, {ndet} detections, {uccd.size} CCDs")
    print("(mins floored, maxs ceiled -- the values build_ccdcorners uses)")
    print("ccdnum  xmin    xmax    ymin    ymax    nexp")
    for r in box:
        print(f"{r['ccdnum']:5d} {np.floor(r['xmin']):7.0f} {np.ceil(r['xmax']):7.0f} "
              f"{np.floor(r['ymin']):7.0f} {np.ceil(r['ymax']):7.0f} {r['nexp']:6d}")

    if args.out:
        fitsio.write(args.out, box, clobber=True)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
