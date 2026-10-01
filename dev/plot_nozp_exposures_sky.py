"""
Sky map of the skims in /data8/shared/decampm/[GRIZ]/ with no row in
data/delve.exposures.completeness.fits (so fakes get no detections there), colored by the
reason: no zeropoint (no DELVE catalog star matched) or a tag/band with no S/N-threshold
calibration. Drawn over the DELVE catalog healpixels (COADDS) the zeropoint is taken from.
Exposure position = median of its skim's detections.
"""

import os
import re
import sys
from multiprocessing import Pool

import fitsio
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(REPO, "scripts"))
import measure_exposure_depth as M  # noqa: E402

SKIM_DIR = "/data8/shared/decampm/{B}"
COADD_DIR = os.path.dirname(M.COADD)
NSIDE = 32
SKIM_NAME = re.compile(r"D(\d{8})_([griz])_cat\.fits$")


def skims():
    out = []
    for B in "GRIZ":
        for f in os.listdir(SKIM_DIR.format(B=B)):
            m = SKIM_NAME.match(f)
            if m:
                out.append((int(m[1]), m[2]))
    return out


def position(args):
    e, b = args
    s = fitsio.read(M.SKIM.format(B=b.upper(), e=e, b=b), columns=["RA", "DEC"])
    ra = np.degrees(np.angle(np.mean(np.exp(1j * np.radians(s["RA"]))))) % 360
    return float(ra), float(np.median(s["DEC"]))


def to_plot(ra):
    """RA in degrees -> mollweide x in radians, RA increasing to the left."""
    return -np.radians(((ra + 180) % 360) - 180)


def main():
    depth = fitsio.read(os.path.join(REPO, "data", "delve.exposures.depth.fits"),
                        columns=["expnum", "band", "tag", "zp"])
    table = fitsio.read(os.path.join(REPO, "data", "delve.exposures.completeness.fits"),
                        columns=["expnum", "band"])
    have = set(zip(table["expnum"].tolist(), table["band"].tolist()))
    drow = {(int(e), b): i for i, (e, b) in enumerate(zip(depth["expnum"], depth["band"]))}
    missing = [k for k in skims() if k not in have]
    reason = []
    for k in missing:
        i = drow.get(k)
        if i is None:
            reason.append("no depth row")
        elif not np.isfinite(depth["zp"][i]):
            reason.append("no zeropoint")
        else:
            reason.append(f"uncalibrated {depth['tag'][i].strip()} {k[1]}")
    reason = np.array(reason)
    with Pool(min(len(missing), 48)) as pool:
        ra, dec = np.array(pool.map(position, missing)).T

    coadd = np.array(sorted(int(f[8:13]) for f in os.listdir(COADD_DIR) if f.startswith("cat_hpx_")))
    fig = plt.figure(figsize=(15, 8.5))
    ax = fig.add_subplot(111, projection="mollweide")
    pra, pdec = hp.pix2ang(NSIDE, coadd, lonlat=True)
    ax.scatter(to_plot(pra), np.radians(pdec), s=9, marker="s", c="#d5e3f0", lw=0,
               label="DELVE catalog healpixels (zp reference)")
    colors = ["#d6453d", "#eb9b34", "#2a78d6", "#52514e"]
    for col, r in zip(colors, np.unique(reason)):
        s = reason == r
        ax.scatter(to_plot(ra[s]), np.radians(dec[s]), s=22, c=col, alpha=0.8, lw=0.3,
                   edgecolor="white", label=f"{r} ({s.sum()})")
    for name, (r, dd) in {"LMC": (80.89, -69.76), "SMC": (13.19, -72.83)}.items():
        ax.annotate(name, (to_plot(r), np.radians(dd)), fontsize=10, weight="bold", ha="center",
                    xytext=(0, 10), textcoords="offset points")
    l = np.linspace(0, 360, 721)
    for bb, ls in [(0, "-"), (10, ":"), (-10, ":")]:
        gra, gdec = hp.Rotator(coord=["G", "C"])(l, np.full_like(l, bb), lonlat=True)
        x = to_plot(gra)
        brk = np.where(np.abs(np.diff(x)) > 1)[0] + 1
        for seg_x, seg_y in zip(np.split(x, brk), np.split(np.radians(gdec), brk)):
            ax.plot(seg_x, seg_y, color="#52514e", lw=0.8, ls=ls)
    ticks = np.arange(-150, 181, 30)
    ax.set_xticks(np.radians(ticks))
    ax.set_xticklabels([f"{(-t) % 360:d}°" for t in ticks], fontsize=8)
    ax.grid(color="#c8c7c0", lw=0.5)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=4, fontsize=9, markerscale=1.5,
              frameon=False)
    ax.set_title(f"{len(missing)} of the skims in /data8/shared/decampm/[GRIZ] have no completeness row; "
                 "Galactic plane and |b| = 10° drawn", pad=14)
    fig.tight_layout()
    out = os.path.join(HERE, "nozp_exposures_sky.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"wrote {out}")
    for r in np.unique(reason):
        print(f"  {r}: {np.sum(reason == r)}")


if __name__ == "__main__":
    main()
