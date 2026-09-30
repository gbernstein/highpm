"""
How many true stars does clean_cat's single-epoch star/galaxy cut (extval <= 1,
highpm.cat_reader.wavg_extended_class_y6a2) throw away?

Reference: isolated high-confidence Y6 Gold stars (EXT_MASH == 0) detected in the
exposure with clean flags. Plots the fraction passing the class cut vs. expected
single-epoch S/N = 10**(-0.4 (m - m1_ccd)), from each exposure's measured depth
(scripts/measure_exposure_depth.py). If the curves for different exposure times and
processing collapse, the loss is an S/N effect and can be modelled from m1 too.
"""

import argparse
import os
import sys

import fitsio
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
sys.path.insert(0, os.path.join(HERE, ".."))
import calibrate_snr_threshold as cal  # noqa: E402
from highpm.cat_reader import wavg_extended_class_y6a2  # noqa: E402

SNR_BINS = np.logspace(np.log10(3), np.log10(300), 25)
TAG_STYLE = {"DES": "o-", "DELVE": "s--"}


def class_pass(row, cc):
    """(expected S/N, passes class cut) for detected EXT_MASH == 0 gold stars."""
    e, b = int(row["expnum"]), str(row["band"])
    mag, _, ccd, ra, dec = cal.reference_stars(e, b, cc, mash_max=0)
    s = fitsio.read(cal.SKIM.format(B=b.upper(), e=e, b=b),
                    columns=["RA", "DEC", "FLAGS", "IMAFLAGS_ISO", "SPREAD_MODEL", "SPREADERR_MODEL"])
    idx = cal.match(ra, dec, s["RA"], s["DEC"])
    j = np.maximum(idx, 0)
    det = (idx >= 0) & (s["FLAGS"][j] < 4) & (s["IMAFLAGS_ISO"][j] == 0)
    ext = wavg_extended_class_y6a2(s["SPREAD_MODEL"], s["SPREADERR_MODEL"])[j]
    snr = 10 ** (-0.4 * (mag - row["m1_ccd"][ccd]))
    return snr[det], ext[det] <= 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", default=cal.DEPTH)
    ap.add_argument("--expnums", type=int, nargs="+", required=True)
    args = ap.parse_args()

    depth = fitsio.read(args.depth)
    depth = depth[np.isin(depth["expnum"], args.expnums)]
    corners = fitsio.read(cal.CORNERS, ext=1, columns=["expnum", "ccdnum", "ra", "dec"])
    mid = np.sqrt(SNR_BINS[1:] * SNR_BINS[:-1])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for row in depth[np.argsort(depth["exptime"])]:
        snr, ok = class_pass(row, corners[corners["expnum"] == row["expnum"]])
        n, _ = np.histogram(snr, SNR_BINS)
        k, _ = np.histogram(snr[ok], SNR_BINS)
        use = n >= 20
        frac = k[use] / n[use]
        ax.plot(mid[use], frac, TAG_STYLE.get(row["tag"], "^:"), ms=4, lw=1.3,
                label=f"{row['expnum']} {row['band']} {row['exptime']:.0f}s {row['tag']}")
        print(f"{row['expnum']} {row['band']} {row['exptime']:4.0f}s {row['tag']:5s}  pass fraction at "
              + ", ".join(f"S/N {x:.0f}: {np.interp(np.log(x), np.log(mid[use]), frac):.2f}"
                          for x in (5, 10, 20, 50)))
    ax.set_xscale("log")
    ax.set_xlabel("expected single-epoch S/N")
    ax.set_ylabel("fraction of detected true stars with extval <= 1")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.set_title("clean_cat star/galaxy cut on Y6 Gold EXT_MASH == 0 stars")
    fig.tight_layout()
    out = os.path.join(HERE, "star_class_loss.png")
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
