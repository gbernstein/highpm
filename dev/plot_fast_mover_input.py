"""Scatter plot of the detections that survive the modest-mover rounds and
get passed into fast_movers (the O(N^2)-ish pairwise stage). Reconstructs
the same cleaning + modest-removal bookkeeping as scripts/PM.py:run_pm
without rerunning the fit, by reading the modestN_detections extensions
already written to the in-progress/completed movers file.
"""

import argparse
import os
import sys

import fitsio
import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from highpm.cat_reader import clean_cat, read_cat_header
from highpm.gnomonic_converter import projectGnomonic
from highpm.pm_setup import read_pm_cat

# NGC 300 optical-diameter ellipse (SIMBAD basic data): center
# 00:54:53.4465638304 -37:41:03.168402396, major/minor 20.89'/13.49', PA=114 deg (E of N).
NGC300_RA, NGC300_DEC = 13.722694016, -37.684213445
NGC300_MAJOR_DEG, NGC300_MINOR_DEG = 20.89 / 60, 13.49 / 60
NGC300_PA_DEG = 114.0


def ngc300_ellipse_xieta(ra0, dec0, n=200):
    """Optical-diameter ellipse boundary points, projected into this catalog's xi/eta frame."""
    t = np.linspace(0, 2 * np.pi, n)
    a, b = NGC300_MAJOR_DEG / 2, NGC300_MINOR_DEG / 2
    x, y = a * np.cos(t), b * np.sin(t)  # offsets along major/minor axes
    pa = np.radians(NGC300_PA_DEG)  # position angle, from North through East
    north = x * np.cos(pa) - y * np.sin(pa)
    east = x * np.sin(pa) + y * np.cos(pa)
    ra = NGC300_RA + east / np.cos(np.radians(NGC300_DEC))
    dec = NGC300_DEC + north
    zeros = np.zeros_like(ra)
    xi, eta, *_ = projectGnomonic(ra, dec, zeros, zeros, ra0, dec0)
    return xi, eta

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detections", required=True, help="cleaned_detections_hpXXXXX.fits")
    parser.add_argument("--movers", required=True, help="real_PM_hpXXXXX.fits (must already have modestN_detections extensions)")
    parser.add_argument("--config", default="/home2/vwetzell/gitrepos/highpm/config/config.yaml")
    parser.add_argument("--healpix", type=int, required=True)
    parser.add_argument("-o", "--out", default="fast_mover_input_scatter.png")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if not ("ra0" in config or "dec0" in config):
        cat_header = read_cat_header(args.detections)
        config["ra0"] = cat_header["ra0"]
        config["dec0"] = cat_header["dec0"]

    cat = read_pm_cat(args.detections)
    print(f"Detections loaded: {len(cat)}")

    cleanmask = clean_cat(cat, config)
    cat_idx = np.arange(len(cat))[cleanmask]
    cat = cat[cleanmask]
    print(f"Detections after cleaning: {len(cat)}")

    # Mirror highpm.utils.detections_for_removal exactly: a modest mover's
    # *member* detections (not its clipped ones) are only removed from the
    # fast-mover pool if the mover itself is confidently non-fast (low PM,
    # low PM error) -- so removal is decided per-mover, not per-detection.
    pm_lim = config["pm_lim"]
    pm_err_lim = config["pm_err_lim"]

    movers = fitsio.FITS(args.movers)
    extnames = [movers[i].get_extname() for i in range(len(movers))]
    removed_idx = []
    for mtype in ("modest1", "modest2", "modest3"):
        movers_ext, det_ext = f"{mtype}_movers", f"{mtype}_detections"
        if movers_ext not in extnames or det_ext not in extnames:
            print(f"{movers_ext}/{det_ext} not present, skipping")
            continue
        mv = movers[movers_ext].read()
        pmra_err = 1000 * mv["c_vxvx"]
        pmdec_err = 1000 * mv["c_vyvy"]
        keep_mover = (np.abs(pmra_err) < pm_err_lim) & (np.abs(pmdec_err) < pm_err_lim) & \
            (np.hypot(mv["pmra"], mv["pmdec"]) < pm_lim)
        removable_idx = set(np.where(keep_mover)[0].tolist())

        det = movers[det_ext].read()
        sel = np.array([i in removable_idx for i in det["idx"]]) & ~det["clipped"]
        removed_idx.append(det["detections"][sel])
        print(f"{det_ext}: {sel.sum()} member detections removed ({len(mv)} movers, {keep_mover.sum()} pass removal criteria)")
    removed_idx = np.unique(np.concatenate(removed_idx)) if removed_idx else np.array([], dtype=np.int64)

    remaining_mask = ~np.isin(cat_idx, removed_idx)
    remaining = cat[remaining_mask]
    print(f"Detections remaining for fast mover search: {len(remaining)}")

    xi = remaining["XI"]
    eta = remaining["ETA"]

    fig, ax = plt.subplots(figsize=(8, 7))
    hb = ax.hexbin(xi, eta, gridsize=150, cmap="managua", mincnt=1, edgecolor="none")
    fig.colorbar(hb, ax=ax, label="detections / bin")

    ngc300_xi, ngc300_eta = ngc300_ellipse_xieta(config["ra0"], config["dec0"])
    ax.plot(ngc300_xi, ngc300_eta, "-", color="lime", lw=1.5, label="NGC 300 (SIMBAD optical diameter)")
    ax.legend(fontsize=9, loc="upper right")

    ax.set_xlabel(r"$\xi$ (deg)", fontsize=14)
    ax.set_ylabel(r"$\eta$ (deg)", fontsize=14)
    ax.set_title(f"hp{args.healpix:05d}: {len(remaining)} detections into fast_movers", fontsize=12)
    ax.set_aspect("equal")
    ax.invert_xaxis()
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"Saved {args.out}")
