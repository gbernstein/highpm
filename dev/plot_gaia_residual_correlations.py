"""Star-by-star residuals (ours - Gaia) for one healpixel, and their correlations.

Residuals per matched star: dra (RA*cos(dec)), ddec [mas]; dpmra, dpmdec
[mas/yr]; dparallax [mas]. Our positions are at the fit reference epoch
(mjd_ref, ~J2016.0), the same epoch as Gaia. Shows

  (1) a corner plot of the five residuals against each other, with robust
      (outlier-clipped) Pearson r annotated -- covariance between our own
      fit parameters can produce some of these, so the median formal
      correlation is also printed for comparison;
  (2) each residual against Gaia PM (a PM-proportional error, as suggested by
      the GPR-Gaia drift, shows up as slope), G mag, BP-RP colour and sky
      position, with binned medians.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import numpy.lib.recfunctions as rfn
from astropy import units as u
from astropy.coordinates import SkyCoord

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from healpix_region_data import load_coadd, load_movers

RES = ["dra", "ddec", "dpmra", "dpmdec", "dpar"]
LAB = {"dra": "dRA* [mas]", "ddec": "dDec [mas]", "dpmra": "dPMRA [mas/yr]",
       "dpmdec": "dPMDEC [mas/yr]", "dpar": "dParallax [mas]"}


def robust_std(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def clip_mask(x, n=5):
    return np.abs(x - np.median(x)) < n * robust_std(x)


def binned(x, y, nb=10, minn=30):
    edges = np.quantile(x, np.linspace(0, 1, nb + 1))
    idx = np.clip(np.digitize(x, edges) - 1, 0, nb - 1)
    xc, ym, ye = [], [], []
    for i in range(nb):
        s = idx == i
        if s.sum() >= minn:
            xc.append(np.median(x[s]))
            ym.append(np.median(y[s]))
            ye.append(1.253 * robust_std(y[s]) / np.sqrt(s.sum()))
    return np.array(xc), np.array(ym), np.array(ye)


def slope(x, y):
    ok = clip_mask(x, 6) & clip_mask(y)
    p, cov = np.polyfit(x[ok], y[ok], 1, cov=True)
    return p[0], np.sqrt(cov[0, 0])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--healpix", type=int, required=True)
    ap.add_argument("--pmcatalog-dir", default="/data8/shared/decampm/PMSculptor_8deg/PMCatalog/")
    ap.add_argument("--gaia-dir", default="/data8/shared/decampm/Gaia/")
    ap.add_argument("--nside", type=int, default=32)
    ap.add_argument("--match-arcsec", type=float, default=0.3)
    ap.add_argument("--max-ruwe", type=float, default=1.4)
    ap.add_argument("--gmag-range", type=float, nargs=2, default=[13, 19.5])
    ap.add_argument("-o", "--out", default=None)
    a = ap.parse_args()
    pix = {a.healpix}

    movers = load_movers(a.pmcatalog_dir, a.nside, pix, pattern="real_PM_hp*.fits", exts=("modest1_movers",))
    gaia = load_coadd(a.gaia_dir, pix, pattern="GaiaSource_*.fits")
    gaia = gaia[np.isfinite(gaia["PMRA"]) & ~((gaia["PMRA"] == 0) & (gaia["PMDEC"] == 0) & (gaia["PARALLAX"] == 0))]
    mc = SkyCoord(movers["ra"] * u.deg, movers["dec"] * u.deg)
    gc = SkyCoord(gaia["RA"] * u.deg, gaia["DEC"] * u.deg)
    idx, d2d, _ = mc.match_to_catalog_sky(gc)
    _, d2d2, _ = mc.match_to_catalog_sky(gc, nthneighbor=2)
    ok = (d2d.arcsec < a.match_arcsec) & (d2d2.arcsec >= a.match_arcsec)
    t = rfn.merge_arrays([movers[ok], gaia[idx[ok]]], flatten=True, usemask=False)
    keep = (t["RUWE"] <= a.max_ruwe) & (t["PHOT_G_MEAN_MAG"] >= a.gmag_range[0]) & (t["PHOT_G_MEAN_MAG"] <= a.gmag_range[1])
    t = t[keep]
    print(f"[hp{a.healpix}] {len(movers)} movers, {len(gaia)} Gaia, {len(t)} unique clean matches")

    cosd = np.cos(np.radians(t["DEC"]))
    r = {"dra": (t["ra"] - t["RA"]) * cosd * 3.6e6, "ddec": (t["dec"] - t["DEC"]) * 3.6e6,
         "dpmra": t["pmra"] - t["PMRA"], "dpmdec": t["pmdec"] - t["PMDEC"],
         "dpar": t["parallax"] - t["PARALLAX"]}
    ours = {"dra": t["c_xx"], "ddec": t["c_yy"], "dpmra": t["c_vxvx"], "dpmdec": t["c_vyvy"], "dpar": t["c_pipi"]}
    gsig = {"dra": t["RA_ERROR"], "ddec": t["DEC_ERROR"], "dpmra": t["PMRA_ERROR"],
            "dpmdec": t["PMDEC_ERROR"], "dpar": t["PARALLAX_ERROR"]}
    pull = {k: r[k] / np.hypot(1e3 * np.sqrt(ours[k]), gsig[k]) for k in RES}

    good = np.ones(len(t), bool)
    for k in RES:
        good &= np.isfinite(r[k]) & clip_mask(r[k])
    print(f"[clip] {good.sum()} / {len(t)} inside 5 robust-sigma on all five residuals\n")
    print("median  robust-std  pull-median  pull-robust-std")
    for k in RES:
        print(f"{k:7s} {np.median(r[k][good]):8.3f} {robust_std(r[k][good]):8.3f} "
              f"{np.median(pull[k][good]):8.3f} {robust_std(pull[k][good]):8.3f}")

    R = np.array([r[k][good] for k in RES])
    P = np.array([pull[k][good] for k in RES])
    C, CP = np.corrcoef(R), np.corrcoef(P)
    print("\nPearson r of residuals:\n      " + " ".join(f"{k:>7s}" for k in RES))
    for i, k in enumerate(RES):
        print(f"{k:6s}" + " ".join(f"{C[i, j]:7.3f}" for j in range(5)))
    print("\nPearson r of pulls:")
    for i, k in enumerate(RES):
        print(f"{k:6s}" + " ".join(f"{CP[i, j]:7.3f}" for j in range(5)))
    pairs = {"x-y": ("c_xy", "c_xx", "c_yy"), "x-vx": ("c_xvx", "c_xx", "c_vxvx"), "y-vy": ("c_yvy", "c_yy", "c_vyvy"),
             "x-pi": ("c_xpi", "c_xx", "c_pipi"), "y-pi": ("c_ypi", "c_yy", "c_pipi"),
             "vx-pi": ("c_vxpi", "c_vxvx", "c_pipi"), "vy-pi": ("c_vypi", "c_vyvy", "c_pipi")}
    print("\nMedian formal correlation in OUR fit covariance (expected from fit alone):")
    for n, (c, d1, d2) in pairs.items():
        print(f"  {n}: {np.median(t[c] / np.sqrt(t[d1] * t[d2])):.3f}")

    # (1) corner plot
    fig, axes = plt.subplots(5, 5, figsize=(15, 15))
    for i, ky in enumerate(RES):
        for j, kx in enumerate(RES):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue
            if i == j:
                ax.hist(r[ky][good], bins=60, color="steelblue")
            else:
                ax.hexbin(r[kx][good], r[ky][good], gridsize=40, bins="log", cmap="viridis", mincnt=1)
                ax.text(0.05, 0.92, f"r={C[i, j]:+.3f}", transform=ax.transAxes, color="w",
                        bbox=dict(fc="k", alpha=0.6, lw=0), fontsize=10)
            if i == 4:
                ax.set_xlabel(LAB[kx])
            if j == 0 and i > 0:
                ax.set_ylabel(LAB[ky])
    fig.suptitle(f"hp{a.healpix}: residuals (ours - Gaia), N={good.sum()}", fontsize=14)
    plt.tight_layout()
    out = a.out or f"gaia_residual_corner_hp{a.healpix}.png"
    plt.savefig(out, dpi=120)
    print(f"Saved {out}")

    # (2) residuals vs Gaia PM / G / colour / position
    xs = [("PMRA", t["PMRA"], "Gaia PMRA [mas/yr]"), ("PMDEC", t["PMDEC"], "Gaia PMDEC [mas/yr]"),
          ("G", t["PHOT_G_MEAN_MAG"], "Gaia G"), ("BP_RP", t["BP_RP"], "BP-RP"),
          ("RA", t["RA"], "RA [deg]"), ("DEC", t["DEC"], "Dec [deg]")]
    fig, axes = plt.subplots(5, 6, figsize=(24, 16))
    print("\nSlope of residual vs x (per unit x):")
    for i, ky in enumerate(RES):
        for j, (xn, x, xl) in enumerate(xs):
            ax = axes[i, j]
            g2 = good & np.isfinite(x)
            ax.hexbin(x[g2], r[ky][g2], gridsize=40, bins="log", cmap="Greys", mincnt=1)
            xc, ym, ye = binned(x[g2], r[ky][g2])
            ax.errorbar(xc, ym, ye, fmt="o-", color="crimson", ms=4)
            ax.axhline(0, color="k", lw=0.8, ls="--")
            s, se = slope(x[g2], r[ky][g2])
            ax.set_title(f"slope {s:+.4g} +/- {se:.2g}", fontsize=9)
            print(f"  {ky:7s} vs {xn:6s}: {s:+.4g} +/- {se:.2g}  ({s / se:+.1f} sigma)")
            if i == 4:
                ax.set_xlabel(xl)
            if j == 0:
                ax.set_ylabel(LAB[ky])
            lim = 4 * robust_std(r[ky][good])
            ax.set_ylim(np.median(r[ky][good]) - lim, np.median(r[ky][good]) + lim)
    fig.suptitle(f"hp{a.healpix}: residuals vs Gaia properties (red = binned median)", fontsize=14)
    plt.tight_layout()
    out2 = (a.out or f"gaia_residual_corner_hp{a.healpix}.png").replace(".png", "_vs_props.png")
    plt.savefig(out2, dpi=100)
    print(f"Saved {out2}")
