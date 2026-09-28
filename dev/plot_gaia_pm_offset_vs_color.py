"""Proper-motion offset (ours - Gaia) vs Gaia BP-RP color.

Companion to dev/plot_gaia_pm_error_pull.py, which bins the same our-vs-Gaia
PM comparison by Gaia G magnitude. This asks whether there's a color-
dependent systematic instead: e.g. a differential chromatic refraction (DCR)
residual, or a bias tied to which DECam bands (g/r/i/z) a star was actually
detected in, would show up here as an offset that trends with BP-RP even
though it might look flat vs magnitude.

Unlike the pull-test scripts, this reports the raw offset in mas/yr (not
normalized by the combined formal error), since a color-dependent shift in
the offset itself -- not necessarily its significance per star -- is the
thing of interest here.

Produces:
  (1) pmra, pmdec offset (ours - Gaia) vs BP-RP, binned, inverse-variance
      weighted by the combined formal errors.
  (2) Same, but restricted to a single Gaia magnitude range (--gmag-range)
      so a magnitude-color degeneracy (redder stars tend to be fainter in a
      magnitude-limited sample) doesn't masquerade as a color effect.
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


def robust_std(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def binned_offset(color, offset, sigma, bins):
    edges = np.linspace(*bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.digitize(color, edges) - 1
    wmean = np.full(len(centers), np.nan)
    wmean_err = np.full(len(centers), np.nan)
    rstd = np.full(len(centers), np.nan)
    n = np.zeros(len(centers), dtype=int)
    for i in range(len(centers)):
        sel = idx == i
        n[i] = sel.sum()
        if n[i] >= 20:
            w = 1.0 / sigma[sel] ** 2
            wmean[i] = np.sum(w * offset[sel]) / np.sum(w)
            wmean_err[i] = 1.0 / np.sqrt(np.sum(w))
            rstd[i] = robust_std(offset[sel])
    return centers, wmean, wmean_err, rstd, n


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--healpix", type=int, nargs="+", required=True, help="Explicit healpix indices to plot")
    parser.add_argument("--pmcatalog-dir", default="/data8/shared/decampm/PMSculptor_8deg_garyb/PMCatalog/")
    parser.add_argument("--gaia-dir", default="/data8/shared/decampm/Gaia/")
    parser.add_argument("--nside", type=int, default=32)
    parser.add_argument("--match-arcsec", type=float, default=0.3)
    parser.add_argument("--max-ruwe", type=float, default=1.4,
                         help="Drop Gaia matches with RUWE above this (1.4 is the standard "
                              "'reliable single-star astrometric solution' cut)")
    parser.add_argument("--require-unique", action="store_true",
                         help="Drop movers whose 2nd-nearest Gaia neighbor is also within --match-arcsec "
                              "(ambiguous/crowded match)")
    parser.add_argument("--pm-pattern", default="real_PM_hp*.fits")
    parser.add_argument("--gaia-pattern", default="GaiaSource_*.fits")
    parser.add_argument("--color-range", type=float, nargs=2, default=(-0.5, 3.5),
                         help="BP-RP range to bin over")
    parser.add_argument("--color-bins", type=int, default=32)
    parser.add_argument("--gmag-range", type=float, nargs=2, default=None,
                         help="Restrict bottom row to matched stars with GMAG_MIN <= PHOT_G_MEAN_MAG <= GMAG_MAX")
    parser.add_argument("-o", "--out", default="gaia_pm_offset_vs_color.png")
    parser.add_argument("--title", default="Sculptor 8 degree radius (46 GPR2-complete healpix): "
                                            "PM offset vs Gaia BP-RP color")
    args = parser.parse_args()

    pixels = set(args.healpix)
    movers = load_movers(args.pmcatalog_dir, args.nside, pixels,
                          pattern=args.pm_pattern, exts=("modest1_movers",))
    gaia = load_coadd(args.gaia_dir, pixels, pattern=args.gaia_pattern)
    gaia = gaia[~np.isnan(gaia["PMRA"])]
    two_param = (gaia["PMRA"] == 0) & (gaia["PMDEC"] == 0) & (gaia["PARALLAX"] == 0)
    print(f"[gaia] dropping {two_param.sum()} / {len(gaia)} apparent 2-parameter (position-only) solutions")
    gaia = gaia[~two_param]

    mover_coords = SkyCoord(ra=movers["ra"] * u.degree, dec=movers["dec"] * u.degree)
    gaia_coords = SkyCoord(ra=gaia["RA"] * u.degree, dec=gaia["DEC"] * u.degree)
    idx, d2d, _ = mover_coords.match_to_catalog_sky(gaia_coords)
    close = d2d.arcsecond < args.match_arcsec

    if args.require_unique:
        idx2, d2d2, _ = mover_coords.match_to_catalog_sky(gaia_coords, nthneighbor=2)
        close &= d2d2.arcsecond >= args.match_arcsec

    matched = rfn.merge_arrays([movers[close], gaia[idx[close]]], flatten=True, usemask=False)
    print(f"[match] {len(movers)} modest1 movers, {len(matched)} matched to Gaia within {args.match_arcsec} arcsec"
          f"{' (unique)' if args.require_unique else ''}")

    if args.max_ruwe is not None:
        keep = matched["RUWE"] <= args.max_ruwe
        print(f"[ruwe] dropping {(~keep).sum()} / {len(matched)} with RUWE > {args.max_ruwe}")
        matched = matched[keep]

    keep = np.isfinite(matched["BP_RP"])
    print(f"[color] dropping {(~keep).sum()} / {len(matched)} with no BP_RP")
    matched = matched[keep]

    offset_ra = matched["pmra"] - matched["PMRA"]     # mas/yr
    offset_dec = matched["pmdec"] - matched["PMDEC"]  # mas/yr
    our_sigma_pmra = 1000 * np.sqrt(matched["c_vxvx"])   # mas/yr
    our_sigma_pmdec = 1000 * np.sqrt(matched["c_vyvy"])  # mas/yr
    sigma_ra = np.hypot(our_sigma_pmra, matched["PMRA_ERROR"])
    sigma_dec = np.hypot(our_sigma_pmdec, matched["PMDEC_ERROR"])

    finite_ra = np.isfinite(offset_ra) & np.isfinite(sigma_ra) & (sigma_ra > 0)
    finite_dec = np.isfinite(offset_dec) & np.isfinite(sigma_dec) & (sigma_dec > 0)

    color = matched["BP_RP"]
    gmag = matched["PHOT_G_MEAN_MAG"]
    cbins = (args.color_range[0], args.color_range[1], args.color_bins + 1)

    for label, off, sig, fin in (("pmra", offset_ra, sigma_ra, finite_ra),
                                  ("pmdec", offset_dec, sigma_dec, finite_dec)):
        w = 1.0 / sig[fin] ** 2
        wmean = np.sum(w * off[fin]) / np.sum(w)
        wmean_err = 1.0 / np.sqrt(np.sum(w))
        print(f"{label:5s} offset: inv-var-weighted mean = {wmean:+.4f} +/- {wmean_err:.4f} mas/yr "
              f"(median = {np.median(off[fin]):+.4f}, robust std = {robust_std(off[fin]):.4f}, n={fin.sum()})")

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    for ax, off, sig, fin, label in zip(axes[0], (offset_ra, offset_dec), (sigma_ra, sigma_dec),
                                         (finite_ra, finite_dec), ("pmra", "pmdec")):
        centers, wmean, wmean_err, rstd, n = binned_offset(color[fin], off[fin], sig[fin], cbins)
        ax.errorbar(centers, wmean, yerr=wmean_err, fmt="o-", color="darkorange", label="weighted mean offset")
        ax.axhline(0.0, color="k", ls="--", lw=1.5)
        ax.set_xlabel("Gaia BP-RP color (mag)")
        ax.set_ylabel(f"{label} offset (ours - Gaia, mas/yr)")
        ax.set_title(f"{label}: all magnitudes, n={fin.sum()}")
        ax.legend(fontsize=9)
        ax.grid()

    for ax, off, sig, fin, label in zip(axes[1], (offset_ra, offset_dec), (sigma_ra, sigma_dec),
                                         (finite_ra, finite_dec), ("pmra", "pmdec")):
        if args.gmag_range is not None:
            gmin, gmax = args.gmag_range
            sel = fin & (gmag >= gmin) & (gmag <= gmax)
            title_suffix = f"{gmin:.1f} <= G <= {gmax:.1f}, n={sel.sum()}"
        else:
            sel = fin
            title_suffix = f"all magnitudes (repeat of top row), n={sel.sum()}"
        centers, wmean, wmean_err, rstd, n = binned_offset(color[sel], off[sel], sig[sel], cbins)
        ax.errorbar(centers, wmean, yerr=wmean_err, fmt="o-", color="steelblue", label="weighted mean offset")
        ax.axhline(0.0, color="k", ls="--", lw=1.5)
        ax.set_xlabel("Gaia BP-RP color (mag)")
        ax.set_ylabel(f"{label} offset (ours - Gaia, mas/yr)")
        ax.set_title(f"{label}: {title_suffix}")
        ax.legend(fontsize=9)
        ax.grid()

    fig.suptitle(args.title, fontsize=14)
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"Saved {args.out}")
