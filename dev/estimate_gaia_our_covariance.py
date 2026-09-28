"""Estimate the correlation coefficient rho between OUR PM errors and Gaia's
PM errors, per component, from the pull distribution built in
plot_gaia_pm_error_pull.py.

Model: if the two measurements have correlated errors,
    Var(our - gaia) = sigma_our^2 + sigma_gaia^2 - 2*rho*sigma_our*sigma_gaia
but the pull statistic divides by sqrt(sigma_our^2+sigma_gaia^2) (the
independence-assumption denominator), so
    Var(pull) = 1 - 2*rho*w,   w = 2*sigma_our*sigma_gaia / (sigma_our^2+sigma_gaia^2)
w is known per star. Binning by w and fitting a line to the local (robust)
pull variance recovers rho as -slope/2, with the intercept as a sanity check
(should be ~1 if there's no leftover pure miscalibration on top of the
correlation).
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--healpix", type=int, nargs="+", required=True)
    parser.add_argument("--pmcatalog-dir", default="/data8/shared/decampm/PMSculptor_8deg_garyb/PMCatalog/")
    parser.add_argument("--gaia-dir", default="/data8/shared/decampm/Gaia/")
    parser.add_argument("--nside", type=int, default=32)
    parser.add_argument("--match-arcsec", type=float, default=0.3)
    parser.add_argument("--max-ruwe", type=float, default=1.4)
    parser.add_argument("--pm-pattern", default="real_PM_hp*.fits")
    parser.add_argument("--gaia-pattern", default="GaiaSource_*.fits")
    parser.add_argument("--nbins", type=int, default=20)
    parser.add_argument("-o", "--out", default="gaia_our_covariance.png")
    parser.add_argument("--title", default="Sculptor 8 degree radius (46 GPR2-complete healpix)")
    args = parser.parse_args()

    pixels = set(args.healpix)
    movers = load_movers(args.pmcatalog_dir, args.nside, pixels,
                          pattern=args.pm_pattern, exts=("modest1_movers",))
    gaia = load_coadd(args.gaia_dir, pixels, pattern=args.gaia_pattern)
    gaia = gaia[~np.isnan(gaia["PMRA"])]
    two_param = (gaia["PMRA"] == 0) & (gaia["PMDEC"] == 0) & (gaia["PARALLAX"] == 0)
    gaia = gaia[~two_param]

    mover_coords = SkyCoord(ra=movers["ra"] * u.degree, dec=movers["dec"] * u.degree)
    gaia_coords = SkyCoord(ra=gaia["RA"] * u.degree, dec=gaia["DEC"] * u.degree)
    idx, d2d, _ = mover_coords.match_to_catalog_sky(gaia_coords)
    close = d2d.arcsecond < args.match_arcsec
    matched = rfn.merge_arrays([movers[close], gaia[idx[close]]], flatten=True, usemask=False)
    if args.max_ruwe is not None:
        matched = matched[matched["RUWE"] <= args.max_ruwe]
    print(f"[match] {len(matched)} matched stars")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    results = {}
    for ax, comp, our_col_var, gaia_err_col, mu_col, gaia_mu_col in zip(
        axes, ("pmra", "pmdec"), ("c_vxvx", "c_vyvy"),
        ("PMRA_ERROR", "PMDEC_ERROR"), ("pmra", "pmdec"), ("PMRA", "PMDEC"),
    ):
        sigma_our = 1000 * np.sqrt(matched[our_col_var])
        sigma_gaia = matched[gaia_err_col]
        pull = (matched[mu_col] - matched[gaia_mu_col]) / np.hypot(sigma_our, sigma_gaia)

        # Restrict to the well-behaved core to keep this an error-budget
        # estimate, not something dominated by the ~0.2% outlier tail.
        core = np.abs(pull) < 5
        sigma_our, sigma_gaia, pull = sigma_our[core], sigma_gaia[core], pull[core]

        w = 2 * sigma_our * sigma_gaia / (sigma_our**2 + sigma_gaia**2)

        edges = np.linspace(0, w.max(), args.nbins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        var_robust = np.full(args.nbins, np.nan)
        n = np.zeros(args.nbins, dtype=int)
        idx_bin = np.digitize(w, edges) - 1
        for i in range(args.nbins):
            sel = idx_bin == i
            n[i] = sel.sum()
            if n[i] >= 200:
                var_robust[i] = robust_std(pull[sel]) ** 2

        good = np.isfinite(var_robust)
        # Weighted linear fit: Var(pull) = a - 2*rho*w, weight by sqrt(n) (rough).
        coeffs, cov = np.polyfit(centers[good], var_robust[good], 1, w=np.sqrt(n[good]), cov=True)
        slope, intercept = coeffs
        rho = -slope / 2
        rho_err = np.sqrt(cov[0, 0]) / 2
        results[comp] = (rho, rho_err, intercept)
        print(f"{comp}: intercept={intercept:.3f}, slope={slope:.3f}, "
              f"rho={rho:.3f} +/- {rho_err:.3f}")

        ax.scatter(centers, var_robust, s=40, color="steelblue", label="binned robust Var(pull)")
        xx = np.linspace(0, w.max(), 100)
        ax.plot(xx, intercept + slope * xx, "r-",
                label=fr"fit: $1-2\rho w$, $\rho={rho:.3f}\pm{rho_err:.3f}$")
        ax.plot(xx, 1 - 0 * xx, "k--", lw=1, label="no correlation (rho=0)")
        ax.set_xlabel(r"$w = 2\sigma_{our}\sigma_{gaia}/(\sigma_{our}^2+\sigma_{gaia}^2)$", fontsize=12)
        ax.set_ylabel("Var(pull) (robust)", fontsize=12)
        ax.set_title(comp)
        ax.legend(fontsize=9)
        ax.grid()

    fig.suptitle(args.title, fontsize=14)
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"Saved {args.out}")
