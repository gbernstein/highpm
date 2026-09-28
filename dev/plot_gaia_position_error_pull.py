"""Per-component formal-error accuracy check against Gaia, for POSITION
(the position-space analog of plot_gaia_pm_error_pull.py).

Assumes Gaia's RA/DEC and their formal errors are correct, and asks: for the
same stars, are OUR fitted positions (x0, y0 at mjd_ref, from modest1_movers)
consistent with Gaia's, given both sets of formal errors? For each component
c in {ra, dec}, the pull

    pull_c = (our_c - gaia_c) / sqrt(our_sigma_c^2 + gaia_sigma_c^2)

should be a standard normal (mean 0, std 1) if both sets of formal errors are
accurate and there is no residual systematic offset between the two frames.
A nonzero MEAN pull is the position-space signature of exactly the kind of
constant frame offset under investigation (e.g. a GPR/Gaia epoch or
propagation mismatch): unlike the PM pull test, a mean offset here directly
measures a positional zero-point difference between our reference frame and
Gaia's, at our fit's reference epoch (mjd_ref).

Our x0/y0 come from the same 5-parameter fit5d() solution as pmra/pmdec
(highpm/pmfit.py), stored as ra/dec (deg) with covariance c_xx/c_yy (arcsec^2,
in the tangent-plane frame -- same small-field approximation already used to
report pmra/pmdec directly from vx/vy without further rotation). Gaia's
RA_ERROR/DEC_ERROR are true-angle (mas, already include the cos(dec) factor
the same way our ra/dec residuals do once scaled by cos(dec)*3600*1000).

Since our mjd_ref (default 57388.0) is ~0.5 day from Gaia's own reference
epoch (J2016.0 = MJD 57388.5), Gaia's positions are optionally propagated by
their own PM to mjd_ref for rigor; at Sculptor's typical ~0.2 mas/yr PMs this
shifts Gaia positions by ~0.0003 mas, i.e. negligible, but the option is
included so that "epoch mismatch" can be explicitly ruled in or out rather
than assumed.

Produces:
  (1) Pull histograms for ra and dec separately, each with a fitted Gaussian
      overlay and reported mean/std -- a nonzero mean here is the direct
      position-space test for a constant frame offset.
  (2) Pull std (and mean) vs Gaia G magnitude, per component, to see whether
      any offset/calibration issue holds across the full magnitude range or
      is magnitude-dependent (e.g. driven by faint-star systematics).
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import numpy.lib.recfunctions as rfn
from astropy import units as u
from astropy.coordinates import SkyCoord
from scipy.stats import norm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from healpix_region_data import load_coadd, load_movers

GAIA_REF_MJD = 57388.5  # J2016.0


def robust_std(pull):
    """Median-absolute-deviation sigma: robust to the heavy non-Gaussian tail
    from mismatches/blends/real high-PM outliers, unlike the naive std."""
    return 1.4826 * np.median(np.abs(pull - np.median(pull)))


def binned_pull_stats(x, pull, bins):
    edges = np.linspace(*bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.digitize(x, edges) - 1
    std = np.full(len(centers), np.nan)
    mean = np.full(len(centers), np.nan)
    n = np.zeros(len(centers), dtype=int)
    for i in range(len(centers)):
        sel = idx == i
        n[i] = sel.sum()
        if n[i] >= 20:
            std[i] = robust_std(pull[sel])
            mean[i] = np.median(pull[sel])
    return centers, mean, std, n


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--healpix", type=int, nargs="+", required=True, help="Explicit healpix indices to plot")
    parser.add_argument("--pmcatalog-dir", default="/data8/shared/decampm/PMSculptor_8deg_garyb/PMCatalog/")
    parser.add_argument("--gaia-dir", default="/data8/shared/decampm/Gaia/")
    parser.add_argument("--nside", type=int, default=32)
    parser.add_argument("--match-arcsec", type=float, default=0.3)
    parser.add_argument("--max-ruwe", type=float, default=None,
                         help="Drop Gaia matches with RUWE above this (1.4 is the standard "
                              "'reliable single-star astrometric solution' cut)")
    parser.add_argument("--require-unique", action="store_true",
                         help="Drop movers whose 2nd-nearest Gaia neighbor is also within --match-arcsec "
                              "(ambiguous/crowded match)")
    parser.add_argument("--propagate-gaia-epoch", action="store_true",
                         help="Propagate Gaia RA/DEC by their own PM from J2016.0 to --mjd-ref before "
                              "differencing (negligible at typical Sculptor PMs, included for rigor)")
    parser.add_argument("--mjd-ref", type=float, default=57388.0,
                         help="Our fit's reference epoch (must match config['mjd_ref'] used to produce "
                              "the PMCatalog); only used if --propagate-gaia-epoch is set")
    parser.add_argument("--pm-pattern", default="real_PM_hp*.fits")
    parser.add_argument("--gaia-pattern", default="GaiaSource_*.fits")
    parser.add_argument("--pull-range", type=float, default=8.0, help="Histogram +/- range in sigma")
    parser.add_argument("--gmag-range", type=float, nargs=2, default=None,
                         help="Restrict to matched stars with GMAG_MIN <= PHOT_G_MEAN_MAG <= GMAG_MAX")
    parser.add_argument("-o", "--out", default="gaia_position_error_pull.png")
    parser.add_argument("--title", default="Sculptor 8 degree radius (46 GPR2-complete healpix)")
    args = parser.parse_args()

    pixels = set(args.healpix)
    movers = load_movers(args.pmcatalog_dir, args.nside, pixels,
                          pattern=args.pm_pattern, exts=("modest1_movers",))
    gaia = load_coadd(args.gaia_dir, pixels, pattern=args.gaia_pattern)
    gaia = gaia[~np.isnan(gaia["RA"]) & ~np.isnan(gaia["DEC"])]

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

    if args.gmag_range is not None:
        gmin, gmax = args.gmag_range
        keep = (matched["PHOT_G_MEAN_MAG"] >= gmin) & (matched["PHOT_G_MEAN_MAG"] <= gmax)
        print(f"[gmag] dropping {(~keep).sum()} / {len(matched)} outside {gmin} <= G <= {gmax}")
        matched = matched[keep]

    gaia_ra = matched["RA"].copy()
    gaia_dec = matched["DEC"].copy()
    if args.propagate_gaia_epoch:
        dt_yr = (args.mjd_ref - GAIA_REF_MJD) / 365.25
        has_pm = ~np.isnan(matched["PMRA"]) & ~np.isnan(matched["PMDEC"])
        cosdec = np.cos(np.radians(gaia_dec))
        gaia_ra[has_pm] += matched["PMRA"][has_pm] * dt_yr / 3.6e6 / cosdec[has_pm]
        gaia_dec[has_pm] += matched["PMDEC"][has_pm] * dt_yr / 3.6e6
        print(f"[epoch] propagated {has_pm.sum()} Gaia positions by dt={dt_yr * 365.25:.2f} days "
              f"(J2016.0 -> mjd_ref={args.mjd_ref})")

    cosdec = np.cos(np.radians(matched["dec"]))
    delta_ra_mas = (matched["ra"] - gaia_ra) * cosdec * 3.6e6   # mas, true angle
    delta_dec_mas = (matched["dec"] - gaia_dec) * 3.6e6         # mas

    our_sigma_ra = 1000 * np.sqrt(matched["c_xx"])    # mas
    our_sigma_dec = 1000 * np.sqrt(matched["c_yy"])   # mas
    gaia_sigma_ra = matched["RA_ERROR"]               # mas
    gaia_sigma_dec = matched["DEC_ERROR"]             # mas

    pull_ra = delta_ra_mas / np.hypot(our_sigma_ra, gaia_sigma_ra)
    pull_dec = delta_dec_mas / np.hypot(our_sigma_dec, gaia_sigma_dec)

    finite_ra = np.isfinite(pull_ra)
    finite_dec = np.isfinite(pull_dec)
    pull_ra, pull_dec = pull_ra[finite_ra], pull_dec[finite_dec]
    gmag_ra = matched["PHOT_G_MEAN_MAG"][finite_ra]
    gmag_dec = matched["PHOT_G_MEAN_MAG"][finite_dec]

    for label, pull, delta, sigma_our, sigma_gaia in (
            ("ra", pull_ra, delta_ra_mas[finite_ra], our_sigma_ra[finite_ra], gaia_sigma_ra[finite_ra]),
            ("dec", pull_dec, delta_dec_mas[finite_dec], our_sigma_dec[finite_dec], gaia_sigma_dec[finite_dec])):
        frac1 = np.mean(np.abs(pull) < 1.0)
        frac2 = np.mean(np.abs(pull) < 2.0)
        frac5 = np.mean(np.abs(pull) > 5.0)
        w = 1.0 / (sigma_our ** 2 + sigma_gaia ** 2)
        wmean_mas = np.sum(w * delta) / np.sum(w)
        wmean_err_mas = 1.0 / np.sqrt(np.sum(w))
        print(f"{label:5s} offset: inv-var-weighted mean = {wmean_mas:+.4f} +/- {wmean_err_mas:.4f} mas "
              f"(median = {np.median(delta):+.4f} mas)")
        print(f"{label:5s} pull: mean={np.mean(pull):.3f}, median={np.median(pull):.3f}, "
              f"naive std={np.std(pull):.3f}, robust std={robust_std(pull):.3f}, n={len(pull)}, "
              f"frac(|pull|<1)={frac1:.3f} (ideal 0.683), frac(|pull|<2)={frac2:.3f} (ideal 0.954), "
              f"frac(|pull|>5)={frac5:.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for ax, pull, label in zip(axes[0], (pull_ra, pull_dec), ("ra", "dec")):
        rng = args.pull_range
        bins = np.linspace(-rng, rng, 121)
        rmed, rstd = np.median(pull), robust_std(pull)
        ax.hist(pull, bins=bins, density=True, color="steelblue", alpha=0.7, label="measured")
        x = np.linspace(-rng, rng, 400)
        ax.plot(x, norm.pdf(x, 0, 1), "k--", lw=2, label=r"$\mathcal{N}(0,1)$ (ideal)")
        ax.plot(x, norm.pdf(x, rmed, rstd), "r-", lw=2,
                label=fr"$\mathcal{{N}}({rmed:.2f}, {rstd:.2f}^2)$ (robust fit)")
        ax.set_yscale("log")
        ax.set_ylim(1e-5, 1)
        ax.set_xlabel(f"{label} pull " + r"$(x_{our}-x_{gaia})/\sqrt{\sigma_{our}^2+\sigma_{gaia}^2}$", fontsize=11)
        ax.set_ylabel("density")
        ax.set_title(f"{label}: mean={np.mean(pull):.2f}, naive std={np.std(pull):.2f} (outlier-driven), "
                     f"robust std={rstd:.2f}, n={len(pull)}")
        ax.legend(fontsize=9)
        ax.grid()

    for ax, gmag, pull, label in zip(axes[1], (gmag_ra, gmag_dec), (pull_ra, pull_dec), ("ra", "dec")):
        centers, mean, std, n = binned_pull_stats(gmag, pull, (12, 21, 37))
        ax.plot(centers, std, "o-", color="steelblue", label="robust std")
        ax.plot(centers, mean, "s--", color="darkorange", label="median (offset)")
        ax.axhline(1.0, color="k", ls="--", lw=1.5)
        ax.axhline(0.0, color="gray", ls=":", lw=1.0)
        ax.set_xlabel("Gaia G magnitude", fontsize=11)
        ax.set_ylabel(f"{label} pull std / mean")
        ax.set_ylim(-1, max(3, np.nanmax(std) * 1.1 if np.any(np.isfinite(std)) else 3))
        ax.legend(fontsize=9)
        ax.grid()

    fig.suptitle(args.title, fontsize=14)
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"Saved {args.out}")
