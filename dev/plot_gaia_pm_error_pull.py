"""Per-component formal-error accuracy check against Gaia.

Assumes Gaia's PMRA/PMDEC and their formal errors are correct, and asks:
for the same stars, are OUR formal errors (from modest1_movers) calibrated
correctly? For each component c in {ra, dec}, the pull

    pull_c = (our_pmc - gaia_pmc) / sqrt(our_sigma_c^2 + gaia_sigma_c^2)

should be a standard normal (mean 0, std 1) if both sets of formal errors
are accurate and the measurements agree. A pull std > 1 means our formal
errors are UNDERestimated (too optimistic); < 1 means OVERestimated.

Produces:
  (1) Pull histograms for pmra and pmdec separately, each with a fitted
      Gaussian overlay and reported mean/std.
  (2) Pull std vs Gaia G magnitude, per component, to see whether the
      calibration holds across the full magnitude range.
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


def robust_std(pull):
    """Median-absolute-deviation sigma: robust to the heavy non-Gaussian tail
    from mismatches/blends/real high-PM outliers, unlike the naive std."""
    return 1.4826 * np.median(np.abs(pull - np.median(pull)))


def binned_pull_std(x, pull, bins):
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
    return centers, std, mean, n


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--pm-pattern", default="real_PM_hp*.fits")
    parser.add_argument("--gaia-pattern", default="GaiaSource_*.fits")
    parser.add_argument("--pull-range", type=float, default=8.0, help="Histogram +/- range in sigma")
    parser.add_argument("--gmag-range", type=float, nargs=2, default=None,
                         help="Restrict to matched stars with GMAG_MIN <= PHOT_G_MEAN_MAG <= GMAG_MAX")
    parser.add_argument("-o", "--out", default="gaia_pm_error_pull.png")
    parser.add_argument("--title", default="Sculptor 8 degree radius (46 GPR2-complete healpix)")
    args = parser.parse_args()

    pixels = set(args.healpix)
    movers = load_movers(args.pmcatalog_dir, args.nside, pixels,
                          pattern=args.pm_pattern, exts=("modest1_movers",))
    gaia = load_coadd(args.gaia_dir, pixels, pattern=args.gaia_pattern)
    gaia = gaia[~np.isnan(gaia["PMRA"])]
    # This local Gaia dump fills 2-parameter (position-only) solutions with
    # placeholder zeros for PMRA/PMDEC/PARALLAX/RUWE instead of NaN, rather
    # than leaving them out -- those aren't real "PM=0" measurements, so a
    # RUWE cut alone lets them through (0 <= 1.4). Drop them explicitly.
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

    if args.gmag_range is not None:
        gmin, gmax = args.gmag_range
        keep = (matched["PHOT_G_MEAN_MAG"] >= gmin) & (matched["PHOT_G_MEAN_MAG"] <= gmax)
        print(f"[gmag] dropping {(~keep).sum()} / {len(matched)} outside {gmin} <= G <= {gmax}")
        matched = matched[keep]

    our_sigma_pmra = 1000 * np.sqrt(matched["c_vxvx"])   # mas/yr
    our_sigma_pmdec = 1000 * np.sqrt(matched["c_vyvy"])  # mas/yr
    gaia_sigma_pmra = matched["PMRA_ERROR"]              # mas/yr
    gaia_sigma_pmdec = matched["PMDEC_ERROR"]            # mas/yr

    pull_ra = (matched["pmra"] - matched["PMRA"]) / np.hypot(our_sigma_pmra, gaia_sigma_pmra)
    pull_dec = (matched["pmdec"] - matched["PMDEC"]) / np.hypot(our_sigma_pmdec, gaia_sigma_pmdec)

    finite_ra = np.isfinite(pull_ra)
    finite_dec = np.isfinite(pull_dec)
    pull_ra, pull_dec = pull_ra[finite_ra], pull_dec[finite_dec]
    gmag_ra = matched["PHOT_G_MEAN_MAG"][finite_ra]
    gmag_dec = matched["PHOT_G_MEAN_MAG"][finite_dec]

    for label, pull in (("pmra", pull_ra), ("pmdec", pull_dec)):
        frac1 = np.mean(np.abs(pull) < 1.0)
        frac2 = np.mean(np.abs(pull) < 2.0)
        frac5 = np.mean(np.abs(pull) > 5.0)
        print(f"{label:5s} pull: mean={np.mean(pull):.3f}, naive std={np.std(pull):.3f}, "
              f"robust std={robust_std(pull):.3f}, n={len(pull)}, "
              f"frac(|pull|<1)={frac1:.3f} (ideal 0.683), frac(|pull|<2)={frac2:.3f} (ideal 0.954), "
              f"frac(|pull|>5)={frac5:.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for ax, pull, label in zip(axes[0], (pull_ra, pull_dec), ("pmra", "pmdec")):
        rng = args.pull_range
        bins = np.linspace(-rng, rng, 121)
        rstd = robust_std(pull)
        ax.hist(pull, bins=bins, density=True, color="steelblue", alpha=0.7, label="measured")
        x = np.linspace(-rng, rng, 400)
        ax.plot(x, norm.pdf(x, 0, 1), "k--", lw=2, label=r"$\mathcal{N}(0,1)$ (ideal)")
        ax.plot(x, norm.pdf(x, np.median(pull), rstd), "r-", lw=2,
                label=fr"$\mathcal{{N}}({np.median(pull):.2f}, {rstd:.2f}^2)$ (robust fit)")
        ax.set_yscale("log")
        ax.set_ylim(1e-5, 1)
        ax.set_xlabel(f"{label} pull " + r"$(\mu_{our}-\mu_{gaia})/\sqrt{\sigma_{our}^2+\sigma_{gaia}^2}$", fontsize=11)
        ax.set_ylabel("density")
        ax.set_title(f"{label}: naive std={np.std(pull):.2f} (outlier-driven), robust std={rstd:.2f}, n={len(pull)}")
        ax.legend(fontsize=9)
        ax.grid()

    for ax, gmag, pull, label in zip(axes[1], (gmag_ra, gmag_dec), (pull_ra, pull_dec), ("pmra", "pmdec")):
        centers, std, mean, n = binned_pull_std(gmag, pull, (12, 21, 37))
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
