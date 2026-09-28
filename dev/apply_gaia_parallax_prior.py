"""Use Gaia's parallax measurement as an informative prior on OUR fit's
parallax parameter, to see whether tightening the parallax constraint (and
thereby breaking some of the parallax/proper-motion degeneracy in our own
covariance matrix) improves our per-component formal PM error calibration.

Our 5-parameter fit (highpm.pmfit.singleFit) solves for (x0, y0, vx, vy, pi)
jointly with a weak Gaussian prior on parallax (config.yaml: parallax_prior =
0.15 arcsec std, i.e. nearly uninformative). The full posterior covariance is
stored per-star in modest1_movers as c_vxvx, c_vyvy, c_pipi, c_vxvy, c_vxpi,
c_vypi (the marginal covariance of (vx, vy, pi) after marginalizing over
position/color). Proper motion and parallax are correlated through the
parallax-factor/time design matrix (this is the classic PM/parallax
degeneracy for limited time baselines), so a tighter, off-the-shelf parallax
constraint from Gaia should, in principle, shrink our own PM covariance too.

Because the model is linear-Gaussian, this can be done exactly from the
stored (mean, covariance) alone -- no need to refit from raw detections:

  1. Let C3 be the stored 3x3 marginal covariance of (vx, vy, pi), and
     P3 = inv(C3) the marginal precision. Schur-complement marginalization
     commutes with adding a diagonal term restricted to the pi row/column
     (the position/color block is untouched by that addition), so:
         P3 = P3_likelihood + diag(0, 0, 1/parallax_prior_old^2)
     i.e. we can strip out our own pipeline's existing (weak) parallax prior
     exactly by subtracting its precision contribution from the pi-pi entry.
  2. Add Gaia's parallax as the new prior:
         P3_new = P3_likelihood + diag(0, 0, 1/gaia_parallax_err^2)
  3. The linear term is unaffected by step 1 (old prior mean was 0, so it
     contributed nothing to the normal-equations RHS); the new prior mean is
     Gaia's parallax, so:
         b3_likelihood = P3 @ (vx, vy, pi)_old
         b3_new = b3_likelihood + (0, 0, gaia_parallax / gaia_parallax_err^2)
  4. New posterior: C3_new = inv(P3_new), (vx, vy, pi)_new = C3_new @ b3_new.

Then rebuild the per-component pull vs Gaia PM using the *updated* our_sigma
and our_pm, and compare the calibration (robust pull std) before vs after.
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

MAS_PER_ARCSEC = 1000.0


def robust_std(x):
    """Median-absolute-deviation sigma: robust to the heavy non-Gaussian tail
    from mismatches/blends/real high-PM outliers, unlike the naive std."""
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def report(label, pull):
    frac1 = np.mean(np.abs(pull) < 1.0)
    frac2 = np.mean(np.abs(pull) < 2.0)
    frac5 = np.mean(np.abs(pull) > 5.0)
    print(f"{label:22s} mean={np.mean(pull):+.3f}, naive std={np.std(pull):.3f}, "
          f"robust std={robust_std(pull):.3f}, n={len(pull)}, "
          f"frac(|pull|<1)={frac1:.3f} (ideal 0.683), frac(|pull|<2)={frac2:.3f} (ideal 0.954), "
          f"frac(|pull|>5)={frac5:.4f}")


def binned_pull_std(x, pull, bins):
    edges = np.linspace(*bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.digitize(x, edges) - 1
    std = np.full(len(centers), np.nan)
    err = np.full(len(centers), np.nan)
    n = np.zeros(len(centers), dtype=int)
    for i in range(len(centers)):
        sel = idx == i
        n[i] = sel.sum()
        if n[i] >= 20:
            std[i] = robust_std(pull[sel])
            # Standard error of a std estimate under approximate normality.
            err[i] = std[i] / np.sqrt(2 * (n[i] - 1))
    return centers, std, err, n


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--healpix", type=int, nargs="+", required=True)
    parser.add_argument("--pmcatalog-dir", default="/data8/shared/decampm/PMSculptor_8deg_garyb/PMCatalog/")
    parser.add_argument("--gaia-dir", default="/data8/shared/decampm/Gaia/")
    parser.add_argument("--nside", type=int, default=32)
    parser.add_argument("--match-arcsec", type=float, default=0.3)
    parser.add_argument("--max-ruwe", type=float, default=1.4)
    parser.add_argument("--parallax-prior-old", type=float, default=0.15,
                         help="Std (arcsec) of the parallax prior already baked into our fit "
                              "(config.yaml fitting.parallax_prior)")
    parser.add_argument("--pm-pattern", default="real_PM_hp*.fits")
    parser.add_argument("--gaia-pattern", default="GaiaSource_*.fits")
    parser.add_argument("--pull-range", type=float, default=8.0)
    parser.add_argument("--gmag-min", type=float, default=12.0)
    parser.add_argument("--gmag-max", type=float, default=None,
                         help="Bright/faint edges for the magnitude-binned panels. "
                              "Default: extend to the faintest matched Gaia source.")
    parser.add_argument("-o", "--out", default="gaia_parallax_prior_pull.png")
    parser.add_argument("--title", default="Sculptor 8 degree radius (46 GPR2-complete healpix)")
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
    matched = rfn.merge_arrays([movers[close], gaia[idx[close]]], flatten=True, usemask=False)
    print(f"[match] {len(movers)} modest1 movers, {len(matched)} matched to Gaia within {args.match_arcsec} arcsec")

    if args.max_ruwe is not None:
        keep = matched["RUWE"] <= args.max_ruwe
        print(f"[ruwe] dropping {(~keep).sum()} / {len(matched)} with RUWE > {args.max_ruwe}")
        matched = matched[keep]

    # --- Build our stored (vx, vy, pi) mean and 3x3 covariance, arcsec units ---
    vx = matched["pmra"] / MAS_PER_ARCSEC  # back to arcsec/yr (pmra/pmdec are *1000 in fits_writer)
    vy = matched["pmdec"] / MAS_PER_ARCSEC
    pi_ours = matched["parallax"]  # already arcsec

    C3 = np.zeros((len(matched), 3, 3))
    C3[:, 0, 0] = matched["c_vxvx"]
    C3[:, 1, 1] = matched["c_vyvy"]
    C3[:, 2, 2] = matched["c_pipi"]
    C3[:, 0, 1] = C3[:, 1, 0] = matched["c_vxvy"]
    C3[:, 0, 2] = C3[:, 2, 0] = matched["c_vxpi"]
    C3[:, 1, 2] = C3[:, 2, 1] = matched["c_vypi"]

    good = np.all(np.isfinite(C3.reshape(len(matched), -1)), axis=1)
    print(f"[cov] dropping {(~good).sum()} / {len(matched)} with non-finite covariance entries")

    P3 = np.linalg.inv(C3[good])
    mean_old = np.stack([vx[good], vy[good], pi_ours[good]], axis=-1)
    b3 = np.einsum("kij,kj->ki", P3, mean_old)

    # Strip our own (weak) parallax prior out of the pi-pi precision entry --
    # its contribution to b3 is zero since its prior mean is 0.
    old_prior_precision = args.parallax_prior_old**-2
    P3_likelihood = P3.copy()
    P3_likelihood[:, 2, 2] -= old_prior_precision

    # Add Gaia's parallax as the new prior (mas -> arcsec).
    gaia_pi = matched["PARALLAX"][good] / MAS_PER_ARCSEC
    gaia_pi_err = matched["PARALLAX_ERROR"][good] / MAS_PER_ARCSEC
    new_prior_precision = gaia_pi_err**-2

    P3_new = P3_likelihood.copy()
    P3_new[:, 2, 2] += new_prior_precision
    b3_new = b3.copy()
    b3_new[:, 2] += new_prior_precision * gaia_pi

    C3_new = np.linalg.inv(P3_new)
    mean_new = np.einsum("kij,kj->ki", C3_new, b3_new)

    matched = matched[good]
    our_pmra_new = mean_new[:, 0] * MAS_PER_ARCSEC
    our_pmdec_new = mean_new[:, 1] * MAS_PER_ARCSEC
    our_sigma_pmra_new = MAS_PER_ARCSEC * np.sqrt(C3_new[:, 0, 0])
    our_sigma_pmdec_new = MAS_PER_ARCSEC * np.sqrt(C3_new[:, 1, 1])

    our_sigma_pmra_old = MAS_PER_ARCSEC * np.sqrt(matched["c_vxvx"])
    our_sigma_pmdec_old = MAS_PER_ARCSEC * np.sqrt(matched["c_vyvy"])

    print(f"[parallax] median our sigma_pi (old, mas) = "
          f"{MAS_PER_ARCSEC * np.median(np.sqrt(C3[good][:, 2, 2])):.3f}, "
          f"median Gaia sigma_pi (mas) = {MAS_PER_ARCSEC * np.median(gaia_pi_err):.3f}")
    print(f"[pm sigma shrinkage] median our_sigma_pmra: {np.median(our_sigma_pmra_old):.3f} -> "
          f"{np.median(our_sigma_pmra_new):.3f} mas/yr "
          f"({100 * (1 - np.median(our_sigma_pmra_new / our_sigma_pmra_old)):.1f}% smaller)")
    print(f"[pm sigma shrinkage] median our_sigma_pmdec: {np.median(our_sigma_pmdec_old):.3f} -> "
          f"{np.median(our_sigma_pmdec_new):.3f} mas/yr "
          f"({100 * (1 - np.median(our_sigma_pmdec_new / our_sigma_pmdec_old)):.1f}% smaller)")

    gaia_sigma_pmra = matched["PMRA_ERROR"]
    gaia_sigma_pmdec = matched["PMDEC_ERROR"]

    pull_ra_old = (matched["pmra"] - matched["PMRA"]) / np.hypot(our_sigma_pmra_old, gaia_sigma_pmra)
    pull_dec_old = (matched["pmdec"] - matched["PMDEC"]) / np.hypot(our_sigma_pmdec_old, gaia_sigma_pmdec)
    pull_ra_new = (our_pmra_new - matched["PMRA"]) / np.hypot(our_sigma_pmra_new, gaia_sigma_pmra)
    pull_dec_new = (our_pmdec_new - matched["PMDEC"]) / np.hypot(our_sigma_pmdec_new, gaia_sigma_pmdec)

    gmag = matched["PHOT_G_MEAN_MAG"]

    report("pmra  (before)", pull_ra_old)
    report("pmra  (gaia-pi prior)", pull_ra_new)
    report("pmdec (before)", pull_dec_old)
    report("pmdec (gaia-pi prior)", pull_dec_new)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for ax, pull_old, pull_new, label in zip(axes[0], (pull_ra_old, pull_dec_old),
                                              (pull_ra_new, pull_dec_new), ("pmra", "pmdec")):
        rng = args.pull_range
        bins = np.linspace(-rng, rng, 121)
        ax.hist(pull_old, bins=bins, density=True, color="gray", alpha=0.5,
                label=f"before (robust std={robust_std(pull_old):.2f})")
        ax.hist(pull_new, bins=bins, density=True, color="steelblue", alpha=0.6,
                label=f"Gaia-parallax prior (robust std={robust_std(pull_new):.2f})")
        x = np.linspace(-rng, rng, 400)
        ax.plot(x, norm.pdf(x, 0, 1), "k--", lw=2, label=r"$\mathcal{N}(0,1)$ (ideal)")
        ax.set_yscale("log")
        ax.set_ylim(1e-5, 1)
        ax.set_xlabel(f"{label} pull", fontsize=11)
        ax.set_ylabel("density")
        ax.set_title(label)
        ax.legend(fontsize=9)
        ax.grid()

    gmag_max = args.gmag_max if args.gmag_max is not None else float(np.ceil(np.nanmax(gmag) * 2) / 2)
    nbins = max(1, round((gmag_max - args.gmag_min) / 0.25))
    print(f"[gmag] binning pull std over G in [{args.gmag_min}, {gmag_max}] with {nbins} bins")

    for ax, pull_old, pull_new, label in zip(axes[1], (pull_ra_old, pull_dec_old),
                                              (pull_ra_new, pull_dec_new), ("pmra", "pmdec")):
        c_old, s_old, e_old, n_old = binned_pull_std(gmag, pull_old, (args.gmag_min, gmag_max, nbins + 1))
        c_new, s_new, e_new, n_new = binned_pull_std(gmag, pull_new, (args.gmag_min, gmag_max, nbins + 1))
        ax.errorbar(c_old, s_old, yerr=e_old, fmt="o", color="gray", label="before")
        ax.errorbar(c_new, s_new, yerr=e_new, fmt="o", color="steelblue", label="Gaia-parallax prior")
        ax.axhline(1.0, color="k", ls="--", lw=1.5, label="ideal (std=1)")
        ax.set_xlabel("Gaia G magnitude", fontsize=11)
        ax.set_ylabel(f"{label} robust pull std")
        ax.set_xlim(args.gmag_min, gmag_max)
        ax.set_ylim(0.7, 1.1)
        ax.legend(fontsize=9)
        ax.grid()

    fig.suptitle(args.title, fontsize=14)
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"Saved {args.out}")
