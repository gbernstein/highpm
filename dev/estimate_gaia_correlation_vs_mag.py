"""What Gaia/our PM error correlation rho(G) would explain the pull std vs G?

If our and Gaia's PM errors have correlation rho, then for the pull built with
the independence denominator (plot_gaia_pm_error_pull.py)
    Var(pull) = 1 - rho * w,   w = 2*sigma_our*sigma_gaia / (sigma_our^2 + sigma_gaia^2)
so per G bin, assuming both sets of formal errors are right,
    rho(G) = (1 - s^2) / <w>,   s = robust pull std in the bin.
w <= 1, and |rho| <= 1 bounds how much of the deficit correlation can explain:
bins needing rho > 1 can't be explained by correlation alone.

For comparison, the alternative explanation -- no correlation, our errors
scaled by k -- needs
    k(G)^2 = (s^2 - 1 + f) / f,   f = <sigma_our^2 / (sigma_our^2 + sigma_gaia^2)>.
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--healpix", type=int, nargs="+", required=True)
    parser.add_argument("--pmcatalog-dir", default="/data8/shared/decampm/PMSculptor_8deg_garyb/PMCatalog/")
    parser.add_argument("--gaia-dir", default="/data8/shared/decampm/Gaia/")
    parser.add_argument("--nside", type=int, default=32)
    parser.add_argument("--match-arcsec", type=float, default=0.3)
    parser.add_argument("--max-ruwe", type=float, default=1.4)
    parser.add_argument("--pm-pattern", default="real_PM_hp*.fits")
    parser.add_argument("--gaia-pattern", default="GaiaSource_*.fits")
    parser.add_argument("--gmag-bins", type=float, nargs=3, default=(12, 21, 37), help="linspace(lo, hi, n) edges")
    parser.add_argument("-o", "--out", default="gaia_correlation_vs_mag.png")
    parser.add_argument("--title", default="Sculptor 8 degree radius (46 GPR2-complete healpix)")
    args = parser.parse_args()

    pixels = set(args.healpix)
    movers = load_movers(args.pmcatalog_dir, args.nside, pixels,
                          pattern=args.pm_pattern, exts=("modest1_movers",))
    gaia = load_coadd(args.gaia_dir, pixels, pattern=args.gaia_pattern)
    gaia = gaia[~np.isnan(gaia["PMRA"])]
    # 2-parameter solutions are stored as zeros, not NaN (see plot_gaia_pm_error_pull.py).
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

    edges = np.linspace(*args.gmag_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)

    for col, comp, var_col, gerr_col, gmu_col in zip(
        range(2), ("pmra", "pmdec"), ("c_vxvx", "c_vyvy"), ("PMRA_ERROR", "PMDEC_ERROR"), ("PMRA", "PMDEC"),
    ):
        s_our = 1000 * np.sqrt(matched[var_col])  # mas/yr
        s_gaia = matched[gerr_col]
        pull = (matched[comp] - matched[gmu_col]) / np.hypot(s_our, s_gaia)
        ok = np.isfinite(pull)
        s_our, s_gaia, pull, gmag = s_our[ok], s_gaia[ok], pull[ok], matched["PHOT_G_MEAN_MAG"][ok]

        w = 2 * s_our * s_gaia / (s_our**2 + s_gaia**2)
        f = s_our**2 / (s_our**2 + s_gaia**2)

        nb = len(centers)
        s, wbar, fbar, med_our, med_gaia = (np.full(nb, np.nan) for _ in range(5))
        n = np.zeros(nb, dtype=int)
        ib = np.digitize(gmag, edges) - 1
        for i in range(nb):
            sel = ib == i
            n[i] = sel.sum()
            if n[i] < 20:
                continue
            s[i] = robust_std(pull[sel])
            wbar[i] = w[sel].mean()
            fbar[i] = f[sel].mean()
            med_our[i] = np.median(s_our[sel])
            med_gaia[i] = np.median(s_gaia[sel])
        rho = (1 - s**2) / wbar
        k = np.sqrt((s**2 - 1 + fbar) / fbar)

        print(f"\n{comp}:  G     n      s    <w>    rho_needed   k_needed   med sig_our  med sig_gaia [mas/yr]")
        for i in range(nb):
            if np.isfinite(s[i]):
                print(f"      {centers[i]:5.2f} {n[i]:6d}  {s[i]:.3f}  {wbar[i]:.3f}  {rho[i]:8.3f}   {k[i]:8.3f}"
                      f"   {med_our[i]:9.3f}   {med_gaia[i]:9.3f}")

        ax = axes[0, col]
        ax.semilogy(centers, med_our, "o-", color="steelblue", label=r"median $\sigma_{our}$")
        ax.semilogy(centers, med_gaia, "s-", color="darkorange", label=r"median $\sigma_{gaia}$")
        ax.set_ylabel(f"{comp} error [mas/yr]")
        ax.set_title(comp)
        ax.legend(fontsize=9)
        ax.grid(which="both", alpha=0.5)

        ax = axes[1, col]
        ax.plot(centers, rho, "o-", color="firebrick", label=r"$\rho$ needed $=(1-s^2)/\langle w\rangle$")
        ax.plot(centers, wbar, "--", color="gray", label=r"$\langle w\rangle$")
        ax.axhline(1, color="k", ls=":", lw=1)
        ax.axhline(0, color="k", ls="-", lw=0.8)
        ax.set_ylabel(r"$\rho$ needed")
        ax.set_ylim(-0.1, 3)  # bright bins need rho >> 1; clip so the rho=1 crossing is visible
        ax.legend(fontsize=9)
        ax.grid()

        ax = axes[2, col]
        ax.plot(centers, k, "o-", color="seagreen", label=r"$k$ needed ($\sigma_{our}\to k\,\sigma_{our}$, $\rho=0$)")
        ax.axhline(1, color="k", ls="--", lw=1)
        ax.set_ylabel("our-error scale k")
        ax.set_xlabel("Gaia G magnitude")
        ax.legend(fontsize=9)
        ax.grid()

    fig.suptitle(args.title, fontsize=14)
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"Saved {args.out}")
