"""PM pull against Gaia vs G magnitude, split into GPR reference and
non-reference stars.

For each modest1 mover matched to Gaia, the per-component pull

    pull_c = (our_pmc - gaia_pmc) / sqrt(our_sigma_c^2 + gaia_sigma_c^2)

is binned in Gaia G. Each bin shows the median pull and the robust (MAD)
std, both with bootstrap 1-sigma errorbars. A star counts as a GPR reference
star if at least --min-ref-frac of its unclipped detections were ref=True in
the GPR catalog of their exposure. Gaia stars with an exactly-zero PMRA or
PMDEC (placeholder 2-parameter solutions in this dump) are dropped.
"""

import argparse
import glob
import multiprocessing as mp
import os
import sys

import fitsio
import matplotlib.pyplot as plt
import numpy as np
import numpy.lib.recfunctions as rfn
from astropy import units as u
from astropy.coordinates import SkyCoord
from healpy import ang2pix

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from healpix_region_data import load_coadd, load_movers
from plot_gpr_vs_gaia_per_exposure import _expnum, ref_keys, robust_std


def ref_fraction(movers, pixels, pmcatalog_dir, allkeys):
    """Fraction of each mover's unclipped detections that GPR flagged ref=True."""
    run_dir = os.path.dirname(pmcatalog_dir.rstrip("/"))
    frac = {}
    for hp in pixels:
        det = fitsio.read(os.path.join(pmcatalog_dir, f"real_PM_hp{hp:05d}.fits"), ext="modest1_detections")
        det = det[~det["clipped"]]
        c = fitsio.read(os.path.join(run_dir, "HealpixDetectionCatalog", f"cleaned_detections_hp{hp:05d}.fits"),
                        columns=["EXPNUM", "CCDNUM", "OBJECT_NUMBER", "TRAP_FLAG"])[det["detections"]]
        # clean_cat drops TRAP_FLAG detections before fitting; make sure this run did.
        assert not c["TRAP_FLAG"].any(), f"hp{hp:05d}: {c['TRAP_FLAG'].sum()} fitted detections are trap-flagged"
        k =(c["EXPNUM"].astype(np.int64) * 10**10 + c["CCDNUM"].astype(np.int64) * 10**8
             + c["OBJECT_NUMBER"].astype(np.int64))
        isref = np.isin(k, allkeys).astype(float)
        n = int(det["idx"].max()) + 1
        frac[hp] = np.bincount(det["idx"], isref, n) / np.maximum(np.bincount(det["idx"], minlength=n), 1)
    hpix = ang2pix(32, movers["ra"], movers["dec"], lonlat=True)
    return np.array([frac[h][i] if h in frac and i < len(frac[h]) else np.nan
                     for h, i in zip(hpix.astype(int), movers["idx"].astype(int))])


def binned(x, pull, edges, nboot, rng, min_n=20):
    """Per-bin median and robust std of pull, each with a bootstrap error."""
    nb = len(edges) - 1
    out = {k: np.full(nb, np.nan) for k in ("med", "med_err", "std", "std_err")}
    idx = np.digitize(x, edges) - 1
    for i in range(nb):
        p = pull[idx == i]
        if len(p) < min_n:
            continue
        boot = p[rng.integers(0, len(p), size=(nboot, len(p)))]
        bmed = np.median(boot, axis=1)
        bstd = 1.4826 * np.median(np.abs(boot - bmed[:, None]), axis=1)
        out["med"][i], out["std"][i] = np.median(p), robust_std(p)
        out["med_err"][i], out["std_err"][i] = np.std(bmed), np.std(bstd)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", default="/data8/shared/decampm/PMSculptor_8deg_garyb/")
    parser.add_argument("--healpix-npy", default=None, help="Defaults to <run-dir>/pmsculptor8deg_healpix.npy")
    parser.add_argument("--gpr-dirs", nargs="+", default=[f"/data8/shared/decampm/GPR2/CAT/{b}/" for b in "griz"])
    parser.add_argument("--gaia-dir", default="/data8/shared/decampm/Gaia/")
    parser.add_argument("--min-ref-frac", type=float, default=0.5)
    parser.add_argument("--match-arcsec", type=float, default=0.3)
    parser.add_argument("--max-ruwe", type=float, default=None)
    parser.add_argument("--gmag-bins", type=float, nargs=3, default=(12, 21, 19), help="linspace(start, stop, num) edges")
    parser.add_argument("--nboot", type=int, default=300)
    parser.add_argument("--nproc", type=int, default=24)
    parser.add_argument("--keys-cache", default=None, help="Cache the GPR ref-star keys to this .npy")
    parser.add_argument("-o", "--out", default="gaia_pm_pull_ref_vs_nonref.png")
    args = parser.parse_args()

    pmcatalog_dir = os.path.join(args.run_dir, "PMCatalog")
    pixels = set(int(h) for h in np.load(args.healpix_npy or os.path.join(args.run_dir, "pmsculptor8deg_healpix.npy")))

    movers = load_movers(pmcatalog_dir, 32, pixels, pattern="real_PM_hp*.fits", exts=("modest1_movers",))

    if args.keys_cache and os.path.exists(args.keys_cache):
        allkeys = np.load(args.keys_cache)
    else:
        files = sorted(glob.glob(os.path.join(args.run_dir, "PositionCorrectedExposureCatalog", "position_corrected_*.fits")))
        with mp.get_context("fork").Pool(args.nproc) as pool:
            allkeys = np.sort(np.concatenate(pool.starmap(ref_keys, [(_expnum(f), args.gpr_dirs) for f in files])))
        if args.keys_cache:
            np.save(args.keys_cache, allkeys)
    print(f"[ref] {len(allkeys)} GPR ref=True detections")
    movers = rfn.append_fields(movers, "ref_frac", ref_fraction(movers, pixels, pmcatalog_dir, allkeys), usemask=False)

    gaia = load_coadd(args.gaia_dir, pixels, pattern="GaiaSource_*.fits")
    gaia = gaia[np.isfinite(gaia["PMRA"]) & np.isfinite(gaia["PMDEC"])]
    zero_pm = (gaia["PMRA"] == 0) | (gaia["PMDEC"] == 0)
    print(f"[gaia] dropping {zero_pm.sum()} / {len(gaia)} with PMRA or PMDEC exactly 0")
    gaia = gaia[~zero_pm]
    if args.max_ruwe is not None:
        gaia = gaia[gaia["RUWE"] <= args.max_ruwe]

    idx, d2d, _ = SkyCoord(ra=movers["ra"] * u.degree, dec=movers["dec"] * u.degree).match_to_catalog_sky(
        SkyCoord(ra=gaia["RA"] * u.degree, dec=gaia["DEC"] * u.degree))
    close = d2d.arcsecond < args.match_arcsec
    m = rfn.merge_arrays([movers[close], gaia[idx[close]]], flatten=True, usemask=False)
    pull = {"pmra": (m["pmra"] - m["PMRA"]) / np.hypot(1000 * np.sqrt(m["c_vxvx"]), m["PMRA_ERROR"]),
            "pmdec": (m["pmdec"] - m["PMDEC"]) / np.hypot(1000 * np.sqrt(m["c_vyvy"]), m["PMDEC_ERROR"])}
    ok = np.isfinite(pull["pmra"]) & np.isfinite(pull["pmdec"]) & np.isfinite(m["ref_frac"])
    m, pull = m[ok], {k: v[ok] for k, v in pull.items()}
    isref = m["ref_frac"] >= args.min_ref_frac
    print(f"[match] {len(m)} matched to Gaia within {args.match_arcsec} arcsec: "
          f"{isref.sum()} ref, {(~isref).sum()} non-ref")

    edges = np.linspace(*args.gmag_bins[:2], int(args.gmag_bins[2]))
    centers = 0.5 * (edges[:-1] + edges[1:])
    rng = np.random.default_rng(0)
    groups = ((isref, "firebrick", "o", -0.06, "GPR reference stars"),
              (~isref, "steelblue", "s", 0.06, "non-reference stars"))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for j, comp in enumerate(("pmra", "pmdec")):
        for sel, color, marker, dx, name in groups:
            b = binned(m["PHOT_G_MEAN_MAG"][sel], pull[comp][sel], edges, args.nboot, rng)
            label = f"{name} (n={sel.sum()})"
            axes[0, j].errorbar(centers + dx, b["med"], b["med_err"], fmt=marker + "-", color=color, ms=5, capsize=2, label=label)
            axes[1, j].errorbar(centers + dx, b["std"], b["std_err"], fmt=marker + "-", color=color, ms=5, capsize=2, label=label)
            for i in range(len(centers)):
                print(f"{comp:5s} {name:20s} G={centers[i]:5.2f} n={np.sum(sel & (np.digitize(m['PHOT_G_MEAN_MAG'], edges) - 1 == i)):6d} "
                      f"median={b['med'][i]:+.3f}+-{b['med_err'][i]:.3f} robust_std={b['std'][i]:.3f}+-{b['std_err'][i]:.3f}")
        axes[0, j].axhline(0, color="k", ls="--", lw=1)
        axes[1, j].axhline(1, color="k", ls="--", lw=1)
        axes[0, j].set_title(comp)
        axes[0, j].set_ylabel(f"{comp} pull median")
        axes[1, j].set_ylabel(f"{comp} pull robust std")
        axes[1, j].set_xlabel("Gaia G magnitude")
        for ax in axes[:, j]:
            ax.axvspan(edges[0], 16, color="0.85", zorder=0)
            ax.grid(alpha=0.4)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle(r"Sculptor 8 deg: PM pull vs Gaia, $(\mu_{our}-\mu_{gaia})/\sqrt{\sigma_{our}^2+\sigma_{gaia}^2}$"
                 f"\nref = ref_frac >= {args.min_ref_frac}; Gaia PMRA/PMDEC == 0 removed; "
                 f"errorbars: bootstrap 1$\\sigma$; shaded G<16", fontsize=11)
    fig.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Saved {args.out}")
