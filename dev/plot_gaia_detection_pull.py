"""Per-DETECTION position pull against Gaia vs G magnitude, split into GPR
reference and non-reference detections.

Every detection in a run's cleaned_detections_hp*.fits that survives
clean_cat (same config and $DES_EXPOSURES as PM.py, so FLAGS/IMAFLAGS/
extendedness/TRAP_FLAG/exclusion regions/exposure thinning all apply) and
lies inside its own healpixel is matched to Gaia propagated to that
detection's epoch, in the fit's own healpix (xi, eta) frame:

    pred = xi_eta(Gaia J2016) + J @ (pmra*, pmdec) * dt + parallax * (PAR_XI, PAR_ETA)

where J is the local gnomonic Jacobian at the Gaia position. Gaia's full 5x5
astrometric covariance is propagated to the epoch as A C A^T with
A = [J | PAR | J*dt], and added to the detection's own (xi, eta) covariance:

    pull_xi = (xi_det - xi_pred) / sqrt(C_det,xixi + C_gaia,xixi)   (same for eta)

A detection counts as a reference detection if GPR flagged it ref=True in
its own exposure. Detections with an unknown color (nonzero DXI/DETA_DCOLOR)
are dropped, since their position depends on the color the fit solves for.
Gaia stars with an exactly-zero PMRA or PMDEC are dropped.
"""

import argparse
import glob
import multiprocessing as mp
import os
import sys

import fitsio
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.spatial import cKDTree

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from healpix_region_data import load_coadd
from plot_gpr_vs_gaia_per_exposure import _expnum, ref_keys, robust_std
from highpm.cat_reader import clean_cat
from highpm.gnomonic_converter import gnomonicJacobian, projectGnomonic
from highpm.pm_setup import read_pm_cat

GAIA_REF_MJD = 57388.5  # J2016.0
YEAR = 365.25
MAS = 3.6e6             # mas per degree

_G = {}


def gaia_near(hpix):
    """Gaia in this healpixel and its neighbors, minus exactly-zero PMs."""
    pixels = set(int(p) for p in hp.get_all_neighbours(32, hpix)) | {hpix}
    pixels.discard(-1)
    g = load_coadd(_G["gaia_dir"], pixels, pattern="GaiaSource_*.fits")
    g = g[np.isfinite(g["PMRA"]) & np.isfinite(g["PMDEC"]) & np.isfinite(g["PARALLAX"])]
    g = g[(g["PMRA"] != 0) & (g["PMDEC"] != 0)]
    if _G["max_ruwe"] is not None:
        g = g[g["RUWE"] <= _G["max_ruwe"]]
    return g


def gaia_cov(g):
    """(N, 5, 5) Gaia covariance in mas / mas/yr, order (ra*, dec, parallax, pmra, pmdec)."""
    names = ["RA", "DEC", "PARALLAX", "PMRA", "PMDEC"]
    err = np.column_stack([g[n + "_ERROR"] for n in names]).astype(np.float64)
    corr = np.broadcast_to(np.eye(5), (len(g), 5, 5)).copy()
    for i in range(5):
        for j in range(i + 1, 5):
            corr[:, i, j] = corr[:, j, i] = g[f"{names[i]}_{names[j]}_CORR"]
    return corr * err[:, :, None] * err[:, None, :]


def per_healpix(path):
    hpix = int(os.path.basename(path)[-10:-5])
    config = dict(_G["config"])
    hdr = fitsio.read_header(path, ext=1)
    config["ra0"], config["dec0"] = hdr["RA0"], hdr["DEC0"]
    ra0, dec0 = hdr["RA0"], hdr["DEC0"]

    cat = read_pm_cat(path)
    extra = fitsio.read(path, ext=1, columns=["CCDNUM", "OBJECT_NUMBER"])
    n_all = len(cat)
    keep = clean_cat(cat, config)
    n_clean = keep.sum()
    keep &= hp.ang2pix(32, cat["BEST_RA"], cat["BEST_DEC"], lonlat=True) == hpix
    n_inpix = keep.sum()
    keep &= (cat["DXI_DCOLOR"] == 0) & (cat["DETA_DCOLOR"] == 0)
    cat, extra = cat[keep], extra[keep]

    g = gaia_near(hpix)
    gxi, geta, *_ = projectGnomonic(g["RA"], g["DEC"], 0 * g["RA"], 0 * g["RA"], ra0, dec0)
    dxi_dra, dxi_ddec, deta_dra, deta_ddec = gnomonicJacobian(g["RA"], g["DEC"], ra0, dec0)
    cosd = np.cos(np.radians(g["DEC"]))
    # J maps local (East = cos(dec) dRA, North = dDec) offsets into (xi, eta).
    J = np.stack([np.stack([dxi_dra / cosd, dxi_ddec], -1), np.stack([deta_dra / cosd, deta_ddec], -1)], -2)

    dist, j = cKDTree(np.column_stack([gxi, geta])).query(
        np.column_stack([cat["XI"], cat["ETA"]]), distance_upper_bound=_G["cand_deg"])
    ok = np.isfinite(dist)
    cat, extra, j = cat[ok], extra[ok], j[ok]

    dt = (cat["MJD"] - GAIA_REF_MJD) / YEAR
    Jj = J[j]
    pm = np.einsum("nab,nb->na", Jj, np.column_stack([g["PMRA"][j], g["PMDEC"][j]]))
    par = np.column_stack([cat["PAR_XI"], cat["PAR_ETA"]])
    pred_xi = gxi[j] * MAS + pm[:, 0] * dt + g["PARALLAX"][j] * par[:, 0]
    pred_eta = geta[j] * MAS + pm[:, 1] * dt + g["PARALLAX"][j] * par[:, 1]
    dxi = cat["XI"] * MAS - pred_xi
    deta = cat["ETA"] * MAS - pred_eta
    close = np.hypot(dxi, deta) < _G["match_mas"]
    cat, extra, j, dt, Jj, par, dxi, deta = (a[close] for a in (cat, extra, j, dt, Jj, par, dxi, deta))

    # A (N, 2, 5): d(pred)/d(ra*, dec, parallax, pmra, pmdec)
    A = np.concatenate([Jj, par[:, :, None], Jj * dt[:, None, None]], axis=2)
    Cg = np.einsum("nai,nij,nbj->nab", A, gaia_cov(g)[j], A)

    ex, ey, rho = (1000 * cat["BEST_RA_ERR"], 1000 * cat["BEST_DEC_ERR"], cat["BEST_RA_DEC_CORR"])
    cxx = ex ** 2 + Cg[:, 0, 0]
    cyy = ey ** 2 + Cg[:, 1, 1]
    cxy = rho * ex * ey + Cg[:, 0, 1]
    chi2 = (cyy * dxi ** 2 - 2 * cxy * dxi * deta + cxx * deta ** 2) / (cxx * cyy - cxy ** 2)

    keys = (cat["EXPNUM"].astype(np.int64) * 10**10 + extra["CCDNUM"].astype(np.int64) * 10**8
            + extra["OBJECT_NUMBER"].astype(np.int64))
    isref = np.isin(keys, _G["allkeys"])
    print(f"hp{hpix:05d}: {n_all} dets, {n_clean} after clean_cat, {n_inpix} in pixel, "
          f"{len(cat)} known-color matched to Gaia ({isref.sum()} ref)", flush=True)
    return dict(gmag=g["PHOT_G_MEAN_MAG"][j].astype(np.float32),
                pull_xi=(dxi / np.sqrt(cxx)).astype(np.float32),
                pull_eta=(deta / np.sqrt(cyy)).astype(np.float32),
                chi2=chi2.astype(np.float32),
                gaia_err_frac=(np.sqrt(Cg[:, 0, 0] / cxx)).astype(np.float32),
                source_id=g["SOURCE_ID"][j], isref=isref,
                counts=np.array([n_all, n_clean, n_inpix, len(cat)]))


def binned(gmag, pull, source_id, edges, min_n=20):
    """Per-bin median and robust std with errors using the number of distinct
    Gaia stars (not detections) in the bin, since detections of one star
    share the same Gaia error and aren't independent."""
    nb = len(edges) - 1
    out = {k: np.full(nb, np.nan) for k in ("med", "med_err", "std", "std_err", "n", "nstar")}
    idx = np.digitize(gmag, edges) - 1
    for i in range(nb):
        sel = idx == i
        out["n"][i] = sel.sum()
        if sel.sum() < min_n:
            continue
        p = pull[sel]
        nstar = len(np.unique(source_id[sel]))
        s = robust_std(p)
        out["med"][i], out["std"][i], out["nstar"][i] = np.median(p), s, nstar
        out["med_err"][i] = 1.2533 * s / np.sqrt(nstar)
        out["std_err"][i] = 1.166 * s / np.sqrt(nstar)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", default="/data8/shared/decampm/PMSculptor_8deg_garyb/")
    parser.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "config.yaml"))
    parser.add_argument("--des-exposures", default="/home2/vwetzell/gitrepos/pixmappy/pixmappy/data/delveExposures.hdf5",
                         help="$DES_EXPOSURES as set for the PM jobs (exposure thinning in clean_cat)")
    parser.add_argument("--gpr-dirs", nargs="+", default=[f"/data8/shared/decampm/GPR2/CAT/{b}/" for b in "griz"])
    parser.add_argument("--gaia-dir", default="/data8/shared/decampm/Gaia/")
    parser.add_argument("--max-ruwe", type=float, default=None)
    parser.add_argument("--match-mas", type=float, default=300.0, help="Match radius after propagating Gaia to the epoch")
    parser.add_argument("--candidate-arcsec", type=float, default=2.0)
    parser.add_argument("--gmag-bins", type=float, nargs=3, default=(15, 21, 13),
                        help="linspace(start, stop, num) edges; detections brighter than start are dropped")
    parser.add_argument("--nproc", type=int, default=8)
    parser.add_argument("--keys-cache", default=None, help="Cache the GPR ref-star keys to this .npy")
    parser.add_argument("--cache", default=None, help="Cache the per-detection pulls to this .npz")
    parser.add_argument("-o", "--out", default="gaia_detection_pull_ref_vs_nonref.png")
    args = parser.parse_args()

    if args.cache and os.path.exists(args.cache):
        d = dict(np.load(args.cache))
    else:
        if args.keys_cache and os.path.exists(args.keys_cache):
            allkeys = np.load(args.keys_cache)
        else:
            files = sorted(glob.glob(os.path.join(args.run_dir, "PositionCorrectedExposureCatalog", "position_corrected_*.fits")))
            with mp.get_context("fork").Pool(24) as pool:
                allkeys = np.sort(np.concatenate(pool.starmap(ref_keys, [(_expnum(f), args.gpr_dirs) for f in files])))
            if args.keys_cache:
                np.save(args.keys_cache, allkeys)
        with open(args.config) as f:
            config = yaml.safe_load(f)
        # exclusion_regions_file is relative to the repo root, like PM.py's cwd.
        if config.get("exclusion_regions_file") and not os.path.isabs(config["exclusion_regions_file"]):
            config["exclusion_regions_file"] = os.path.join(os.path.dirname(os.path.abspath(args.config)), "..",
                                                            config["exclusion_regions_file"])
        os.environ.setdefault("DES_EXPOSURES", args.des_exposures)
        _G.update(config=config, allkeys=allkeys, gaia_dir=args.gaia_dir, max_ruwe=args.max_ruwe,
                  match_mas=args.match_mas, cand_deg=args.candidate_arcsec / 3600.0)

        pixels = np.load(os.path.join(args.run_dir, "pmsculptor8deg_healpix.npy"))
        paths = [os.path.join(args.run_dir, "HealpixDetectionCatalog", f"cleaned_detections_hp{int(h):05d}.fits") for h in pixels]
        with mp.get_context("fork").Pool(args.nproc) as pool:
            parts = pool.map(per_healpix, paths, chunksize=1)
        d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0] if k != "counts"}
        d["counts"] = np.sum([p["counts"] for p in parts], axis=0)
        if args.cache:
            np.savez(args.cache, **d)

    n_all, n_clean, n_inpix, n_match = d["counts"]
    bright = d["gmag"] < args.gmag_bins[0]
    print(f"[gmag] dropping {bright.sum()} detections with G < {args.gmag_bins[0]}")
    d = {k: (v[~bright] if k != "counts" else v) for k, v in d.items()}
    isref = d["isref"]
    print(f"[detections] {n_all} total, {n_clean} after clean_cat, {n_inpix} in own pixel, "
          f"{n_match} known-color matched to Gaia: {isref.sum()} ref, {(~isref).sum()} non-ref")
    for name, sel in (("ref", isref), ("non-ref", ~isref)):
        g16 = sel & (d["gmag"] >= 16)
        print(f"[{name}] G>=16: median Gaia share of total xi error = {np.median(d['gaia_err_frac'][g16]):.3f}, "
              f"median chi2(2 dof) = {np.median(d['chi2'][g16]):.3f} (ideal 1.386)")

    edges = np.linspace(*args.gmag_bins[:2], int(args.gmag_bins[2]))
    centers = 0.5 * (edges[:-1] + edges[1:])
    groups = ((isref, "firebrick", "o", -0.06, "GPR reference detections"),
              (~isref, "steelblue", "s", 0.06, "non-reference detections"))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for col, comp in enumerate(("xi", "eta")):
        for sel, color, marker, dx, name in groups:
            b = binned(d["gmag"][sel], d[f"pull_{comp}"][sel], d["source_id"][sel], edges)
            label = f"{name} (n={sel.sum():,}, {len(np.unique(d['source_id'][sel])):,} stars)"
            axes[0, col].errorbar(centers + dx, b["med"], b["med_err"], fmt=marker + "-", color=color, ms=5, capsize=2, label=label)
            axes[1, col].errorbar(centers + dx, b["std"], b["std_err"], fmt=marker + "-", color=color, ms=5, capsize=2, label=label)
            for i in range(len(centers)):
                print(f"{comp:3s} {name:25s} G={centers[i]:5.2f} n={int(b['n'][i]):8d} stars={b['nstar'][i]:7.0f} "
                      f"median={b['med'][i]:+.3f}+-{b['med_err'][i]:.3f} robust_std={b['std'][i]:.3f}+-{b['std_err'][i]:.3f}")
        axes[0, col].axhline(0, color="k", ls="--", lw=1)
        axes[1, col].axhline(1, color="k", ls="--", lw=1)
        axes[0, col].set_title(rf"$\{comp}$ (healpix frame, $\approx$ {'RA' if comp == 'xi' else 'Dec'})")
        axes[0, col].set_ylabel(rf"$\{comp}$ pull median")
        axes[1, col].set_ylabel(rf"$\{comp}$ pull robust std")
        axes[1, col].set_xlabel("Gaia G magnitude")
        for ax in axes[:, col]:
            ax.axvspan(edges[0], 16, color="0.85", zorder=0)
            ax.grid(alpha=0.4)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(r"Sculptor 8 deg: per-detection position pull vs Gaia at the exposure epoch, "
                 r"$(x_{det}-x_{gaia}(t))/\sqrt{\sigma_{det}^2+\sigma_{gaia}^2(t)}$"
                 "\nclean_cat applied; Gaia 5-param covariance propagated; Gaia PMRA/PMDEC == 0 removed; "
                 "errorbars use N distinct stars; shaded G<16", fontsize=10)
    fig.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Saved {args.out}")
