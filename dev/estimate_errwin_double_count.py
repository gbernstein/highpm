"""How much of the ERRWIN (SExtractor windowed centroid) variance would have to
already be inside the GPR cov_model -- i.e. double counted when
detection_packaging.rotate_covariances_to_healpix_frame adds the two -- to
explain the Gaia pull std < 1?

Per detection the fit uses C = E + G (E: ERRWIN ellipse, G: GPR cov_model,
both rotated into the healpix (xi, eta) frame). If a fraction f of E is
already in G, the right covariance is C_f = C - f*E. For every Gaia-matched
modest1 mover, refit its final member detections with C_f for f on a grid
(same 5-parameter model + parallax prior as pmfit.singleFit), then form the
Gaia pull with the refit PM and PM error. The refit is applied as a change
relative to f=0, so fit details not reproduced here (colour term) cancel:
    mu_f    = mu_cat + (mu_refit(f) - mu_refit(0))
    sigma_f = sigma_cat * sigma_refit(f) / sigma_refit(0)
Per G bin, f* is where the robust pull std crosses 1.
"""

import argparse
import glob
import os
import sys
from multiprocessing import Pool

import fitsio
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import yaml
from astropy import units as u
from astropy.coordinates import SkyCoord

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from highpm.gnomonic_converter import gnomonicJacobian  # noqa: E402

DEGREE = 3600.0
DAY = 1.0 / 365.2425
F_GRID = np.linspace(0.0, 1.0, 21)


def robust_std(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def errwin_healpix_cov(det, errwin, ra0, dec0):
    """ERRWIN ellipse -> healpix (xi, eta) covariance in arcsec^2, exactly as
    rotate_covariances_to_healpix_frame does it."""
    dxi_dra, dxi_ddec, deta_dra, deta_ddec = gnomonicJacobian(det["BEST_RA"], det["BEST_DEC"], ra0, dec0)
    a = errwin["ERRAWIN_WORLD"] * DEGREE
    b = errwin["ERRBWIN_WORLD"] * DEGREE
    pa = np.radians(errwin["ERRTHETAWIN_J2000"])
    ee = a * a - b * b
    exx = (a * a + b * b + ee * np.cos(pa)) / 2.0
    eyy = (a * a + b * b - ee * np.cos(pa)) / 2.0
    exy = ee * np.sin(pa) / 2.0
    cosdec = np.cos(np.radians(det["BEST_DEC"]))
    j00, j01 = dxi_dra / cosdec, dxi_ddec
    j10, j11 = deta_dra / cosdec, deta_ddec
    return np.array([
        j00 * j00 * exx + 2 * j00 * j01 * exy + j01 * j01 * eyy,
        j10 * j10 * exx + 2 * j10 * j11 * exy + j11 * j11 * eyy,
        j00 * j10 * exx + (j00 * j11 + j01 * j10) * exy + j01 * j11 * eyy,
    ]).T


def refit(xy, t, par, C, E, parallax_prior):
    """5-parameter GLS for every f in F_GRID at once. Returns pm (F, 2) in
    arcsec/yr and pm sigma (F, 2)."""
    n = len(t)
    one, zero = np.ones(n), np.zeros(n)
    A = np.swapaxes(np.array([[one, zero, t, zero, par[:, 0]], [zero, one, zero, t, par[:, 1]]]), 1, 2)  # (2, N, 5)
    Cf = C[None, :, :] - F_GRID[:, None, None] * E[None, :, :]  # (F, N, 3)
    det = Cf[..., 0] * Cf[..., 1] - Cf[..., 2] ** 2
    invC = np.array([[Cf[..., 1], -Cf[..., 2]], [-Cf[..., 2], Cf[..., 0]]]) / det  # (2, 2, F, N)
    xy0 = xy - xy.mean(axis=0)
    alpha = np.einsum("ikm,ijfk,jkn->fmn", A, invC, A)
    beta = np.einsum("ikm,ijfk,kj->fm", A, invC, xy0)
    alpha[:, 4, 4] += parallax_prior ** -2
    p = np.linalg.solve(alpha, beta[..., None])[..., 0]
    cov = np.linalg.inv(alpha)
    return p[:, 2:4], np.sqrt(cov[:, [2, 3], [2, 3]])


def process_healpix(args_tuple):
    hp_idx, a = args_tuple
    pm_file = os.path.join(a.base, "PMCatalog", f"real_PM_hp{hp_idx:05d}.fits")
    det_file = os.path.join(a.base, "HealpixDetectionCatalog", f"cleaned_detections_hp{hp_idx:05d}.fits")
    gaia_files = glob.glob(os.path.join(a.gaia_dir, f"GaiaSource_*{hp_idx:05d}.fits")) or \
        glob.glob(os.path.join(a.gaia_dir, f"GaiaSource_*{hp_idx}.fits"))
    if not (os.path.exists(pm_file) and os.path.exists(det_file) and gaia_files):
        return None

    movers = fitsio.read(pm_file, ext="modest1_movers")
    dets_tbl = fitsio.read(pm_file, ext="modest1_detections")
    in_pix = hp.ang2pix(a.nside, movers["ra"], movers["dec"], lonlat=True) == hp_idx
    mover_rows = np.flatnonzero(in_pix)
    movers = movers[in_pix]

    gaia = fitsio.read(gaia_files[0])
    gaia = gaia[np.isfinite(gaia["PMRA"]) & (gaia["PMRA"] != 0) & (gaia["PMDEC"] != 0)]
    idx, d2d, _ = SkyCoord(movers["ra"] * u.deg, movers["dec"] * u.deg).match_to_catalog_sky(
        SkyCoord(gaia["RA"] * u.deg, gaia["DEC"] * u.deg))
    ok = (d2d.arcsec < a.match_arcsec) & (gaia["RUWE"][idx] <= a.max_ruwe)
    mover_rows, movers, gaia = mover_rows[ok], movers[ok], gaia[idx[ok]]
    if not len(movers):
        return None

    # mover idx in the detections table is the row index within the movers extension
    members = dets_tbl[~dets_tbl["clipped"]]
    order = np.argsort(members["idx"], kind="stable")
    members = members[order]
    starts = np.searchsorted(members["idx"], movers["idx"], side="left")
    ends = np.searchsorted(members["idx"], movers["idx"], side="right")
    need = np.unique(np.concatenate([members["detections"][s:e] for s, e in zip(starts, ends)]))

    hdr = fitsio.read_header(det_file, ext=1)
    ra0, dec0 = hdr["RA0"], hdr["DEC0"]
    cols = ["XI", "ETA", "MJD", "PAR_XI", "PAR_ETA", "BEST_RA", "BEST_DEC", "BEST_RA_ERR", "BEST_DEC_ERR",
            "BEST_RA_DEC_CORR", "EXPNUM", "CCDNUM", "OBJECT_NUMBER"]
    det = fitsio.read(det_file, ext=1, rows=need, columns=cols)

    # ERRWIN isn't carried into the packed catalog; get it back from the
    # per-exposure position-corrected files by (EXPNUM, CCDNUM, OBJECT_NUMBER).
    errwin = np.full(len(det), np.nan, dtype=[("ERRAWIN_WORLD", "f8"), ("ERRBWIN_WORLD", "f8"),
                                               ("ERRTHETAWIN_J2000", "f8")])
    for expnum in np.unique(det["EXPNUM"]):
        sel = np.flatnonzero(det["EXPNUM"] == expnum)
        pc = fitsio.read(os.path.join(a.base, "PositionCorrectedExposureCatalog",
                                      f"position_corrected_{expnum:08d}.fits"),
                         columns=["CCDNUM", "OBJECT_NUMBER", "ERRAWIN_WORLD", "ERRBWIN_WORLD", "ERRTHETAWIN_J2000"])
        key_pc = pc["CCDNUM"].astype(np.int64) * 10_000_000 + pc["OBJECT_NUMBER"]
        key_det = det["CCDNUM"][sel].astype(np.int64) * 10_000_000 + det["OBJECT_NUMBER"][sel]
        o = np.argsort(key_pc)
        pos = np.clip(np.searchsorted(key_pc[o], key_det), 0, len(o) - 1)
        hit = key_pc[o][pos] == key_det
        for c in errwin.dtype.names:
            errwin[c][sel[hit]] = pc[c][o][pos[hit]]

    C = np.array([det["BEST_RA_ERR"] ** 2, det["BEST_DEC_ERR"] ** 2,
                  det["BEST_RA_DEC_CORR"] * det["BEST_RA_ERR"] * det["BEST_DEC_ERR"]]).T
    E = errwin_healpix_cov(det, errwin, ra0, dec0)
    row_of = {r: i for i, r in enumerate(need)}

    out = []
    for m, s, e, g in zip(movers, starts, ends, gaia):
        rows = np.array([row_of[r] for r in members["detections"][s:e]])
        if len(rows) < 5 or not np.all(np.isfinite(E[rows])):
            continue
        xy = np.array([det["XI"][rows], det["ETA"][rows]]).T * DEGREE
        t = (det["MJD"][rows] - a.mjd_ref) * DAY
        par = np.array([det["PAR_XI"][rows], det["PAR_ETA"][rows]]).T
        try:
            pm, sig = refit(xy, t, par, C[rows], E[rows], a.parallax_prior)
        except np.linalg.LinAlgError:
            continue
        cat_pm = np.array([m["pmra"], m["pmdec"]])  # mas/yr
        cat_sig = 1000 * np.sqrt([m["c_vxvx"], m["c_vyvy"]])
        pm_f = cat_pm[None, :] + 1000 * (pm - pm[0])
        sig_f = cat_sig[None, :] * sig / sig[0]
        efrac = np.mean((E[rows, 0] + E[rows, 1]) / (C[rows, 0] + C[rows, 1]))
        out.append((g["PHOT_G_MEAN_MAG"], g["PMRA"], g["PMDEC"], g["PMRA_ERROR"], g["PMDEC_ERROR"],
                    efrac, pm_f, sig_f, 1000 * sig[0], cat_sig))
    print(f"hp {hp_idx}: {len(out)} stars", flush=True)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="/data8/shared/decampm/PMSculptor_8deg_garyb")
    parser.add_argument("--gaia-dir", default="/data8/shared/decampm/Gaia/")
    parser.add_argument("--config", default=os.path.join(REPO, "config", "config.yaml"))
    parser.add_argument("--healpix", type=int, nargs="*", default=None, help="default: all in the run")
    parser.add_argument("--nside", type=int, default=32)
    parser.add_argument("--match-arcsec", type=float, default=0.3)
    parser.add_argument("--max-ruwe", type=float, default=1.4)
    parser.add_argument("--gmin", type=float, default=16.0, help="our errors aren't reliable brighter than this")
    parser.add_argument("--cores", type=int, default=8)
    parser.add_argument("--cache", default="errwin_double_count.npz")
    parser.add_argument("-o", "--out", default="errwin_double_count.png")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    args.mjd_ref = cfg["mjd_ref"]
    args.parallax_prior = cfg["fitting"]["parallax_prior"]

    if os.path.exists(args.cache):
        d = dict(np.load(args.cache))
        print(f"loaded {args.cache}")
    else:
        pix = args.healpix or [int(p) for p in np.load(os.path.join(args.base, "pmsculptor8deg_healpix.npy"))]
        with Pool(args.cores) as pool:
            results = [r for rr in pool.map(process_healpix, [(p, args) for p in pix]) if rr for r in rr]
        cols = list(zip(*results))
        d = dict(gmag=np.array(cols[0]), gpmra=np.array(cols[1]), gpmdec=np.array(cols[2]),
                 gsra=np.array(cols[3]), gsdec=np.array(cols[4]), efrac=np.array(cols[5]),
                 pm=np.array(cols[6]), sig=np.array(cols[7]), sig0=np.array(cols[8]), catsig=np.array(cols[9]))
        np.savez(args.cache, **d)

    n = len(d["gmag"])
    print(f"{n} stars; refit f=0 sigma / catalog sigma: median {np.median(d['sig0'] / d['catsig'], axis=0)}")

    edges = np.arange(args.gmin, 21.01, 0.5)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ib = np.digitize(d["gmag"], edges) - 1
    gaia_pm = np.array([d["gpmra"], d["gpmdec"]]).T
    gaia_sig = np.array([d["gsra"], d["gsdec"]]).T

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    cmap = plt.get_cmap("viridis")
    fstar = np.full((len(centers), 2), np.nan)
    print("\n  G      n    <E/C>   s(f=0) pmra/pmdec   s(f=1) pmra/pmdec   f* pmra/pmdec")
    for i, gc in enumerate(centers):
        sel = ib == i
        if sel.sum() < 100:
            continue
        s = np.empty((len(F_GRID), 2))
        for c in range(2):
            pull = (d["pm"][sel, :, c] - gaia_pm[sel, c][:, None]) / np.hypot(d["sig"][sel, :, c],
                                                                               gaia_sig[sel, c][:, None])
            s[:, c] = [robust_std(pull[:, k]) for k in range(len(F_GRID))]
            if s[0, c] < 1 <= s[-1, c]:
                fstar[i, c] = np.interp(1.0, s[:, c], F_GRID)
        print(f"{gc:5.2f} {sel.sum():7d}  {np.mean(d['efrac'][sel]):.3f}    {s[0, 0]:.3f} / {s[0, 1]:.3f}"
              f"       {s[-1, 0]:.3f} / {s[-1, 1]:.3f}       "
              + " / ".join("  -  " if np.isnan(x) else f"{x:.2f}" for x in fstar[i]))
        col = cmap((gc - args.gmin) / (21 - args.gmin))
        axes[0].plot(F_GRID, s[:, 0], color=col, label=f"G={gc:.2f}")
        axes[1].plot(F_GRID, s[:, 1], color=col)

    for ax, comp in zip(axes[:2], ("pmra", "pmdec")):
        ax.axhline(1, color="k", ls="--")
        ax.set_xlabel("f = fraction of ERRWIN variance already in cov_model")
        ax.set_ylabel(f"{comp} robust pull std")
        ax.set_title(comp)
        ax.grid()
    axes[0].legend(fontsize=7, ncol=2)

    ax = axes[2]
    ax.plot(centers, fstar[:, 0], "o-", label="f* pmra")
    ax.plot(centers, fstar[:, 1], "s-", label="f* pmdec")
    efrac_bin = [np.mean(d["efrac"][ib == i]) if np.any(ib == i) else np.nan for i in range(len(centers))]
    ax.plot(centers, efrac_bin, "k--", label=r"$\langle$ERRWIN share of detection variance$\rangle$")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Gaia G magnitude")
    ax.set_ylabel("fraction")
    ax.set_title("f needed for pull std = 1")
    ax.legend(fontsize=9)
    ax.grid()
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Saved {args.out}")
