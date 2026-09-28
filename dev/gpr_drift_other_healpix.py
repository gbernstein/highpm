"""GPR - Gaia per-exposure drift for arbitrary complete healpixels (ref stars and all), read straight from GPR2/CAT.

MJD and parallax factors come from delveExposures.hdf5 (same recipe as position_correction.sky2bestSky).
"""
import argparse, os, sys
import multiprocessing as mp
import fitsio, h5py, numpy as np
from astropy.table import Table
from scipy.spatial import cKDTree

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from healpix_region_data import load_coadd
from exposuresFromHealpixList import _radec_from_table
from highpm.detection_packaging import get_exposures_near_healpix
from highpm.position_correction import getGPRFile
from plot_gpr_vs_gaia_per_exposure import fit_line, robust_std, GAIA_REF_MJD, YEAR, MAS

GPR = [f"/data8/shared/decampm/GPR2/CAT/{b}/" for b in "griz"]
_G = {}


def med(d, e):
    if len(d) < 20:
        return np.nan, np.nan
    for _ in range(2):
        k = (np.abs(d - np.median(d)) < 4 * robust_std(d)) & (np.abs(e - np.median(e)) < 4 * robust_std(e))
        d, e = d[k], e[k]
    return np.median(d), 1.2533 * robust_std(d) / np.sqrt(len(d))


def one(expnum):
    f = getGPRFile(expnum, GPR)
    i = np.where(_G["en"] == expnum)[0]
    if f is None or len(i) == 0:
        return None
    ra0, dec0 = _G["ra0"], _G["dec0"]
    cra, sra, cdec, sdec = np.cos(np.radians(ra0)), np.sin(np.radians(ra0)), np.cos(np.radians(dec0)), np.sin(np.radians(dec0))
    R = np.array([[-sra, cra, 0], [-cra * sdec, -sra * sdec, cdec], [cra * cdec, sra * cdec, sdec]])
    par = -(R @ _G["obs"][i[0]])
    mjd = _G["mjd"][i[0]]
    g = fitsio.read(f, ext=1, columns=["ref", "new_rd", "cov_model"])
    g = g[(g["cov_model"][:, 0, 0] > 0) & (g["cov_model"][:, 1, 1] > 0)]
    ra, dec = g["new_rd"][:, 0], g["new_rd"][:, 1]
    d, j = _G["tree"].query(np.column_stack([ra * _G["cos0"], dec]), distance_upper_bound=2 / 3600.0)
    ok = np.isfinite(d)
    if ok.sum() < 20:
        return None
    ra, dec, j, ref = ra[ok], dec[ok], j[ok], g["ref"][ok].astype(bool)
    ga = _G["gaia"]
    dt = (mjd - GAIA_REF_MJD) / YEAR
    cd = np.cos(np.radians(ga["DEC"][j]))
    dra = (ra - (ga["RA"][j] + (ga["PMRA"][j] * dt + ga["PARALLAX"][j] * par[0]) / MAS / cd)) * cd * MAS
    dde = (dec - (ga["DEC"][j] + (ga["PMDEC"][j] * dt + ga["PARALLAX"][j] * par[1]) / MAS)) * MAS
    c = np.hypot(dra, dde) < 300
    out = [mjd]
    for s in (c & ref, c):
        out += [s.sum(), *med(dra[s], dde[s]), *med(dde[s], dra[s])]
    return tuple(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("healpix", type=int, nargs="+")
    ap.add_argument("--nproc", type=int, default=24)
    a = ap.parse_args()
    t = Table.read("/home2/vwetzell/gitrepos/pixmappy/pixmappy/data/delveExposures.hdf5", path="__astropy_table__")
    en, ra, dec = _radec_from_table(t)
    _G.update(en=np.asarray(t["expnum"]), mjd=np.asarray(t["mjdmid"]), obs=np.asarray(t["obsicrs"]))
    print(f"{'hp':>6} {'set':>4} {'comp':>4} {'n_exp':>5} {'slope mas/yr':>16} {'offset J2016':>14}")
    for h in a.healpix:
        need = [int(e) for e in get_exposures_near_healpix(h, ra, dec, en, nside=32)]
        import healpy as hp
        r0, d0 = hp.pix2ang(32, h, lonlat=True)
        gaia = load_coadd("/data8/shared/decampm/Gaia/", {h}, pattern="GaiaSource_*.fits")
        gaia = gaia[np.isfinite(gaia["PMRA"]) & ~((gaia["PMRA"] == 0) & (gaia["PMDEC"] == 0) & (gaia["PARALLAX"] == 0))]
        gaia = gaia[(gaia["RUWE"] <= 1.4) & (gaia["PHOT_G_MEAN_MAG"] >= 15) & (gaia["PHOT_G_MEAN_MAG"] <= 19)]
        cos0 = np.cos(np.radians(d0))
        _G.update(gaia=gaia, cos0=cos0, ra0=r0, dec0=d0, tree=cKDTree(np.column_stack([gaia["RA"] * cos0, gaia["DEC"]])))
        with mp.get_context("fork").Pool(a.nproc) as p:
            rows = [r for r in p.map(one, need, chunksize=4) if r is not None]
        R = np.array(rows)
        np.save(f"drift_hp{h}.npy", R)
        ty = R[:, 0]
        for k, name in enumerate(("ref", "all")):
            b = 1 + 5 * k
            for comp, off in (("RA", 2), ("Dec", 4)):
                y, e = R[:, b + off - 1 + (0 if comp == "RA" else 0)], None
            n = R[:, b]
            for comp, iy in (("RA", b + 1), ("Dec", b + 3)):
                y, e = R[:, iy], np.maximum(R[:, iy + 1], 1e-3)
                g = np.isfinite(y) & np.isfinite(e)
                if g.sum() < 10:
                    print(f"{h:6d} {name:>4} {comp:>4} {g.sum():5d} insufficient"); continue
                x = 2016 + (ty[g] - GAIA_REF_MJD) / YEAR - 2016
                pp, pe, _ = fit_line(x, y[g], e[g])
                print(f"{h:6d} {name:>4} {comp:>4} {g.sum():5d} {pp[1]:+7.3f}+/-{pe[1]:.3f} {pp[0]:+8.3f}", flush=True)
