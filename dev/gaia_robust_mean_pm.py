"""Robust mean Gaia proper motion per healpixel listed in a healpix .npy file."""
import argparse
import glob
from multiprocessing import Pool

import healpy as hp
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

NSIDE = 32


def _read(args):
    path, hpix = args
    d = fits.getdata(path)
    ok = np.isfinite(d["PMRA"]) & np.isfinite(d["PMDEC"]) & (d["RUWE"] < 1.4)
    d = d[ok]
    pix = hp.ang2pix(NSIDE, d["RA"], d["DEC"], lonlat=True)
    m = np.isin(pix, hpix)
    return pix[m], d["PMRA"][m], d["PMDEC"][m]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--healpix", default="/data8/shared/decampm/PMSculptor_8deg/pmsculptor8deg_healpix.npy")
    ap.add_argument("--gaia-dir", default="/data8/shared/decampm/Gaia")
    ap.add_argument("--output", default="dev/gaia_robust_mean_pm.csv")
    ap.add_argument("--nproc", type=int, default=16)
    a = ap.parse_args()
    hpix = np.load(a.healpix)
    files = sorted(glob.glob(f"{a.gaia_dir}/GaiaSource_*.fits"))
    with Pool(a.nproc) as p:
        res = p.map(_read, [(f, hpix) for f in files], chunksize=8)
    pix, pmra, pmdec = (np.concatenate(x) for x in zip(*res))
    with open(a.output, "w") as f:
        f.write("healpix,n,pmra_mean,pmra_err,pmdec_mean,pmdec_err\n")
        for h in hpix:
            m = pix == h
            n = m.sum()
            if n < 5:
                f.write(f"{h},{n},nan,nan,nan,nan\n")
                continue
            ra, _, _ = sigma_clipped_stats(pmra[m], sigma=3, maxiters=10)
            de, _, _ = sigma_clipped_stats(pmdec[m], sigma=3, maxiters=10)
            # error on the mean from the clipped scatter
            _, _, sra = sigma_clipped_stats(pmra[m], sigma=3, maxiters=10)
            _, _, sde = sigma_clipped_stats(pmdec[m], sigma=3, maxiters=10)
            f.write(f"{h},{n},{ra:.4f},{sra/np.sqrt(n):.4f},{de:.4f},{sde/np.sqrt(n):.4f}\n")
    ra, _, sra = sigma_clipped_stats(pmra, sigma=3, maxiters=10)
    de, _, sde = sigma_clipped_stats(pmdec, sigma=3, maxiters=10)
    n = len(pmra)
    print(f"combined N={n} pmra={ra:.4f}+-{sra/np.sqrt(n):.4f} pmdec={de:.4f}+-{sde/np.sqrt(n):.4f}")


if __name__ == "__main__":
    main()
