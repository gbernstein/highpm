"""
Per-exposure photometric depth from each exposure's own single-epoch skim.

Detection is an S/N threshold, so an exposure's completeness is set by two numbers
its catalog already carries:
  zp       -- magnitude of unit FLUX_PSF, from matches to DELVE catalog point sources
              (COADD, WAVG_MAG_PSF), which cover every processed healpixel. On
              exposures that also have gold stars it agrees with the gold zp to
              0.002 mag (nmad; offset 2-4 mmag, the same inside and outside Y6).
              Gold (data/gold_bright) now only sets y6_frac: the fraction of the
              zp stars that are Y6 Gold rather than DELVE DR3 gold (0 outside gold).
  sigma    -- background noise of FLUX_PSF: median FLUXERR_PSF of faint (S/N < 8)
              clean detections, per CCD. Source photon noise is negligible there.
m1 = zp - 2.5 log10(sigma) is the magnitude at S/N = 1; the S/N threshold nu that
turns it into m50 = m1 - 2.5 log10(nu) is calibrated separately per band and
processing (scripts/calibrate_snr_threshold.py). Exposure time, read noise, sky,
seeing and transparency all enter through zp and sigma, with no T_EFF.

Writes data/delve.exposures.depth.fits, one row per exposure with a skim.
"""

import os
from multiprocessing import Pool

import fitsio
import h5py
import healpy as hp
import numpy as np
from scipy.spatial import cKDTree

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SKIM = "/data8/shared/decampm/{B}/D{e:08d}_{b}_cat.fits"
EXPOSURES = os.path.expanduser("~/gitrepos/pixmappy/pixmappy/data/delveExposures.hdf5")
BRIGHT = os.path.join(REPO, "data", "gold_bright", "bright_{:05d}.fits")
COADD = "/data8/shared/decampm/COADDS/cat_hpx_{:05d}.fits"
OUT = os.path.join(REPO, "data", "delve.exposures.depth.fits")

NSIDE = 32
NCCD = 63  # index by CCDNUM (1-62)
MATCH_ARCSEC = 1.0
ZP_MAG = (17.0, 21.0)  # reference mags used for zp (unsaturated, high S/N)
ZP_MIN_SNR = 20.0
FAINT_MAX_SNR = 8.0  # detections used for the background noise level
MIN_FAINT_PER_CCD = 10
MIN_ZP_STARS = 10

SKIM_COLS = ["RA", "DEC", "CCDNUM", "FLUX_PSF", "FLUXERR_PSF", "FLAGS", "IMAFLAGS_ISO",
             "FWHM_WORLD", "TAG"]

DTYPE = [("expnum", "i4"), ("band", "U1"), ("exptime", "f4"), ("tag", "U8"),
         ("zp", "f8"), ("zp_mad", "f4"), ("n_zp", "i4"), ("y6_frac", "f4"),
         ("m1", "f8"), ("m1_ccd_std", "f4"), ("m1_ccd", "f4", (NCCD,)),
         ("fwhm", "f4"), ("n_ccd", "i2")]


def read_skim(expnum, band):
    s = fitsio.read(SKIM.format(B=band.upper(), e=expnum, b=band), columns=SKIM_COLS)
    clean = ((s["FLAGS"] < 4) & (s["IMAFLAGS_ISO"] == 0)
             & (s["FLUX_PSF"] > 0) & (s["FLUXERR_PSF"] > 0))
    return s, clean


def load_bright(ra, dec, band):
    pix = np.unique(hp.ang2pix(NSIDE, ra, dec, lonlat=True))
    parts = []
    for p in pix:
        try:
            parts.append(fitsio.read(BRIGHT.format(p)))
        except (OSError, IOError):  # outside gold coverage, or a pixel with no bright stars
            continue
    if not parts:
        return None
    ref = np.concatenate(parts)
    return ref[(ref[band] > ZP_MAG[0]) & (ref[band] < ZP_MAG[1])]


def load_reference(ra, dec, band):
    """DELVE catalog point sources (ra, dec, mag) in ZP_MAG around the positions, or None."""
    B = band.upper()
    cols = ["RA", "DEC", f"WAVG_MAG_PSF_{B}", f"EXTENDED_CLASS_{B}", f"WAVG_FLAGS_{B}"]
    parts = [fitsio.read(COADD.format(p), columns=cols)
             for p in np.unique(hp.ang2pix(NSIDE, ra, dec, lonlat=True)) if os.path.exists(COADD.format(p))]
    if not parts:
        return None
    c = np.concatenate(parts)
    mag = c[f"WAVG_MAG_PSF_{B}"]
    keep = ((c[f"EXTENDED_CLASS_{B}"] >= 0) & (c[f"EXTENDED_CLASS_{B}"] <= 1) & (c[f"WAVG_FLAGS_{B}"] < 4)
            & (mag > ZP_MAG[0]) & (mag < ZP_MAG[1]))
    return c["RA"][keep], c["DEC"][keep], mag[keep].astype(float)


def match(ra1, dec1, ra2, dec2, radius_arcsec=MATCH_ARCSEC):
    """Nearest (ra2, dec2) index for each (ra1, dec1) within radius, else -1. Flat sky."""
    cosd = np.cos(np.radians(np.median(dec2)))
    d, i = cKDTree(np.c_[ra2 * cosd, dec2]).query(
        np.c_[ra1 * cosd, dec1], distance_upper_bound=radius_arcsec / 3600)
    return np.where(np.isfinite(d), i, -1)


def zeropoint(s, clean, band):
    """(zp, zp_mad, n, y6_frac) from DELVE catalog stars; zp is nan if too few match."""
    snr = s["FLUX_PSF"] / np.where(s["FLUXERR_PSF"] > 0, s["FLUXERR_PSF"], np.inf)
    use = clean & (snr > ZP_MIN_SNR)
    if use.sum() == 0:
        return np.nan, np.nan, 0, np.nan
    ra, dec, flux = s["RA"][use], s["DEC"][use], s["FLUX_PSF"][use]
    ref = load_reference(ra, dec, band)
    if ref is None or len(ref[0]) == 0:
        return np.nan, np.nan, 0, np.nan
    idx = match(ref[0], ref[1], ra, dec)
    ok = idx >= 0
    if ok.sum() < MIN_ZP_STARS:
        return np.nan, np.nan, int(ok.sum()), np.nan
    zps = ref[2][ok] + 2.5 * np.log10(flux[idx[ok]])
    zp = np.median(zps)
    return zp, 1.4826 * np.median(np.abs(zps - zp)), int(ok.sum()), y6_fraction(ref[0][ok], ref[1][ok], band)


def y6_fraction(ra, dec, band):
    """Fraction of the zp stars that are Y6 Gold, among those in gold (0 if none are)."""
    gold = load_bright(ra, dec, band)
    if gold is None or len(gold) == 0:
        return 0.0
    idx = match(ra, dec, gold["ra"], gold["dec"])
    ok = idx >= 0
    return float(gold["y6"][idx[ok]].mean()) if ok.any() else 0.0


def noise_per_ccd(s, clean):
    """Median background FLUXERR_PSF per CCDNUM (nan where too few faint detections)."""
    faint = clean & (s["FLUX_PSF"] / s["FLUXERR_PSF"] < FAINT_MAX_SNR)
    sigma = np.full(NCCD, np.nan)
    for n in np.unique(s["CCDNUM"][faint]):
        sel = faint & (s["CCDNUM"] == n)
        if sel.sum() >= MIN_FAINT_PER_CCD and 0 < n < NCCD:
            sigma[n] = np.median(s["FLUXERR_PSF"][sel])
    return sigma


def measure(args):
    expnum, band, exptime = args
    row = np.zeros(1, DTYPE)[0]
    row["expnum"], row["band"], row["exptime"] = expnum, band, exptime
    row["m1_ccd"] = np.nan
    try:
        s, clean = read_skim(expnum, band)
    except (OSError, IOError):
        return None
    if len(s) == 0:
        return None
    row["tag"] = s["TAG"][0].strip() or "DES"
    row["fwhm"] = np.median(s["FWHM_WORLD"][clean]) * 3600 if clean.any() else np.nan
    row["zp"], row["zp_mad"], row["n_zp"], row["y6_frac"] = zeropoint(s, clean, band)
    m1_ccd = row["zp"] - 2.5 * np.log10(noise_per_ccd(s, clean))
    row["m1_ccd"] = m1_ccd
    good = np.isfinite(m1_ccd)
    row["n_ccd"] = good.sum()
    row["m1"] = np.median(m1_ccd[good]) if good.any() else np.nan
    row["m1_ccd_std"] = np.std(m1_ccd[good]) if good.sum() > 1 else np.nan
    return row


def exposure_list():
    with h5py.File(EXPOSURES, "r") as f:
        t = f["__astropy_table__"]
        expnum = t["expnum"][:].astype(int)
        band = np.char.decode(t["band"][:]).astype(str)
        exptime = t["exptime"][:].astype(float)
    keep = np.isin(band, list("griz"))
    return list(zip(expnum[keep].tolist(), band[keep].tolist(), exptime[keep].tolist()))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--expnums", type=int, nargs="*", help="only these exposures (for testing)")
    ap.add_argument("-o", "--out", default=OUT)
    ap.add_argument("-j", "--jobs", type=int, default=32)
    args = ap.parse_args()

    todo = exposure_list()
    if args.expnums:
        todo = [t for t in todo if t[0] in set(args.expnums)]
    rows = []
    with Pool(args.jobs) as pool:
        for i, r in enumerate(pool.imap_unordered(measure, todo, chunksize=16)):
            if r is not None:
                rows.append(r)
            if (i + 1) % 10000 == 0:
                print(f"  {i + 1}/{len(todo)}", flush=True)
    out = np.array(rows, dtype=DTYPE)
    out = out[np.argsort(out["expnum"])]
    fitsio.write(args.out, out, clobber=True)
    ok = np.isfinite(out["m1"])
    print(f"wrote {args.out}: {len(out)} of {len(todo)} exposures have skims, "
          f"{ok.sum()} with a depth ({np.isnan(out['zp']).sum()} lack a zeropoint)")
    for tag in np.unique(out["tag"]):
        print(f"  tag {str(tag)}: {np.sum(out['tag'] == tag)}")


if __name__ == "__main__":
    main()
