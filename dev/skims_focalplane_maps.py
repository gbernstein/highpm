"""Sky maps of skim-visit coverage (/data8/shared/decampm/[GRIZ]) with each visit
drawn as the DECam focal plane, at nside 512: number of visits overlapping each
pixel, and the standard deviation of their dates (yr). Only pixels with at least
MIN_VISITS visits are plotted.

The focal-plane stamp is the median gnomonic offset of each CCD's corners from
the pointing over 200 full exposures in data/y6a1.ccdcorners.fits.gz; it is
sampled on a STEP grid plus its CCD edges, and a visit counts once per pixel.

--thin applies clean_cat's exposure thinning (config.yaml values): per calendar
month, friends-of-friends pointing clusters (LINK_ARCMIN, all bands pooled)
capped at MAX_PER_MONTH exposures. clean_cat ranks by median detection error;
t_eff stands in for that here. Clusters are found on the sphere rather than in
each healpixel's tangent plane.

Writes, in dev/ (suffix _thinned with --thin):
  skims_focalplane_skymap.png, skims_focalplane_counts_nside512.npy
  skims_focalplane_datestd_skymap.png, skims_focalplane_datestd_nside512.npy
  decam_focalplane_stamp.npy  (CCD corners, (61, 2, 4) deg)
"""

import argparse
import glob
import os
import re
import sys

import healpy as hp
import numpy as np
from astropy.table import Table
from astropy.time import Time
from matplotlib.path import Path
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

DEV = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DEV)
from exposuresFromHealpixList import _radec_from_table  # noqa: E402
from skyproj_map import plot_skymap  # noqa: E402

EXPOSURES = "/home2/vwetzell/gitrepos/pixmappy/pixmappy/data/delveExposures.hdf5"
CCDCORNERS = os.path.join(DEV, "..", "data", "y6a1.ccdcorners.fits.gz")
SKIMS = "/data8/shared/decampm/"
NSIDE, STEP = 512, 0.03  # stamp sample spacing (deg), ~1/4 of an nside-512 pixel
MAX_PER_MONTH, LINK_ARCMIN = 5, 10.0
MIN_VISITS = 5
STD_VMAX = 4.0  # shared colour scale (yr) for thinned and unthinned date-spread maps
MJD_2013 = 56293.0


def basis(ra, dec):
    """Pointing unit vector and local east/north tangent vectors, each (..., 3)."""
    a, d = np.radians(ra), np.radians(dec)
    c = np.stack([np.cos(d) * np.cos(a), np.cos(d) * np.sin(a), np.sin(d)], -1)
    e = np.stack([-np.sin(a), np.cos(a), np.zeros_like(a)], -1)
    n = np.stack([-np.sin(d) * np.cos(a), -np.sin(d) * np.sin(a), np.cos(d)], -1)
    return c, e, n


def thin(idx, expo, ra, dec):
    """Indices of idx kept by clean_cat-style per-month pointing-cluster thinning."""
    ymd = Time(np.asarray(expo["mjdmid"])[idx], format="mjd").ymdhms
    month = ymd["year"] * 12 + ymd["month"]
    teff = np.asarray(expo["t_eff"], dtype="f8")[idx]
    xyz = hp.ang2vec(ra[idx], dec[idx], lonlat=True)
    chord = 2 * np.sin(np.radians(LINK_ARCMIN / 60) / 2)
    keep = []
    for m in np.unique(month):
        sel = np.flatnonzero(month == m)
        pairs = cKDTree(xyz[sel]).query_pairs(chord, output_type="ndarray")
        g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(sel.size,) * 2)
        _, lab = connected_components(g, directed=False)
        for grp in np.split(sel[np.argsort(lab, kind="stable")], np.cumsum(np.bincount(lab))[:-1]):
            keep.append(grp[np.argsort(-teff[grp], kind="stable")[:MAX_PER_MONTH]])
    return idx[np.sort(np.concatenate(keep))]


def focalplane_stamp(en, ra, dec):
    """Median (xi, eta) corners (deg) of each CCD, (61, 2, 4), and stamp sample points (rad), (P, 2)."""
    cc = Table.read(CCDCORNERS)
    u, n61 = np.unique(cc["expnum"], return_counts=True)
    pick = np.random.default_rng(1).choice(np.intersect1d(u[n61 == 61], en), 200, replace=False)
    cc = cc[np.isin(cc["expnum"], pick)]
    order = np.argsort(en)
    j = order[np.searchsorted(en, cc["expnum"], sorter=order)]
    c, e, n = basis(ra[j], dec[j])
    p = basis(np.asarray(cc["ra"])[:, :4], np.asarray(cc["dec"])[:, :4])[0]  # (M, 4, 3); entry 4 is the CCD center
    w = np.einsum("mkj,mj->mk", p, c)
    xi = np.degrees(np.einsum("mkj,mj->mk", p, e) / w)
    eta = np.degrees(np.einsum("mkj,mj->mk", p, n) / w)
    det = np.asarray(cc["detpos"])
    corners = np.array([[np.median(xi[det == d], 0), np.median(eta[det == d], 0)] for d in np.unique(det)])

    pts = []
    for xs, ys in corners:
        poly = np.column_stack([xs, ys])
        gx, gy = np.meshgrid(np.arange(xs.min(), xs.max(), STEP), np.arange(ys.min(), ys.max(), STEP))
        g = np.column_stack([gx.ravel(), gy.ravel()])
        pts.append(g[Path(poly).contains_points(g)])
        for a, b in zip(poly, np.roll(poly, -1, 0)):  # edges, so partially covered pixels count
            t = np.linspace(0, 1, int(np.ceil(np.hypot(*(b - a)) / STEP)) + 1)[:, None]
            pts.append(a + t * (b - a))
    return corners, np.radians(np.concatenate(pts))


def accumulate(k, ra, dec, t_yr, pts):
    """Per-pixel visit count and sums of t and t^2 over visits k, each pixel once per visit."""
    npix = hp.nside2npix(NSIDE)
    counts, s1, s2 = np.zeros(npix, dtype=np.int64), np.zeros(npix), np.zeros(npix)
    for s in np.array_split(k, 600):
        c, e, n = basis(ra[s], dec[s])
        v = c[:, None] + pts[None, :, :1] * e[:, None] + pts[None, :, 1:] * n[:, None]
        pix = np.sort(hp.vec2pix(NSIDE, v[..., 0], v[..., 1], v[..., 2]), axis=1)
        first = np.ones(pix.shape, bool)
        first[:, 1:] = pix[:, 1:] != pix[:, :-1]
        p = pix[first]
        tt = np.broadcast_to(t_yr[s][:, None], pix.shape)[first]
        counts += np.bincount(p, minlength=npix)
        s1 += np.bincount(p, tt, minlength=npix)
        s2 += np.bincount(p, tt * tt, minlength=npix)
    return counts, s1, s2


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--thin", action="store_true", help="apply clean_cat's per-month exposure thinning")
    args = parser.parse_args()
    tag = "_thinned" if args.thin else ""

    expo = Table.read(EXPOSURES, path="__astropy_table__")
    en, ra, dec = _radec_from_table(expo)
    corners, pts = focalplane_stamp(en, ra, dec)
    np.save(os.path.join(DEV, "decam_focalplane_stamp.npy"), corners)
    print(len(pts), "stamp points")

    skims = {int(re.search(r"D(\d+)_", os.path.basename(f)).group(1))
             for b in "GRIZ" for f in glob.glob(f"{SKIMS}{b}/D*_{b.lower()}_cat.fits")}
    k = np.flatnonzero(np.isin(en, list(skims)))
    if args.thin:
        n0 = k.size
        k = thin(k, expo, ra, dec)
        print(f"thinning kept {k.size} of {n0} visits")

    t_yr = (np.asarray(expo["mjdmid"], dtype="f8") - MJD_2013) / 365.25
    counts, s1, s2 = accumulate(k, ra, dec, t_yr, pts)
    np.save(os.path.join(DEV, f"skims_focalplane{tag}_counts_nside512.npy"), counts)

    pix = np.flatnonzero(counts >= 2)  # std undefined for a single visit
    mean = s1[pix] / counts[pix]
    std = np.sqrt(np.maximum(s2[pix] / counts[pix] - mean ** 2, 0))
    np.save(os.path.join(DEV, f"skims_focalplane{tag}_datestd_nside512.npy"), np.column_stack([pix, std]))

    keep = counts[pix] >= MIN_VISITS
    pix, std = pix[keep], std[keep]
    print(f"{k.size} visits; {np.count_nonzero(counts)} pixels covered, {pix.size} with >= {MIN_VISITS} visits; "
          f"max count {counts.max()}; date std median {np.median(std):.2f} yr, max {std.max():.2f} yr")
    who = "thinned" if args.thin else "DECam"
    plot_skymap(pixels=pix, values=counts[pix], nside=NSIDE, norm="log", vmin=MIN_VISITS, vmax=counts.max(),
                xsize=4000, dpi=600, label=f"{who} visits overlapping pixel",
                out=os.path.join(DEV, f"skims_focalplane{tag}_skymap.png"))
    plot_skymap(pixels=pix, values=std, nside=NSIDE, vmin=0, vmax=STD_VMAX, xsize=4000, dpi=600,
                label=("thinned " if args.thin else "") + "std. dev. of visit dates (yr)",
                out=os.path.join(DEV, f"skims_focalplane{tag}_datestd_skymap.png"))


if __name__ == "__main__":
    main()
