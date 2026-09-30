"""Exposure-by-exposure pulls of GPR-corrected detection positions against
Gaia DR3 positions projected to each exposure's epoch.

For every detection in a GPR2 catalog (new_rd) matched to a Gaia 5-parameter
source, in the local (East, North) frame at the star:

    delta = new_rd - gaia(t),   gaia(t) = gaia(J2016.0) + mu * dt + parallax * P(t)

P(t) is the standard parallax factor from the Earth's barycentric position at
the exposure MJD. Gaia's own 5x5 covariance is propagated to t.

Covariance terms, all rotated into local (E, N):
  E    ERRWIN ellipse (SExtractor), from the position-corrected exposure file
  G    GPR cov_model (tangent plane about the GPR file's RA0/DEC0)
  S    GPR 'sig' (isotropic; Gary's own measurement sigma), for comparison
  Cg   propagated Gaia covariance
The pipeline's per-detection covariance is E + G (detection_packaging), so
the headline pull is delta / sqrt(E + G + Cg), per component.

The GPR 'ref' flag marks Gaia stars used to train that exposure's GPR --
their residuals are pulled toward zero by the fit itself, so ref and non-ref
detections are kept separate; non-ref Gaia matches are the fair test.
"""

import argparse
import glob
import os
import sys
from functools import lru_cache
from multiprocessing import Pool

import fitsio
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import get_body_barycentric
from astropy.time import Time
from scipy.spatial import cKDTree

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from highpm.gnomonic_converter import gnomonicJacobian  # noqa: E402

MAS = 3.6e6  # mas per degree
GAIA_EPOCH_MJD = 57388.5  # J2016.0
BANDS = "grizY"

OUT_DTYPE = [
    ("expnum", "i4"), ("band", "i1"), ("ref", "?"), ("color_known", "?"), ("gmag", "f4"),
    ("d_a", "f4"), ("d_d", "f4"),              # new_rd - gaia(t), mas (E, N)
    ("E_aa", "f4"), ("E_dd", "f4"), ("E_ad", "f4"),
    ("G_aa", "f4"), ("G_dd", "f4"), ("G_ad", "f4"),
    ("Cg_aa", "f4"), ("Cg_dd", "f4"), ("Cg_ad", "f4"),
    ("S2", "f4"),                              # sig^2, mas^2
]


def robust_std(x):
    x = x[np.isfinite(x)]
    return 1.4826 * np.median(np.abs(x - np.median(x))) if len(x) else np.nan


@lru_cache(maxsize=64)
def load_gaia_pixel(gaia_dir, pix):
    cols = ["RA", "DEC", "RA_ERROR", "DEC_ERROR", "PARALLAX", "PARALLAX_ERROR", "PMRA", "PMRA_ERROR",
            "PMDEC", "PMDEC_ERROR", "RA_DEC_CORR", "RA_PARALLAX_CORR", "RA_PMRA_CORR", "RA_PMDEC_CORR",
            "DEC_PARALLAX_CORR", "DEC_PMRA_CORR", "DEC_PMDEC_CORR", "PARALLAX_PMRA_CORR",
            "PARALLAX_PMDEC_CORR", "PMRA_PMDEC_CORR", "RUWE", "PHOT_G_MEAN_MAG"]
    f = os.path.join(gaia_dir, f"GaiaSource_{pix:05d}.fits")
    return fitsio.read(f, columns=cols) if os.path.exists(f) else None


def gaia_cov5(g):
    """(N, 5, 5) covariance over (ra*, dec, parallax, pmra, pmdec), mas & mas/yr."""
    s = np.array([g["RA_ERROR"], g["DEC_ERROR"], g["PARALLAX_ERROR"], g["PMRA_ERROR"], g["PMDEC_ERROR"]]).T
    corr = np.repeat(np.eye(5)[None], len(g), axis=0)
    pairs = {(0, 1): "RA_DEC_CORR", (0, 2): "RA_PARALLAX_CORR", (0, 3): "RA_PMRA_CORR",
             (0, 4): "RA_PMDEC_CORR", (1, 2): "DEC_PARALLAX_CORR", (1, 3): "DEC_PMRA_CORR",
             (1, 4): "DEC_PMDEC_CORR", (2, 3): "PARALLAX_PMRA_CORR", (2, 4): "PARALLAX_PMDEC_CORR",
             (3, 4): "PMRA_PMDEC_CORR"}
    for (i, j), c in pairs.items():
        corr[:, i, j] = corr[:, j, i] = np.nan_to_num(g[c])
    return corr * s[:, :, None] * s[:, None, :]


def to_local(cxx, cyy, cxy, ra, dec, ra0, dec0):
    """Tangent-plane (xi, eta about ra0/dec0) covariance -> local (E, N)."""
    a, b, c, d = gnomonicJacobian(ra, dec, ra0, dec0)
    cosd = np.cos(np.radians(dec))
    # J maps local (E, N) -> (xi, eta); we need J^-1 C J^-T
    j00, j01, j10, j11 = a / cosd, b, c / cosd, d
    det = j00 * j11 - j01 * j10
    i00, i01, i10, i11 = j11 / det, -j01 / det, -j10 / det, j00 / det
    return (i00 * i00 * cxx + 2 * i00 * i01 * cxy + i01 * i01 * cyy,
            i10 * i10 * cxx + 2 * i10 * i11 * cxy + i11 * i11 * cyy,
            i00 * i10 * cxx + (i00 * i11 + i01 * i10) * cxy + i01 * i11 * cyy)


def process_exposure(task):
    expnum, a = task
    gpr_files = glob.glob(os.path.join(a.gpr_dir, "*", f"gpr*_{expnum:07d}_*.fits"))
    pc_file = os.path.join(a.base, "PositionCorrectedExposureCatalog", f"position_corrected_{expnum:08d}.fits")
    if not gpr_files or not os.path.exists(pc_file):
        return None
    gpr = fitsio.read(gpr_files[0], columns=["id", "new_rd", "cov_model", "sig", "ref", "color_source"])
    hdr = fitsio.read_header(gpr_files[0], ext=1)
    ra0, dec0 = hdr["RA0"], hdr["DEC0"]
    pc = fitsio.read(pc_file, columns=["CCDNUM", "OBJECT_NUMBER", "MJD", "BAND", "FLAGS", "IMAFLAGS_ISO",
                                       "ERRAWIN_WORLD", "ERRBWIN_WORLD", "ERRTHETAWIN_J2000"])
    pc = pc[(pc["FLAGS"] < 4) & (pc["IMAFLAGS_ISO"] == 0)]
    mjd = float(np.median(pc["MJD"]))
    band = BANDS.index(pc["BAND"][0].strip()) if pc["BAND"][0].strip() in BANDS else -1

    # join GPR rows to the position-corrected rows by CCDNUM_OBJECTNUMBER
    ids = np.char.split(np.char.decode(gpr["id"]) if gpr["id"].dtype.kind == "S" else gpr["id"].astype(str), "_")
    ccd = np.array([int(s[0]) for s in ids])
    obj = np.array([int(s[1]) for s in ids])
    key_g = ccd.astype(np.int64) * 10_000_000 + obj
    key_p = pc["CCDNUM"].astype(np.int64) * 10_000_000 + pc["OBJECT_NUMBER"]
    common, ig, ip = np.intersect1d(key_g, key_p, return_indices=True)
    gpr, pc = gpr[ig], pc[ip]
    ra, dec = gpr["new_rd"][:, 0], gpr["new_rd"][:, 1]

    # Gaia sources near the exposure, projected to this epoch
    pixels = np.unique(hp.ang2pix(32, ra, dec, lonlat=True))
    parts = [g for g in (load_gaia_pixel(a.gaia_dir, int(p)) for p in pixels) if g is not None]
    if not parts:
        return None
    g = np.concatenate(parts)
    g = g[np.isfinite(g["PMRA"]) & (g["PMRA"] != 0) & (g["PMDEC"] != 0) & (g["RUWE"] <= a.max_ruwe)]
    dt = (mjd - GAIA_EPOCH_MJD) / 365.25
    earth = get_body_barycentric("earth", Time(mjd, format="mjd", scale="tdb")).xyz.to_value("au")
    X, Y, Z = earth
    ga, gd = np.radians(g["RA"]), np.radians(g["DEC"])
    pa = X * np.sin(ga) - Y * np.cos(ga)
    pd = X * np.cos(ga) * np.sin(gd) + Y * np.sin(ga) * np.sin(gd) - Z * np.cos(gd)
    off_a = g["PMRA"] * dt + g["PARALLAX"] * pa     # mas, East
    off_d = g["PMDEC"] * dt + g["PARALLAX"] * pd    # mas, North
    g_dec = g["DEC"] + off_d / MAS
    g_ra = g["RA"] + off_a / MAS / np.cos(np.radians(g_dec))

    def unit(r, d):
        r, d = np.radians(r), np.radians(d)
        return np.array([np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)]).T

    dist, gi = cKDTree(unit(g_ra, g_dec)).query(unit(ra, dec), distance_upper_bound=np.radians(a.match_arcsec / 3600))
    ok = np.isfinite(dist)
    if not ok.any():
        return None
    gpr, pc, ra, dec, gi = gpr[ok], pc[ok], ra[ok], dec[ok], gi[ok]
    gm, gra, gdec = g[gi], g_ra[gi], g_dec[gi]

    out = np.zeros(len(gm), dtype=OUT_DTYPE)
    out["expnum"] = expnum
    out["band"] = band
    out["ref"] = gpr["ref"]
    out["color_known"] = gpr["color_source"] != -1
    out["gmag"] = gm["PHOT_G_MEAN_MAG"]
    out["d_a"] = ((ra - gra + 180) % 360 - 180) * np.cos(np.radians(dec)) * MAS
    out["d_d"] = (dec - gdec) * MAS

    # ERRWIN ellipse is already local (E, N); arcsec -> mas
    A = pc["ERRAWIN_WORLD"] * MAS
    B = pc["ERRBWIN_WORLD"] * MAS
    th = np.radians(pc["ERRTHETAWIN_J2000"])
    ee = A * A - B * B
    out["E_aa"] = (A * A + B * B + ee * np.cos(th)) / 2
    out["E_dd"] = (A * A + B * B - ee * np.cos(th)) / 2
    out["E_ad"] = ee * np.sin(th) / 2

    cm = gpr["cov_model"] * 1e6  # arcsec^2 -> mas^2
    out["G_aa"], out["G_dd"], out["G_ad"] = to_local(cm[:, 0, 0], cm[:, 1, 1], cm[:, 0, 1], ra, dec, ra0, dec0)

    ia, idd = np.radians(gm["RA"]), np.radians(gm["DEC"])
    pa_m = X * np.sin(ia) - Y * np.cos(ia)
    pd_m = X * np.cos(ia) * np.sin(idd) + Y * np.sin(ia) * np.sin(idd) - Z * np.cos(idd)
    J = np.zeros((len(gm), 2, 5))
    J[:, 0, 0] = J[:, 1, 1] = 1
    J[:, 0, 2], J[:, 1, 2] = pa_m, pd_m
    J[:, 0, 3] = J[:, 1, 4] = dt
    Cg = np.einsum("nij,njk,nlk->nil", J, gaia_cov5(gm), J)
    out["Cg_aa"], out["Cg_dd"], out["Cg_ad"] = Cg[:, 0, 0], Cg[:, 1, 1], Cg[:, 0, 1]
    out["S2"] = (gpr["sig"] * 1000) ** 2
    return out


def summarize(det, gmin, gmax):
    """Per-exposure robust pull stats (non-ref, known colour, gmin<=G<=gmax)."""
    sel = (~det["ref"]) & det["color_known"] & (det["gmag"] >= gmin) & (det["gmag"] <= gmax)
    d = det[sel]
    pa = d["d_a"] / np.sqrt(d["E_aa"] + d["G_aa"] + d["Cg_aa"])
    pdd = d["d_d"] / np.sqrt(d["E_dd"] + d["G_dd"] + d["Cg_dd"])
    order = np.argsort(d["expnum"], kind="stable")
    d, pa, pdd = d[order], pa[order], pdd[order]
    ex, starts = np.unique(d["expnum"], return_index=True)
    ends = np.r_[starts[1:], len(d)]
    rows = []
    for e, s, t in zip(ex, starts, ends):
        if t - s < 30:
            continue
        sl = slice(s, t)
        rows.append((e, d["band"][s], t - s, robust_std(pa[sl]), robust_std(pdd[sl]),
                     np.median(d["d_a"][sl]), np.median(d["d_d"][sl]),
                     np.median(np.sqrt(d["G_aa"][sl])), np.median(np.sqrt(d["G_dd"][sl])),
                     np.median(d["E_aa"][sl] / (d["E_aa"][sl] + d["G_aa"][sl]))))
    return np.array(rows, dtype=[("expnum", "i4"), ("band", "i1"), ("n", "i4"), ("pull_std_a", "f4"),
                                 ("pull_std_d", "f4"), ("med_d_a", "f4"), ("med_d_d", "f4"),
                                 ("med_sigG_a", "f4"), ("med_sigG_d", "f4"), ("med_Eshare", "f4")])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="/data8/shared/decampm/PMSculptor_8deg_garyb")
    parser.add_argument("--gpr-dir", default="/data8/shared/decampm/GPR2/CAT")
    parser.add_argument("--gaia-dir", default="/data8/shared/decampm/Gaia/")
    parser.add_argument("--expnums", type=int, nargs="*", default=None, help="default: all in the run")
    parser.add_argument("--match-arcsec", type=float, default=0.5)
    parser.add_argument("--max-ruwe", type=float, default=1.4)
    parser.add_argument("--gmin", type=float, default=16.0)
    parser.add_argument("--gmax", type=float, default=21.0)
    parser.add_argument("--cores", type=int, default=16)
    parser.add_argument("--cache", default="gaia_exposure_pulls_detections.npy")
    parser.add_argument("--summary", default="gaia_exposure_pulls.fits")
    parser.add_argument("-o", "--out", default="gaia_exposure_pulls.png")
    args = parser.parse_args()

    if os.path.exists(args.cache):
        det = np.load(args.cache)
        print(f"loaded {args.cache}: {len(det)} detections")
    else:
        exps = args.expnums or [int(e) for e in np.load(os.path.join(args.base, "pmsculptor8deg_exposures.npy"))]
        with Pool(args.cores) as pool:
            parts = [p for p in pool.imap_unordered(process_exposure, [(e, args) for e in exps], chunksize=4)
                     if p is not None]
        det = np.concatenate(parts)
        np.save(args.cache, det)
        print(f"{len(parts)} exposures, {len(det)} Gaia-matched detections -> {args.cache}")

    in_g = (det["gmag"] >= args.gmin) & (det["gmag"] <= args.gmax) & det["color_known"]
    print(f"\nAll detections, known colour, {args.gmin} <= G <= {args.gmax}: pull robust std (E, N)")
    for name, sel in (("non-ref", in_g & ~det["ref"]), ("ref (GPR training)", in_g & det["ref"])):
        d = det[sel]
        for label, va, vd in (("E+G+Cg (pipeline)", d["E_aa"] + d["G_aa"] + d["Cg_aa"], d["E_dd"] + d["G_dd"] + d["Cg_dd"]),
                              ("E+Cg (no GPR term)", d["E_aa"] + d["Cg_aa"], d["E_dd"] + d["Cg_dd"]),
                              ("sig^2+G+Cg", d["S2"] + d["G_aa"] + d["Cg_aa"], d["S2"] + d["G_dd"] + d["Cg_dd"])):
            print(f"  {name:20s} {label:20s} n={len(d):9d}  {robust_std(d['d_a'] / np.sqrt(va)):.3f}  "
                  f"{robust_std(d['d_d'] / np.sqrt(vd)):.3f}")

    # Which term is off: pull std vs ERRWIN share of the pipeline variance (non-ref).
    d = det[in_g & ~det["ref"]]
    share = (d["E_aa"] + d["E_dd"]) / (d["E_aa"] + d["E_dd"] + d["G_aa"] + d["G_dd"])
    pa = d["d_a"] / np.sqrt(d["E_aa"] + d["G_aa"] + d["Cg_aa"])
    pdd = d["d_d"] / np.sqrt(d["E_dd"] + d["G_dd"] + d["Cg_dd"])
    sedges = np.linspace(0, 1, 11)
    sc = 0.5 * (sedges[1:] + sedges[:-1])
    sb = np.digitize(share, sedges) - 1
    s_share = np.array([[robust_std(pa[sb == i]), robust_std(pdd[sb == i]), (sb == i).sum()] for i in range(10)])
    print("\nnon-ref pull std vs ERRWIN share of E+G:\n  share    n        E       N")
    for c, (sa, sd, n) in zip(sc, s_share):
        print(f"  {c:.2f} {int(n):9d}   {sa:.3f}   {sd:.3f}")

    gedges = np.arange(args.gmin, args.gmax + 0.01, 0.5)
    gc = 0.5 * (gedges[1:] + gedges[:-1])
    gb = np.digitize(d["gmag"], gedges) - 1
    s_g = np.array([[robust_std(pa[gb == i]), robust_std(pdd[gb == i])] for i in range(len(gc))])

    summ = summarize(det, args.gmin, args.gmax)
    fitsio.write(args.summary, summ, clobber=True)
    print(f"\n{len(summ)} exposures summarized -> {args.summary}")
    for c, lab in (("pull_std_a", "E"), ("pull_std_d", "N")):
        q = np.percentile(summ[c], [5, 25, 50, 75, 95])
        print(f"  per-exposure pull std {lab}: 5/25/50/75/95% = " + " ".join(f"{x:.3f}" for x in q))
    for b in np.unique(summ["band"]):
        s = summ[summ["band"] == b]
        print(f"  band {BANDS[b]}: {len(s):4d} exposures, median pull std E/N = "
              f"{np.median(s['pull_std_a']):.3f} / {np.median(s['pull_std_d']):.3f}")

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    ax = axes[0, 0]
    bins = np.linspace(0.3, 2.0, 69)
    ax.hist(summ["pull_std_a"], bins=bins, histtype="step", lw=2, label="East")
    ax.hist(summ["pull_std_d"], bins=bins, histtype="step", lw=2, label="North")
    ax.axvline(1, color="k", ls="--")
    ax.set_xlabel("per-exposure robust pull std")
    ax.set_ylabel("exposures")
    ax.set_title(f"non-ref Gaia matches, {args.gmin:g}<=G<={args.gmax:g}, E+G+Cg")
    ax.legend()
    ax.grid()

    ax = axes[0, 1]
    for b in np.unique(summ["band"]):
        s = summ[summ["band"] == b]
        ax.scatter(s["med_sigG_a"], s["pull_std_a"], s=4, alpha=0.5, label=BANDS[b])
    ax.axhline(1, color="k", ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("exposure median GPR sigma (E) [mas]")
    ax.set_ylabel("pull std (E)")
    ax.set_title("vs GPR (turbulence) error amplitude")
    ax.legend(markerscale=4)
    ax.grid()

    ax = axes[0, 2]
    for b in np.unique(summ["band"]):
        s = summ[summ["band"] == b]
        ax.scatter(s["med_Eshare"], s["pull_std_a"], s=4, alpha=0.5, label=BANDS[b])
    ax.axhline(1, color="k", ls="--")
    ax.set_xlabel("exposure median ERRWIN share of E+G")
    ax.set_ylabel("pull std (E)")
    ax.set_title("per exposure vs ERRWIN share")
    ax.grid()

    ax = axes[1, 0]
    ax.plot(sc, s_share[:, 0], "o-", label="East")
    ax.plot(sc, s_share[:, 1], "s-", label="North")
    ax.axhline(1, color="k", ls="--")
    ax.set_xlabel("ERRWIN share of E+G (per detection)")
    ax.set_ylabel("robust pull std")
    ax.set_title("share 0: GPR term dominates; share 1: ERRWIN dominates")
    ax.legend()
    ax.grid()

    ax = axes[1, 1]
    ax.plot(gc, s_g[:, 0], "o-", label="East")
    ax.plot(gc, s_g[:, 1], "s-", label="North")
    ax.axhline(1, color="k", ls="--")
    ax.set_xlabel("Gaia G")
    ax.set_ylabel("robust pull std")
    ax.set_title("all non-ref detections vs G")
    ax.legend()
    ax.grid()

    ax = axes[1, 2]
    ax.hist(summ["med_d_a"], bins=np.linspace(-10, 10, 81), histtype="step", lw=2, label="East")
    ax.hist(summ["med_d_d"], bins=np.linspace(-10, 10, 81), histtype="step", lw=2, label="North")
    ax.set_xlabel("per-exposure median offset new_rd - gaia(t) [mas]")
    ax.set_ylabel("exposures")
    ax.legend()
    ax.grid()

    fig.suptitle("GPR2 detections vs Gaia DR3 projected to exposure epoch (PMSculptor 8 deg exposures)", fontsize=14)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Saved {args.out}")
