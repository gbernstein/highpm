"""Two follow-up tests on (ours - Gaia) residuals, pooled over several healpixels.

A. Does the constant PM offset depend on time coverage or filter mix?
   Per-star detection stats (mean epoch, span, N, band fractions) are built
   from the modest1 detection lists + cleaned detection catalog.
B. Does the parallax colour term depend on how the fit treated colour?
   Split stars into fixed-colour (color_err == 0, colour known per detection)
   and fitted-colour (color_err > 0) fits, and regress dparallax on Gaia
   BP-RP, band fractions and our fitted colour.
"""

import argparse
import os
import sys

import fitsio
import matplotlib.pyplot as plt
import numpy as np
import numpy.lib.recfunctions as rfn
from astropy import units as u
from astropy.coordinates import SkyCoord

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from healpix_region_data import load_coadd, load_movers

PM = "/data8/shared/decampm/PMSculptor_8deg/PMCatalog/"
HDC = "/data8/shared/decampm/PMSculptor_8deg/HealpixDetectionCatalog/"
MJD_REF = 57388.0
BANDS = "grizY"


def robust_std(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def det_stats(hp):
    f = f"{PM}real_PM_hp{hp:05d}.fits"
    m = fitsio.read(f, ext="modest1_movers")
    d = fitsio.read(f, ext="modest1_detections")
    d = d[~d["clipped"]]
    c = fitsio.read(f"{HDC}cleaned_detections_hp{hp:05d}.fits", columns=["MJD", "BAND"])
    r = c[d["detections"]]
    n = int(m["idx"].max()) + 1
    cnt = np.bincount(d["idx"], minlength=n)
    t = (r["MJD"] - MJD_REF) / 365.25
    out = {"n_det": cnt, "t_mean": np.bincount(d["idx"], t, n) / np.maximum(cnt, 1)}
    tmin = np.full(n, np.inf)
    tmax = np.full(n, -np.inf)
    np.minimum.at(tmin, d["idx"], t)
    np.maximum.at(tmax, d["idx"], t)
    out["span"] = tmax - tmin
    for b in BANDS:
        out["f_" + b] = np.bincount(d["idx"], (r["BAND"] == b).astype(float), n) / np.maximum(cnt, 1)
    return {k: v[m["idx"]] for k, v in out.items()}


def load_hp(hp, gmag, max_ruwe, match_arcsec=0.3):
    mv = load_movers(PM, 32, {hp}, pattern="real_PM_hp*.fits", exts=("modest1_movers",))
    st = det_stats(hp)
    # load_movers dedups/filters, so re-key on idx
    full = fitsio.read(f"{PM}real_PM_hp{hp:05d}.fits", ext="modest1_movers")
    lut = {int(i): k for k, i in enumerate(full["idx"])}
    rows = np.array([lut[int(i)] for i in mv["idx"]])
    stats = rfn.merge_arrays([np.rec.fromarrays([v[rows] for v in st.values()], names=list(st))], flatten=True)
    g = load_coadd("/data8/shared/decampm/Gaia/", {hp}, pattern="GaiaSource_*.fits")
    g = g[np.isfinite(g["PMRA"]) & ~((g["PMRA"] == 0) & (g["PMDEC"] == 0) & (g["PARALLAX"] == 0))]
    mc = SkyCoord(mv["ra"] * u.deg, mv["dec"] * u.deg)
    gc = SkyCoord(g["RA"] * u.deg, g["DEC"] * u.deg)
    i1, d1, _ = mc.match_to_catalog_sky(gc)
    _, d2, _ = mc.match_to_catalog_sky(gc, nthneighbor=2)
    ok = (d1.arcsec < match_arcsec) & (d2.arcsec >= match_arcsec)
    t = rfn.merge_arrays([mv[ok], stats[ok], g[i1[ok]]], flatten=True, usemask=False)
    k = (t["RUWE"] <= max_ruwe) & (t["PHOT_G_MEAN_MAG"] >= gmag[0]) & (t["PHOT_G_MEAN_MAG"] <= gmag[1])
    return t[k]


def binned_report(name, x, ys, nb=5):
    edges = np.quantile(x, np.linspace(0, 1, nb + 1))
    idx = np.clip(np.digitize(x, edges[1:-1]), 0, nb - 1)
    print(f"  vs {name}:")
    for i in range(nb):
        s = idx == i
        line = f"    [{edges[i]:7.3f},{edges[i + 1]:7.3f}] n={s.sum():5d} "
        for lab, y in ys.items():
            line += f" {lab}={np.median(y[s]):+.3f}+/-{1.253 * robust_std(y[s]) / np.sqrt(s.sum()):.3f}"
        print(line)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--healpix", type=int, nargs="+", default=[9285, 8644, 9032, 9800, 10183, 9027])
    ap.add_argument("--gmag-range", type=float, nargs=2, default=[13, 19.5])
    ap.add_argument("--max-ruwe", type=float, default=1.4)
    ap.add_argument("-o", "--out", default="gaia_residual_systematics.png")
    a = ap.parse_args()

    t = np.concatenate([load_hp(h, a.gmag_range, a.max_ruwe) for h in a.healpix])
    dpmra, dpmdec = t["pmra"] - t["PMRA"], t["pmdec"] - t["PMDEC"]
    dpar = t["parallax"] - t["PARALLAX"]
    good = (np.abs(dpmra - np.median(dpmra)) < 5 * robust_std(dpmra)) & \
           (np.abs(dpmdec - np.median(dpmdec)) < 5 * robust_std(dpmdec)) & np.isfinite(t["BP_RP"])
    t, dpmra, dpmdec, dpar = t[good], dpmra[good], dpmdec[good], dpar[good]
    print(f"\n=== pooled N={len(t)} from {len(a.healpix)} healpixels")
    print(f"overall median dPMRA={np.median(dpmra):+.3f} dPMDEC={np.median(dpmdec):+.3f} dpar={np.median(dpar):+.3f}")

    print("\n### TEST A: PM offset vs time coverage / filter mix (median +/- err)")
    pmy = {"dPMRA": dpmra, "dPMDEC": dpmdec}
    for nme in ("t_mean", "span", "n_det", "f_g", "f_r", "f_i", "f_z", "PHOT_G_MEAN_MAG"):
        x = t[nme].astype(float)
        if np.ptp(x) > 0:
            binned_report(nme, x + 1e-9 * np.random.default_rng(0).standard_normal(len(x)), pmy)
    fixed = t["color_err"] == 0
    for lab, s in (("fixed-colour", fixed), ("fitted-colour", ~fixed)):
        print(f"  {lab}: n={s.sum()} dPMRA={np.median(dpmra[s]):+.3f} dPMDEC={np.median(dpmdec[s]):+.3f}")
    for nme in ("t_mean", "span", "n_det"):
        for lab, y in pmy.items():
            A = np.vstack([np.ones(len(t)), t[nme]]).T
            p, *_ = np.linalg.lstsq(A, y, rcond=None)
            res = y - A @ p
            se = robust_std(res) / (np.sqrt(len(t)) * np.std(t[nme]))
            print(f"  linear {lab} vs {nme}: slope {p[1]:+.4f} +/- {se:.4f}")

    print("\n### TEST B: parallax offset vs colour handling")
    for lab, s in (("fixed-colour fits", fixed), ("fitted-colour fits", ~fixed)):
        x, y = t["BP_RP"][s], dpar[s]
        p, cov = np.polyfit(x, y, 1, cov=True)
        print(f"  {lab}: n={s.sum()} median dpar={np.median(y):+.3f}, slope vs BP-RP {p[0]:+.3f}+/-{np.sqrt(cov[0, 0]):.3f}, "
              f"intercept {p[1]:+.3f}, scatter about fit {robust_std(y - np.polyval(p, x)):.3f}")
    print("  multi-regression dpar ~ 1 + BP_RP + f_g + f_r + f_i + t_mean (all stars):")
    X = np.vstack([np.ones(len(t)), t["BP_RP"], t["f_g"], t["f_r"], t["f_i"], t["t_mean"]]).T
    p, *_ = np.linalg.lstsq(X, dpar, rcond=None)
    res = dpar - X @ p
    se = robust_std(res) * np.sqrt(np.diag(np.linalg.inv(X.T @ X)))
    for nme, pv, sv in zip(["const", "BP_RP", "f_g", "f_r", "f_i", "t_mean"], p, se):
        print(f"    {nme:6s} {pv:+.3f} +/- {sv:.3f}")
    print(f"    residual robust scatter {robust_std(res):.3f} (was {robust_std(dpar):.3f})")
    fc = ~fixed
    print(f"  fitted colour vs Gaia BP-RP (fitted-colour stars): corr {np.corrcoef(t['color'][fc], t['BP_RP'][fc])[0, 1]:+.3f}")
    for lab, s in (("fixed", fixed), ("fitted", fc)):
        print(f"  {lab}: corr(dpar, our colour)={np.corrcoef(dpar[s], t['color'][s])[0, 1]:+.3f}, "
              f"corr(dpar, BP-RP)={np.corrcoef(dpar[s], t['BP_RP'][s])[0, 1]:+.3f}")
    binned_report("BP_RP (fixed)", t["BP_RP"][fixed], {"dpar": dpar[fixed]}, 6)
    binned_report("BP_RP (fitted)", t["BP_RP"][fc], {"dpar": dpar[fc]}, 6)

    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    for lab, s, c in (("fixed colour", fixed, "steelblue"), ("fitted colour", fc, "crimson")):
        ax[0].scatter(t["BP_RP"][s], dpar[s], s=2, alpha=0.3, color=c, label=lab)
    ax[0].set_xlabel("Gaia BP-RP"); ax[0].set_ylabel("dParallax [mas]"); ax[0].legend(); ax[0].set_ylim(-5, 2)
    ax[1].scatter(t["color"][fc], dpar[fc], s=2, alpha=0.3); ax[1].set_xlabel("our fitted colour (fitted-colour stars)")
    ax[1].set_ylabel("dParallax [mas]"); ax[1].set_ylim(-5, 2)
    ax[2].scatter(t["t_mean"], dpmra, s=2, alpha=0.2, label="dPMRA")
    ax[2].scatter(t["t_mean"], dpmdec, s=2, alpha=0.2, label="dPMDEC")
    ax[2].set_ylim(-3, 3); ax[2].set_xlabel("mean detection epoch - 2016 [yr]"); ax[2].set_ylabel("dPM [mas/yr]"); ax[2].legend()
    plt.tight_layout(); plt.savefig(a.out, dpi=110); print("Saved", a.out)
