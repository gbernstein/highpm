"""
Build an augmented per-exposure completeness table.

Measured (m50, k, c) come from y6a1c.exposures.positions.fits. For exposures
that have a T_EFF (from all_desy6.csv) but no measured completeness, estimate
(m50, k, c) from the t_eff relations fit in dev/TeffTesting.ipynb and flag them
with estimated=True. Both fake-injection and real-star filtering read the result.
"""

import csv
import os

import numpy as np
import fitsio
import h5py

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
MEASURED = os.path.join(DATA, "y6a1c.exposures.positions.fits")
DESY6 = os.path.join(DATA, "all_desy6.csv")
DELVE = os.path.expanduser("~/gitrepos/pixmappy/pixmappy/data/delveExposures.hdf5")
OUT = os.path.join(DATA, "delve.exposures.completeness.fits")

BANDS = ("g", "r", "i", "z")  # ponytail: Y unused by pipeline (band_idx is g/r/i/z)
M50_SLOPE = 1.25  # physical mag-per-dex; only the intercept is fit (per TeffTesting.ipynb)


def load_desy6(path):
    """Return (expnum, band, t_eff, exptime) arrays for rows with a parseable T_EFF."""
    exp, band, teff, exptime = [], [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                t = float(row["T_EFF"])
            except (ValueError, KeyError):
                continue
            exp.append(int(row["EXPNUM"]))
            band.append(row["BAND"])
            teff.append(t)
            exptime.append(float(row["EXPTIME"]))
    return (np.array(exp, "i8"), np.array(band, "U1"),
            np.array(teff, "f8"), np.array(exptime, "f8"))


def load_delve(path):
    """Return (expnum, band, t_eff, exptime) from the DELVE astropy-table HDF5, if present."""
    if not os.path.exists(path):
        return (np.empty(0, "i8"), np.empty(0, "U1"), np.empty(0, "f8"), np.empty(0, "f8"))
    with h5py.File(path, "r") as f:
        ds = f["__astropy_table__"]  # only read the fields we need, not the nested ones
        expnum = ds["expnum"][:].astype("i8")
        band = np.char.decode(ds["band"][:]).astype("U1")  # S1 bytes -> str
        teff = ds["t_eff"][:].astype("f8")
        exptime = ds["exptime"][:].astype("f8")
    return expnum, band, teff, exptime


def gather_exposures():
    """Union of exposure sources (DES Y6 csv + DELVE hdf5)."""
    des, delve = load_desy6(DESY6), load_delve(DELVE)
    return tuple(np.concatenate([des[i], delve[i]]) for i in range(4))


def fit_band(m50, k, c, logt):
    """m0 (fixed-slope intercept), k0, s (linear), median c for one band."""
    m0 = np.mean(m50 - M50_SLOPE * logt)
    k0, s = np.polyfit(logt, k, 1)[::-1]  # np.polyfit returns [s, k0]
    return m0, k0, s, np.median(c)


def build():
    meas = fitsio.read(MEASURED, ext=1, columns=["expnum", "band", "m50", "k", "c"])
    exp_src, band_src, teff_src, exptime_src = gather_exposures()
    teff_by_exp = dict(zip(exp_src.tolist(), teff_src.tolist()))
    measured_set = set(meas["expnum"].tolist())

    # Per-band t_eff -> (m50, k, c) fits from measured rows that also have a T_EFF.
    coeffs = {}
    for b in BANDS:
        sel = meas["band"] == b
        t = np.array([teff_by_exp.get(int(e), np.nan) for e in meas["expnum"][sel]])
        ok = np.isfinite(t) & (t > 0)
        coeffs[b] = fit_band(
            meas["m50"][sel][ok], meas["k"][sel][ok], meas["c"][sel][ok],
            np.log10(t[ok]),
        )

    # Output = all measured rows (estimated=False) + estimates for missing g/r/i/z.
    out_exp = meas["expnum"].tolist()
    out_band = list(meas["band"])
    out_m50 = meas["m50"].tolist()
    out_k = meas["k"].tolist()
    out_c = meas["c"].tolist()
    out_est = [False] * len(out_exp)
    est_exptime = []  # exptime of each estimated row, for the extrapolation flag

    seen = set(measured_set)  # dedup estimates across overlapping sources
    for e, b, t, x in zip(exp_src.tolist(), band_src.tolist(),
                          teff_src.tolist(), exptime_src.tolist()):
        if e in seen or b not in BANDS or not (t > 0):
            continue
        seen.add(e)
        m0, k0, s, cmed = coeffs[b]
        lt = np.log10(t)
        out_exp.append(e)
        out_band.append(b)
        out_m50.append(m0 + M50_SLOPE * lt)
        out_k.append(k0 + s * lt)
        out_c.append(cmed)
        out_est.append(True)
        est_exptime.append(x)

    # Flag estimates whose exposure time is in the 5-95% tails. The fit sample is
    # entirely 90 s, so extrapolation is really about exposure length differing from
    # what the relation was calibrated on. ponytail: tails of the estimated exptime
    # distribution; tighten if you'd rather cut against the fit sample's 90 s.
    x_lo, x_hi = np.percentile(est_exptime, [5, 95]) if est_exptime else (0, np.inf)
    out_ext = [False] * (len(out_exp) - len(est_exptime)) + [
        (x < x_lo or x > x_hi) for x in est_exptime
    ]

    out = np.empty(len(out_exp), dtype=[
        ("expnum", "i4"), ("band", "U1"),
        ("m50", "f8"), ("k", "f8"), ("c", "f8"),
        ("estimated", "?"), ("extrapolated", "?"),
    ])
    out["expnum"] = out_exp
    out["band"] = out_band
    out["m50"] = out_m50
    out["k"] = out_k
    out["c"] = out_c
    out["estimated"] = out_est
    out["extrapolated"] = out_ext
    return out, coeffs, (x_lo, x_hi)


def main():
    out, coeffs, (x_lo, x_hi) = build()
    fitsio.write(OUT, out, clobber=True)
    n_est = int(out["estimated"].sum())
    n_ext = int(out["extrapolated"].sum())
    print(f"wrote {OUT}")
    print(f"  {len(out)} rows: {len(out) - n_est} measured, {n_est} estimated "
          f"({n_ext} extrapolated: exptime outside [{x_lo:.0f}s, {x_hi:.0f}s])")
    for b in BANDS:
        m0, k0, s, cmed = coeffs[b]
        print(f"  {b}: m50={m0:.2f}+1.25*log10(teff)  k={k0:.2f}{s:+.2f}*log10(teff)  c={cmed:.3f}")
    return out, coeffs


if __name__ == "__main__":
    out, coeffs = main()

    # ponytail: one self-check that fails if the join/fit/flagging breaks.
    meas = fitsio.read(MEASURED, ext=1, columns=["expnum"])
    kept = out[~out["estimated"]]
    assert len(kept) == len(meas), "measured rows dropped or duplicated"
    assert np.array_equal(np.sort(kept["expnum"]), np.sort(meas["expnum"]))
    est = out[out["estimated"]]
    assert len(est) > 0, "no exposures were estimated"
    assert np.all(np.isfinite(est["m50"])) and np.all((est["m50"] > 15) & (est["m50"] < 30))
    assert not out["extrapolated"][~out["estimated"]].any(), "measured rows flagged extrapolated"
    assert out["extrapolated"].sum() > 0, "no extrapolated exposures flagged"
    for b in BANDS:
        bsel = est["band"] == b
        if bsel.any():
            assert np.allclose(est["c"][bsel], coeffs[b][3])
    print("self-check passed")
