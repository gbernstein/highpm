"""
Build the per-exposure completeness table read by fake injection and real-star filtering.

Every exposure's curve comes from its own measured depth, via an S/N detection
threshold (see scripts/measure_exposure_depth.py and scripts/calibrate_snr_threshold.py):

    m50 = m1 - 2.5 log10(nu[band, tag]) - seeing_loss(fwhm, kernel[band, tag])
    k   = 1.814 / sqrt((1.814 / k0[band, tag])**2 + m1_ccd_std**2)
    c   = c[band, tag]

m1 is the exposure's median-CCD magnitude at S/N = 1 (zp and background noise from
its skim), so exposure time, read noise, sky, seeing and transparency need no model.
k broadens the calibrated single-CCD width by the exposure's CCD-to-CCD depth spread.
seeing_loss is the S/N a fixed Gaussian detection filter of FWHM `kernel` gives up
against the PSF-fit S/N behind m1 when the seeing (Gaussian, FWHM fwhm) differs from it:
for Gaussians the ratio is 2 kernel fwhm / (kernel^2 + fwhm^2), so nu is the threshold
of a matched filter. kernel is calibrated for DES and DELVE finalcut (0 = no correction).
An exposure with no measured fwhm is taken at FWHM_DEFAULT.
tag is the processing campaign (DES or DELVE finalcut), which sets the threshold.

Exposures with no zeropoint (no DELVE catalog stars in that band, e.g. near the Galactic
plane) or an uncalibrated tag/band (e.g. DECADE_F z) are left out; fake_detections.py then
gives them zero detection probability, with a warning.
"""

import os

import fitsio
import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
DEPTH = os.path.join(DATA, "delve.exposures.depth.fits")
THRESHOLD = os.path.join(DATA, "snr_threshold.fits")
INJECTION = os.path.join(DATA, "y6a1c.exposures.positions.fits")
OUT = os.path.join(DATA, "delve.exposures.completeness.fits")

LOGISTIC_SIGMA = np.pi / np.sqrt(3)  # std of a unit logistic, ~1.814
FWHM_DEFAULT = 1.6  # arcsec, typical seeing, for exposures without a measured fwhm


def seeing_loss(fwhm, kernel):
    """Magnitudes of detection depth lost by a Gaussian filter of FWHM kernel on a
    Gaussian PSF of FWHM fwhm, relative to PSF-fit S/N (0 where kernel <= 0)."""
    fwhm = np.where(np.isfinite(fwhm), fwhm, FWHM_DEFAULT)
    kernel = np.asarray(kernel, dtype=float)
    K = np.where(kernel > 0, kernel, fwhm)
    return 2.5 * np.log10((K ** 2 + fwhm ** 2) / (2 * K * fwhm))


def build():
    depth = fitsio.read(DEPTH)
    thr = fitsio.read(THRESHOLD)
    cal = {(r["band"], r["tag"]): r for r in thr}

    has_depth = np.isfinite(depth["m1"])
    has_cal = np.array([(b, t) in cal for b, t in zip(depth["band"], depth["tag"])])
    d = depth[has_depth & has_cal]

    nu = np.array([cal[(b, t)]["nu"] for b, t in zip(d["band"], d["tag"])])
    k0 = np.array([cal[(b, t)]["k"] for b, t in zip(d["band"], d["tag"])])
    c = np.array([cal[(b, t)]["c"] for b, t in zip(d["band"], d["tag"])])
    kernel = np.array([cal[(b, t)]["kernel"] for b, t in zip(d["band"], d["tag"])])
    spread = np.nan_to_num(d["m1_ccd_std"].astype(float))

    out = np.empty(len(d), dtype=[
        ("expnum", "i4"), ("band", "U1"), ("m50", "f8"), ("k", "f8"), ("c", "f8"),
        ("tag", "U8"), ("exptime", "f4"), ("m1", "f8"), ("m1_ccd_std", "f4"), ("fwhm", "f4"),
    ])
    out["expnum"], out["band"], out["tag"] = d["expnum"], d["band"], d["tag"]
    out["exptime"], out["m1"], out["m1_ccd_std"] = d["exptime"], d["m1"], d["m1_ccd_std"]
    out["fwhm"] = d["fwhm"]
    out["m50"] = d["m1"] - 2.5 * np.log10(nu) - seeing_loss(d["fwhm"].astype(float), kernel)
    out["k"] = LOGISTIC_SIGMA / np.sqrt((LOGISTIC_SIGMA / k0) ** 2 + spread ** 2)
    out["c"] = c
    skipped = dict(no_depth=int((~has_depth).sum()), uncalibrated=int((has_depth & ~has_cal).sum()))
    return out, thr, skipped


def injection_check(out):
    """S/N-model m50 vs. the injection-measured m50 of DES 90 s exposures, per band."""
    inj = fitsio.read(INJECTION, ext=1, columns=["expnum", "band", "m50"])
    _, i_out, i_inj = np.intersect1d(out["expnum"], inj["expnum"], return_indices=True)
    print(f"  cross-check vs. {len(i_out)} injection-measured exposures (model - injection m50):")
    for b in "griz":
        s = out["band"][i_out] == b
        r = out["m50"][i_out][s] - inj["m50"][i_inj][s]
        mad = 1.4826 * np.median(np.abs(r - np.median(r)))
        print(f"    {b}: median {np.median(r):+.3f}, robust std {mad:.3f}, n={s.sum()}")


def main():
    out, thr, skipped = build()
    fitsio.write(OUT, out, clobber=True)
    print(f"wrote {OUT}: {len(out)} exposures "
          f"(skipped {skipped['no_depth']} without a depth, {skipped['uncalibrated']} with an uncalibrated tag)")
    for r in thr:
        print(f"  {r['band']} {r['tag']:6s} nu={r['nu']:.2f} k0={r['k']:.2f} c={r['c']:.3f} "
              f"kernel={r['kernel']:.2f}\"")
    injection_check(out)
    return out


if __name__ == "__main__":
    out = main()

    # ponytail: one self-check that fails if the join or model breaks.
    assert len(out) > 0 and len(np.unique(out["expnum"])) == len(out), "empty or duplicated rows"
    assert np.all(np.isfinite(out["m50"])) and np.all((out["m50"] > 15) & (out["m50"] < 30))
    assert np.all((out["k"] > 0) & (out["c"] > 0) & (out["c"] <= 1))
    print("self-check passed")
