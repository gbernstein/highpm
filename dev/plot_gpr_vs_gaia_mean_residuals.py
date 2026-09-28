"""Save per-exposure (expnum, mean GPR-Gaia residual) as .npy and plot the residuals with a linear fit.

Reads the cache written by plot_gpr_vs_gaia_per_exposure.py (run that first)."""
import numpy as np
import matplotlib.pyplot as plt
from plot_gpr_vs_gaia_per_exposure import fit_line, GAIA_REF_MJD, YEAR

tab = np.load("gpr_vs_gaia_per_exposure.npy")
tab = tab[np.argsort(tab["mjd"])]
out = np.zeros(len(tab), dtype=[("expnum", "i8"), ("mjd", "f8"), ("dra", "f8"), ("ddec", "f8"),
                                ("dra_err", "f8"), ("ddec_err", "f8")])
for k in out.dtype.names:
    out[k] = tab[k]
np.save("gpr_vs_gaia_mean_residuals.npy", out)

tyr = 2016.0 + (tab["mjd"] - GAIA_REF_MJD) / YEAR
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, (key, label) in zip(axes, (("dra", "RA*cos(dec)"), ("ddec", "Dec"))):
    y, err = tab[key], np.maximum(tab[key + "_err"], 1e-3)
    p, perr, _ = fit_line(tyr - 2016.0, y, err)
    ax.plot(tyr, y, ".", color="darkorange", alpha=0.5, ms=4)
    xx = np.array([tyr.min(), tyr.max()])
    ax.plot(xx, p[0] + p[1] * (xx - 2016.0), color="firebrick", lw=2, label=f"slope {p[1]:+.3f} +/- {perr[1]:.3f} mas/yr")
    ax.set_xlabel("year"); ax.set_ylabel(f"mean GPR - Gaia, {label} (mas)")
    ax.legend(); ax.grid()
plt.tight_layout()
plt.savefig("gpr_vs_gaia_mean_residuals.png", dpi=200)
