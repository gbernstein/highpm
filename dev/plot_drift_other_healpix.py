"""Per-exposure GPR - Gaia residual vs year for the healpixels saved by gpr_drift_other_healpix.py (ref stars)."""
import sys
import matplotlib.pyplot as plt
import healpy as hp
import numpy as np
from plot_gpr_vs_gaia_per_exposure import fit_line, GAIA_REF_MJD, YEAR

hps = [int(h) for h in sys.argv[1:]] or [4556, 3696, 4739, 3990, 4458, 4535]
fig, axes = plt.subplots(2, len(hps) + 1, figsize=(3.6 * (len(hps) + 1), 7), sharex=True, sharey="row", squeeze=False)
SCL = np.load("/data8/shared/decampm/drift/gpr_vs_gaia_per_exposure_ref.npy")
sculptor = np.column_stack([SCL["mjd"], SCL["n"], SCL["dra"], SCL["dra_err"], SCL["ddec"], SCL["ddec_err"]])
for c, h in enumerate(hps + ["Sculptor"]):
    R = sculptor if h == "Sculptor" else np.load(f"drift_hp{h}.npy")
    yr = 2016 + (R[:, 0] - GAIA_REF_MJD) / YEAR
    for r, (iy, lab) in enumerate(((2, "RA*cos(dec)"), (4, "Dec"))):
        ax = axes[r, c]
        y, e = R[:, iy], np.maximum(R[:, iy + 1], 1e-3)
        g = np.isfinite(y) & np.isfinite(e)
        p, pe, _ = fit_line(yr[g] - 2016, y[g], e[g])
        ax.plot(yr[g], y[g], ".", color="tab:blue" if h == "Sculptor" else "firebrick", alpha=0.4, ms=4)
        xx = np.array([yr[g].min(), yr[g].max()])
        ax.plot(xx, p[0] + p[1] * (xx - 2016), "k", lw=2, label=f"slope {p[1]:+.3f}±{pe[1]:.3f} mas/yr")
        ax.axhline(0, color="gray", ls="--", lw=1)
        ax.set_ylim(-3, 3.5)
        ax.grid()
        ax.legend(fontsize=8, loc="upper left")
        if c == 0:
            ax.set_ylabel(f"GPR - Gaia, {lab} (mas)")
        if r == 0:
            if h == "Sculptor":
                cen = "RA 15.04, Dec -33.71"
            else:
                ra_c, dec_c = hp.pix2ang(32, h, lonlat=True)
                cen = f"RA {ra_c:.2f}, Dec {dec_c:+.2f}"
            name = "Sculptor 8deg, pooled" if h == "Sculptor" else f"healpix {h}"
            ax.set_title(f"{name} (n_exp={g.sum()})\n{cen}", fontsize=10)
        if r == 1:
            ax.set_xlabel("year")
fig.suptitle("GPR reference stars vs Gaia, per-exposure median, complete healpixels away from Sculptor")
plt.tight_layout()
plt.savefig("drift_other_healpix.png", dpi=150)
