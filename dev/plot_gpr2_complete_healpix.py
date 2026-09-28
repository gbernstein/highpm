"""Plot which Sculptor 8deg healpixels have all their exposures in GPR2/CAT (complete) vs not."""
import glob
import os
import re
import sys

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from matplotlib.patches import Polygon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from exposuresFromHealpixList import _radec_from_table  # noqa: E402
from highpm.detection_packaging import get_exposures_near_healpix  # noqa: E402

B = "/data8/shared/decampm/PMSculptor_8deg/"
NSIDE = 32
SCULPTOR_RA, SCULPTOR_DEC = 15.038750, -33.709000

hpx = np.load(B + "pmsculptor8deg_healpix.npy")
run = {int(x) for x in np.load(B + "pmsculptor8deg_exposures.npy")}
en, ra, dec = _radec_from_table(Table.read(
    "/home2/vwetzell/gitrepos/pixmappy/pixmappy/data/delveExposures.hdf5", path="__astropy_table__"))
new = set()
for b in "griz":
    for f in glob.glob(f"/data8/shared/decampm/GPR2/CAT/{b}/gprx_*_{b}.fits"):
        new.add(int(re.search(r"gprx_(\d+)_", f).group(1)))

fig, ax = plt.subplots(figsize=(8, 7))
n_full = 0
for h in hpx:
    need = {int(e) for e in get_exposures_near_healpix(int(h), ra, dec, en, nside=NSIDE)} & run
    frac = len(need & new) / len(need) if need else 1.0
    complete = frac == 1.0
    n_full += complete
    b = hp.boundaries(NSIDE, int(h), step=4, nest=False)
    lon, lat = hp.vec2ang(b.T, lonlat=True)
    lon = np.where(lon - SCULPTOR_RA > 180, lon - 360, lon)
    ax.add_patch(Polygon(np.c_[lon, lat], closed=True, fc="tab:green" if complete else "tab:red",
                         ec="k", lw=0.5, alpha=0.6))
    clon, clat = hp.pix2ang(NSIDE, int(h), lonlat=True)
    ax.text(clon, clat, f"{frac:.0%}" if not complete else "", ha="center", va="center", fontsize=5)
ax.scatter(SCULPTOR_RA, SCULPTOR_DEC, c="k", marker="*", s=80)
ax.add_patch(plt.Circle((SCULPTOR_RA, SCULPTOR_DEC), 8.0, fill=False, color="k", ls="--"))
ax.set_xlim(SCULPTOR_RA + 12, SCULPTOR_RA - 12)
ax.set_ylim(SCULPTOR_DEC - 10, SCULPTOR_DEC + 10)
ax.set_aspect("equal")
ax.set_xlabel("RA")
ax.set_ylabel("Dec")
ax.grid(alpha=0.3)
ax.set_title(f"Sculptor 8deg healpixels: {n_full}/{len(hpx)} fully in GPR2 (green); red = % present")
plt.savefig("gpr2_complete_healpix.png", dpi=200)
print("saved gpr2_complete_healpix.png", n_full, len(hpx))
