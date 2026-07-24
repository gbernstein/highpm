import numpy as np
import fitsio
import matplotlib.pyplot as plt
import glob

fl = glob.glob("/home2/vwetzell/ProperMotion_v3/PositionCorrectedExposureCatalog/cat/*")

tmp_fits_ls = []
for f in fl:
    tmp_fits = fitsio.read(f,columns=["BEST_RA","BEST_DEC"])
    tmp_fits_ls += [tmp_fits]


all_fits = np.concatenate(tmp_fits_ls)


circle = plt.Circle((15.038750,-33.709000), radius=5, fill=False, color='r', linewidth=2)

fig,ax = plt.subplots()
plt.gca().invert_xaxis()
plt.scatter(
        all_fits["BEST_RA"],
        all_fits["BEST_DEC"],
        s=1,
        alpha=0.5,
)
plt.scatter(15.038750,-33.709000,c='r')
ax.add_patch(circle)
plt.grid()
plt.savefig("sculptor_detections.png")
plt.close()
