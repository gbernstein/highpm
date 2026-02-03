from astropy.table import Table, vstack

import glob

fl = glob.glob("./*movers.fits")

tbl_ls = []

for f in fl:
    tbl = Table.read(f)
    tbl_ls.append(tbl)

tbl = vstack(tbl_ls)

tbl.write("RetII_pm_table.fits", overwrite=True)
