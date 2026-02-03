from astropy.table import Table

tbl = Table.read("~/Documents/proper_motions/RetII/RetII_table.fits")

print(tbl.colnames)
print(len(tbl))
