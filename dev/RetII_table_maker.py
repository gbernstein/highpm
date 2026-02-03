from astropy.table import Table, vstack
import sys
import glob


def RetII_table_maker(directory):

    fl = glob.glob(directory + "/*.fits")

    tbl_ls = []

    for f in fl:

        tbl = Table.read(f)

        tbl_ls.append(tbl)

    return vstack(tbl_ls)


if __name__ == "__main__":

    directory = sys.argv[1]

    tbl = RetII_table_maker(directory)

    tbl.write("RetII_table.fits", overwrite=True)

    print("Table saved as RetII_table.fits")

    print("Done")

    sys.exit(0)
