import astropy.io.fits as pf
from astropy.table import QTable
from astropy.coordinates import SkyCoord
import sys

def read_cat_data(filename):
    # Takes catalog produced by getFinalcutTile.py and returns the 
    # corresponding data table
    hdul = pf.open(filename)
    cat_data = hdul[1].data
    hdul.close()
    cat = QTable(cat_data)
    return cat

if __name__=='__main__':
    help = "Creates .reg file from .fits table\n" +\
        "with columns 'ra' and 'dec'.\n" +\
        "Usage: fits2reg 'filename' radius(in arcsec)"

    if len(sys.argv)==2:
        if sys.argv[1]=='-h' or sys.argv[1]=='--help':
            print(help)
            sys.exit(1)
        catname = sys.argv[1]
    elif len(sys.argv)==3:
        filename = sys.argv[1]
        radius = sys.argv[2]
    else:
        print(help)
        sys.exit(1)

    cat = read_cat_data(filename)
    try:
        center = SkyCoord(cat['ra'], cat['dec'], unit='deg')
    except:
        center = SkyCoord(cat['RA'], cat['DEC'], unit='deg')
    ds9_str=''

    for i in range(len(center)):
        ds9_str += 'circle {}d {}d {}" ;'.format(center[i].ra.degree, \
            center[i].dec.degree,radius)
        
    with open(filename[:-5]+'.reg', 'w') as f:
        f.write(ds9_str)

    print('Regions file saved as '+filename[:-5]+'.reg')
    sys.exit(0)