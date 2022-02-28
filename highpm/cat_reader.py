import astropy.io.fits as pf
from astropy.table import QTable
import numpy as np

def read_cat_header(filename):
    # Takes catalog produced by getFinalcutTile.py and returns the 
    # corresponding header 
    hdul = pf.open(filename)
    header = hdul[1].header
    hdul.close()
    return header

def read_cat_data(filename):
    # Takes catalog produced by getFinalcutTile.py and returns the 
    # corresponding data table
    hdul = pf.open(filename)
    cat_data = hdul[1].data
    hdul.close()
    cat = QTable(cat_data)
    return cat

def clean_cat(catname):
    # Removes detections with "FLAGS"!=0 and "IMAFLAGS!=0"
    cleancat = catname[np.logical_and(catname["FLAGS"]==0, \
                                      catname["IMAFLAGS_ISO"]==0)]
    return cleancat