#!/usr/bin/env python

from __future__ import print_function
import easyaccess as ea
import numpy as np
import astropy.io.fits as pf
from astropy.table import Table
import sys
import os
import argparse
import pixmappy as pm
import math
import subprocess
import glob
from getFinalcutTile import *
from cat_reader import *



def getCoaddOf(tilename):
    query = ("select FILENAME from Y6A1_COADD where TILENAME like '"
        + tilename +"' and BAND like 'Y' and FILETYPE like 'coadd_nobkg';")
    conn = ea.connect()
    tab = conn.query_to_pandas(query)
    if len(tab)<1:
        raise RuntimeError('No tiles found')
    elif len(tab)>1:
        raise RuntimeError('Multiple tiles found')
    return tab['FILENAME'][0]

def getPathOF(filename):
    query = ("select PATH from Y6A1_FILE_ARCHIVE_INFO where FILENAME like '"
        +filename+"';")
    conn = ea.connect()
    tab = conn.query_to_pandas(query)
    if len(tab)<1:
        raise RuntimeError('No tiles found')
    elif len(tab)>1:
        raise RuntimeError('Multiple tiles found')
    return tab['PATH'][0]

if __name__=='__main__':
    help = "Still need to write the help section"

    if len(sys.argv)==2:
        if sys.argv[1]=='-h' or sys.argv[1]=='--help':
            print(help)
            sys.exit(1)
        catname = sys.argv[1]
    else:
        print(help)
        sys.exit(1)

    cat = read_cat_data(catname)

    header = read_cat_header(catname)

    ra = [header['RA0']]
    dec = [header['DEC0']]

    # ra = cat['ra']
    # dec = cat['dec']

    tiles = []
    for i in range(len(ra)):
        try:
            tiles += [getTileOf(ra[i],dec[i])]
        except:
            tiles += [cat['tilename'][i]]

    tiles = np.unique(tiles)

    print(len(tiles))
    print(tiles)

    for i in tiles:
        coadd = getCoaddOf(i)
        fl = glob.glob('./'+coadd[:-10]+'*')
        if len(fl)==0:
            print(i)
            cmd = ['wget',
                '--no-check-certificate',
                '--http-user='+os.environ['USER'],
                '--http-password='+os.environ['PASSWORD'],
                'https://desar2.cosmology.illinois.edu/DESFiles/desarchive/' +\
                getPathOF(coadd) +\
                '/' +\
                coadd +\
                '.fz']
            process = subprocess.Popen(cmd)
            process.wait()

    sys.exit(0)

