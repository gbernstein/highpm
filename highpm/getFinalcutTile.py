#!/usr/bin/env python
# Acquire finalcut objects within a DES tile.
# for purpose of searching for high-pm sources.

### ??? Need to fix up the DB access so that it handles RA wrapping properly.

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

def getTileOf(ra,dec):
    # Find the tile with (ra,dec) in its unique bounds and return its outer ra/dec limits
    # First get the tile that it's in
    query = ("SELECT tilename FROM y6a1_coaddtile_geom WHERE {:f} BETWEEN uramin AND uramax " + \
            " AND {:f} BETWEEN udecmin AND udecmax").format(ra,dec)
    conn = ea.connect()
    tab = conn.query_to_pandas(query)
    if len(tab)<1:
        raise RuntimeError('No tiles found')
    elif len(tab)>1:
        raise RuntimeError('Multiple tiles found')
    return tab['TILENAME'][0]

def getTileBounds(tilename):
    # Find the tile with (ra,dec) in its unique bounds and return its outer ra/dec limits
    # First get the tile that it's in
    query = ("SELECT racmin,racmax,deccmin,deccmax,ra_cent,dec_cent " +\
             "FROM y6a1_coaddtile_geom WHERE tilename='{:s}'".format(tilename))
    conn = ea.connect()
    tab = conn.query_to_pandas(query)
    if len(tab)<1:
        raise RuntimeError('No tiles found')
    elif len(tab)>1:
        raise RuntimeError('Multiple tiles found')
    return tab['RACMIN'][0],tab['RACMAX'][0],tab['DECCMIN'][0],tab['DECCMAX'][0], \
           tab['RA_CENT'][0], tab['DEC_CENT'][0]
           
def getFinalcutCatalog(ramin,ramax,decmin,decmax):
    '''Obtain desired fields for all Finalcut detections in the given ra/dec range.
    '''
    query = ("SELECT c.expnum, i.ccdnum, c.band, c.alphawin_j2000, c.deltawin_j2000, " + \
             "c.flags, c.imaflags_iso, c.xwin_image, " +\
             "c.ywin_image, c.errawin_world, c.errbwin_world, c.errthetawin_j2000, " + \
             "c.flux_auto, c.fluxerr_auto " +\
             "FROM y6a1_finalcut_object c " +\
             "INNER JOIN y6a1_catalog i on i.filename=c.filename " +\
             "WHERE c.ra BETWEEN {:f} AND {:f} AND c.dec BETWEEN {:f} and " +\
             "{:f}").format(ramin,ramax,decmin,decmax)
    conn = ea.connect()
    tab = conn.query_to_pandas(query)
    return Table.from_pandas(tab)

def projectCatalog(cat, ra0, dec0, exposureTable):
    '''Take a finalcut catalog and add columns giving the
    gnomonic projected coordinates about the
    point (`ra0,dec0`).  Fill these using Pixmappy astrometric
    solutions.  Also use the local exposure catalog to enter
    columns for `MJD_MID`, and for the parallax coefficients
    that are the observatory position projected into the
    plane transverse to the `ra0,dec0` direction.

    The input catalog should have columns for
    `expnum, ccdnum, xwin_image` and `ywin_image`.
    '''

    # Check CAL_PATH to find exposure
    try:
        print('Reading files on path', os.environ['CAL_PATH'])
    except KeyError:
        print('Set CAL_PATH environment variable to include directories with astrometric solutions')

    # Read astrometric solution set from default files.
    # Optional argument inhibits use/creation of python pickle files.
    maps = pm.DESMaps()

    # Select the output coordinate system
    frame = pm.Gnomonic(ra0,dec0)
    # And assumed color for sources
    color = 0.6
    # And a rotation matrix into the system with axis along this
    cra,sra = math.cos(ra0 * np.pi / 180.), math.sin(ra0 * np.pi / 180.)
    cdec,sdec = math.cos(dec0 * np.pi / 180.), math.sin(dec0 * np.pi / 180.)
    '''
     cdec  0   sdec         cra  sra 0
       0   1    0          -sra  cra 0
     -sdec 0   cdec          0    0  1 '''
    R_bl = np.array([[-sra,        cra,      0.],
                     [-cra*sdec, -sra*sdec, cdec],
                     [ cra*cdec,  sra*cdec, sdec]])

    # Add columns to the (Astropy) table.
    zz = np.zeros_like(cat['XWIN_IMAGE'])
    cat.add_columns([zz,zz,zz,zz,zz],names=['XI','ETA','MJD','PAR_XI','PAR_ETA'])

    # Divide input into sets with matching ccd,exposure
    
    order = np.lexsort((np.array(cat['CCDNUM']),np.array(cat['EXPNUM'])))
    tmp1 = cat['CCDNUM'][order]
    tmp2 = cat['EXPNUM'][order]
    tmp = np.logical_or(tmp1[1:]!=tmp1[:-1], tmp2[1:]!=tmp2[:-1])
    starts = np.concatenate(([0],np.nonzero(tmp)[0]+1,[len(cat)]))

    expnum = -1
    print('starts',len(starts))
    for iStart in range(len(starts)-1):
        # These are indices of the detections using common WCS
        iUse = order[starts[iStart]:starts[iStart+1]]
        ccdnum = int(cat['CCDNUM'][iUse[0]])
        if cat['EXPNUM'][iUse[0]] != expnum:
            # New exposure. Get the mjd and parallax values for it
            expnum = cat['EXPNUM'][iUse[0]]
            iExp = np.where(exposureTable['expnum']==expnum)[0]
            if len(iExp)<1:
                # Can't use these points.  Flag them for deletion and continue.
                print('***WARNING: Exposure {:d} not in exposure table'.format(expnum))
                mjd = -1
            else:
                iExp = iExp[0]
                mjd = exposureTable['mjd_mid'][iExp]
                observatory = exposureTable['observatory'][iExp]
                # Get transverse components for parallax
                tmp = np.dot(R_bl, observatory)
                par_x = -tmp[0]
                par_y = -tmp[1]

        if mjd<0:
            # This is not useful data
            cat['MJD'][iUse] = mjd
            continue
        # Acquire the WCS for desired exposure, CCD combination
        wcs = maps.getDESWCS(expnum, ccdnum)
        # ValueError will be raised if there is no astrometric solution for this combination.
        wcs.reprojectTo(frame)

        # Use the wcs as a function object to map pixel
        # coordinates into projection-plane coordinates:
        xi, eta = wcs(cat['XWIN_IMAGE'][iUse], cat['YWIN_IMAGE'][iUse], color)
        cat['XI'][iUse] = xi
        cat['ETA'][iUse] = eta
        cat['MJD'][iUse] = mjd
        cat['PAR_XI'][iUse] = par_x
        cat['PAR_ETA'][iUse] = par_y

        # Let's also replace the RA and Dec
        ra, dec = wcs.toSky(cat['XWIN_IMAGE'][iUse], cat['YWIN_IMAGE'][iUse], color)
        cat['ALPHAWIN_J2000'][iUse] = ra
        cat['DELTAWIN_J2000'][iUse] = dec

    # Table is now updated
    # Get rid of the bad rows
    cat.remove_rows(cat['MJD']<0.)
    return

if __name__=='__main__':
    help = "Obtain finalcut catalog for use in finding proper motions.\n" + \
           "Usage:  getFinalCutTile tilename\n " +\
           " or     getFinalCutTile ra dec\n" +\
           "giving either the tilename or the RA, Dec (in degrees) of a\n" +\
           "point in the tile.  The `pixmappy` module must be installed\n" +\
           "and the environment variable `CAL_PATH` must give path to its\n" +\
           "data directory.  The `DES_EXPOSURES` environment variable\n" +\
           "should point to the table of exposure information.\n" +\
           "  The output catalog will be placed in <tilename>.finalcut.cat"
              
    if len(sys.argv)==2:
        if sys.argv[1]=='-h' or sys.argv[1]=='--help':
            print(help)
            sys.exit(1)
        tilename = sys.argv[1]
    elif len(sys.argv)==3:
        ra = float(sys.argv[1])
        dec = float(sys.argv[2])
        tilename = getTileOf(ra,dec)
    else:
        print(help)
        sys.exit(1)

    # Read exposure table
    exposureTable = pf.getdata(os.environ['DES_EXPOSURES'],1)

    # Get finalcut info from DB
    print("->Acquiring data for tile",tilename)
    rd = getTileBounds(tilename)
    ra0 = rd[4]
    dec0= rd[5]
    cat = getFinalcutCatalog(*rd[:4])

    print("->Doing Geometric calculations")
    projectCatalog(cat, ra0, dec0, exposureTable)

    catname = tilename+".finalcut.cat"
    cat.write(catname, format='fits', overwrite=True)
    # Add RA0, DEC0 to header
    with pf.open(catname,mode='update') as ff:
        ff[1].header['RA0']=ra0
        ff[1].header['DEC0']=dec0
    sys.exit(0)
    

