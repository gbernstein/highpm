#!/usr/bin/env python

from __future__ import print_function
import sys
import os
import argparse
import glob
from matplotlib.pyplot import plot
from cat_reader import *
from object2detections import *
from object2plot import *
from getFinalcutTile import *
import subprocess

if __name__=='__main__':
    gaia_cat = read_cat_data(sys.argv[1])
    outdir=None
    if len(sys.argv)==3:
        outdir = sys.argv[2]
    for i in range(len(gaia_cat)):
        try:
            tileCoadd = getTileOf(gaia_cat['ra'][i],gaia_cat['dec'][i])
        except:
            tileCoadd = gaia_cat['tilename'][i]
        tilename = 'DES0043-3332' #gaia_cat['tilename'][i]
        obj_type = str(gaia_cat['mtype'][i]).strip()
        obj_id = gaia_cat['idx'][i]

        gaia_dir = '/home/vwetzell/Documents/DES_HighPM/DES0043-3332/'
        
        if obj_type=='slow':
            fl = gaia_dir+tilename+".finalcut_slow_movers.fits"
            cat = gaia_dir+'NEW_'+tilename+".finalcut.cat"
        if obj_type=='modest':
            fl = gaia_dir+tilename+".finalcut_modest_movers.fits"
            cat = gaia_dir+'pre_modest_'+tilename+".finalcut.cat"
        if obj_type=='fast':
            fl = gaia_dir+tilename+".finalcut_fast_movers.fits"
            cat = gaia_dir+'pre_fast_'+tilename+".finalcut.cat"
        if obj_type=='fast_checked':
            fl = gaia_dir+tilename+".finalcut_fast_checked_movers.fits"
            cat = gaia_dir+'NEW_'+tilename+".finalcut.cat"

        header = read_cat_header(tilename+".finalcut.cat")
        ra0 = header['RA0']
        dec0 = header['DEC0']
        
        if outdir==None:
            tbl = mktbl(cat,tilename,obj_type,obj_id,ra0,dec0)
        
            print(obj_type,tilename,tileCoadd)
            img = glob.glob('/home/vwetzell/Documents/DES_HighPM/GaiaTiles_400_18/coadds/'+tileCoadd+'*r_nobkg.fits.fz')
            if len(img)==1:
                img = img[0]
            
                plotter(tbl.data,tbl.header,img=img,tilename=tilename)
            else:
                plotter(tbl.data,tbl.header,tilename=tilename)

        else:
            tbl = mktbl(cat,tilename,obj_type,obj_id,ra0,dec0,outdir)
        
            print(obj_type,tilename,tileCoadd)
            img = glob.glob('/home/vwetzell/Documents/DES_HighPM/GaiaTiles_400_18/coadds/'+tileCoadd+'*r_nobkg.fits.fz')
            if len(img)==1:
                img = img[0]
            
                plotter(tbl.data,tbl.header,img=img,outdir=outdir,tilename=tilename)
            else:
                plotter(tbl.data,tbl.header,outdir=outdir,tilename=tilename)
    
    sys.exit(0)