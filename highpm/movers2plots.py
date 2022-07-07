#!/usr/bin/env python

from __future__ import print_function
import sys
import os
import argparse
from cat_reader import *
from object2detections import *
from object2plot import *
import subprocess
from getFinalcutTile import *

if __name__=='__main__':
    tilename  = sys.argv[1]
    obj_type = sys.argv[2]
    try:
        os.system('rm -r ./'+tilename+'_'+obj_type)
        os.system('mkdir ./'+tilename+'_'+obj_type)
    except:
        os.system('mkdir ./'+tilename+'_'+obj_type)

    if obj_type not in ['slow','modest','fast','fast_checked']:
            print("Supported object types are: 'slow', 'modest', 'fast'")
            sys.exit(1)
    if obj_type=='slow':
        fl = tilename+".finalcut_slow_movers.fits"
        cat = 'NEW_'+tilename+".finalcut.cat"
    if obj_type=='modest':
        fl = tilename+".finalcut_modest_movers.fits"
        cat = 'pre_modest_'+tilename+".finalcut.cat"
    if obj_type=='fast':
        fl = tilename+".finalcut_fast_movers.fits"
        cat = 'pre_fast_'+tilename+".finalcut.cat"
    if obj_type=='fast_checked':
        fl = tilename+".finalcut_fast_checked_movers.fits"
        cat = 'NEW_'+tilename+".finalcut.cat"

    movers = read_cat_data(fl)

    if len(sys.argv)==5:
        movers=movers[movers['pm']>int(sys.argv[4])]

    img = sys.argv[3]

    for i in movers['idx']:

        ramin,ramax,decmin,decmax,ra0,dec0 = getTileBounds(tilename)
        
        mktbl(cat,tilename,obj_type,i,ra0=ra0,dec0=dec0,outdir=tilename+'_'+obj_type)

        fl_i = './'+tilename+'_'+obj_type+'/'\
            +tilename+'_'+obj_type+'_'+str(i)+'_finalcut.cat'

        data = read_cat_data(fl_i)
        header = read_cat_header(fl_i)

        plotter(data,header,img,outdir=tilename+'_'+obj_type)

    sys.exit(0)
