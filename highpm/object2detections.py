#!/usr/bin/env python
# Takes tile title, object type, and object id
# Returns subset of finalcut catalog associated with that object

from __future__ import print_function
from turtle import clone
import astropy.io.fits as pf
import numpy as np
import sys
import os
import argparse
from astropy.table import Table,QTable
from getFinalcutTile import *
from cat_reader import *


def mktbl(cat,tilename,obj_type,obj_id,ra0,dec0,outdir=None):
    """
    Table maker.

    Parameters
    ----------
    cat : 
    tilename : 
    obj_type : 
    obj_id : 
    outdir : 

    Yeilds
    ------
    ./<outdir>/<tilename>.finalcut_<obj_type>_movers.fits
    ./<outdir>/<tilename>.finalcut_<obj_type>_detections.fits

    """


    data_cat = Table.read(cat)

    objects = tilename+".finalcut_"+obj_type+"_movers.fits"

    data_obj = Table.read(objects)

    detections = tilename+".finalcut_"+obj_type+"_detections.fits"

    data_det = Table.read(detections)

    obj_info = data_obj[obj_id]

    hdr = pf.Header()
    hdr['idx'] = obj_info['idx']
    hdr['mtype'] = obj_info['mtype']
    hdr['ra'] = obj_info['ra']
    hdr['dec'] = obj_info['dec']
    hdr['pm'] = obj_info['pm']
    hdr['pmra'] = obj_info['pmra']
    hdr['pmdec'] = obj_info['pmdec']
    hdr['parallax'] = obj_info['parallax']

    cov_list = [
        'c_xx',
        'c_yy',
        'c_vxvx',
        'c_vyvy',
        'c_pipi',

        'c_xy',
        'c_xvx',
        'c_xvy',
        'c_xpi',

        'c_yvx',
        'c_yvy',
        'c_ypi',

        'c_vxvy',
        'c_vxpi',

        'c_vypi'
    ]

    for i in cov_list:
        hdr[i] = obj_info[i]

    hdr['chisqTot'] = obj_info['chisqTotal']
    hdr['dof'] = obj_info['dof']

    obj_detections = data_det['detections'][np.array(data_det['idx']) \
        ==obj_id] 

    obj_clipped = data_det['clipped'][np.array(data_det['idx'])==obj_id] 

    tbl = data_cat[obj_detections]

    tbl.add_column(obj_clipped,name='clipped')

    g_flux = tbl[tbl['BAND']=='g']['FLUX_PSF']
    g_mag = np.mean(tbl[tbl['BAND']=='g']['ZEROPOINT'] \
        -2.5*np.log10(g_flux))
    i_flux = tbl[tbl['BAND']=='i']['FLUX_PSF']
    i_mag = np.mean(tbl[tbl['BAND']=='i']['ZEROPOINT'] \
        -2.5*np.log10(i_flux))

    if len(tbl[tbl['BAND']=='g']['FLUX_PSF'])!=0 \
        and len(tbl[tbl['BAND']=='i']['FLUX_PSF'])!=0:
        color = g_mag - i_mag

        exposureTable = pf.getdata(os.environ['DES_EXPOSURES'],1)

        tbl.remove_columns(['XI','ETA','MJD','PAR_XI','PAR_ETA'])

        projectCatalog(tbl, \
            ra0,dec0, \
            exposureTable,color)


    ls_format = ['K','A','K']+2*['D']+2*['K']+7*['D']+['41A']+7*['D']+['K']+2*['D']+['K']+5*['D']
    ls_col = []
    for i in range(len(tbl.colnames)):
        ls_col += [pf.Column(name=tbl.colnames[i],format=ls_format[i],array=np.array(tbl[tbl.colnames[i]]))]

    table_hdu = pf.BinTableHDU.from_columns(ls_col,header=hdr)

    
    if outdir==None:
        table_hdu.writeto(tilename+'_'+tilename+'_'+obj_type+'_'+str(obj_id) \
            +'_finalcut.cat',overwrite=True)
    if outdir!=None:
        table_hdu.writeto('./'+outdir+'/'\
            +tilename+'_'+obj_type+'_'+str(obj_id) \
            +'_finalcut.cat',overwrite=True)

    return table_hdu

if __name__=="__main__":
    if len(sys.argv)>=4:
        tilename  = sys.argv[1]
        obj_type = sys.argv[2]
        obj_id = int(sys.argv[3])

        if obj_type not in ['slow','modest','fast','fast_checked']:
            print("Supported object types are: 'slow', 'modest', 'fast'")
            sys.exit(1)

        if obj_type=='slow' or obj_type=='fast_checked':
            cat = 'NEW_'+tilename+".fake_finalcut.cat"
        if obj_type=='modest':
            cat = 'pre_modest_'+tilename+".finalcut.cat"
        if obj_type=='fast':
            cat = 'pre_fast_'+tilename+".finalcut.cat"

        ramin,ramax,decmin,decmax,ra0,dec0 = getTileBounds(tilename)

        if len(sys.argv)==5:
            outdir = sys.argv[4]
            mktbl(cat,tilename,obj_type,obj_id,ra0,dec0,outdir=outdir)
        else:
            mktbl(cat,tilename,obj_type,obj_id,ra0,dec0)

        sys.exit(0)

    else:
        print('The format used for entries is... ')
        print('python object2detections.py  \
            "TILE_TITLE" "OBJECT_TYPE" OBJECT_ID')
        sys.exit(1)

    

    
    


