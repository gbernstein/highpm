#!/usr/bin/env python
# Takes tile title, object type, and object id
# Returns subset of finalcut catalog associated with that object

from __future__ import print_function
import astropy.io.fits as pf
import numpy as np
import sys
import os
import argparse
from astropy.table import Table,QTable


def mktbl(cat,tilename,obj_type,obj_id,outdir=None):
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


    empty_primary = pf.PrimaryHDU(header=hdr)

    obj_detections = data_det['detections'][np.array(data_det['idx']) \
        ==obj_id] 

    obj_clipped = data_det['clipped'][np.array(data_det['idx'])==obj_id] 

    tbl = data_cat[obj_detections]

    tbl.add_column(obj_clipped,name='clipped')

    table_hdu = pf.BinTableHDU(data=tbl,header=hdr)
    
    if outdir==None:
        table_hdu.writeto(tilename+'_'+obj_type+'_'+str(obj_id) \
            +'_finalcut.cat',overwrite=True)
    if outdir!=None:
        table_hdu.writeto('./'+outdir+'/'\
            +obj_type+'_'+str(obj_id) \
            +'_finalcut.cat',overwrite=True)

    return

if __name__=="__main__":
    if len(sys.argv)==4:
        tilename  = sys.argv[1]
        obj_type = sys.argv[2]
        obj_id = int(sys.argv[3])

        if obj_type not in ['slow','modest','fast','fast_checked']:
            print("Supported object types are: 'slow', 'modest', 'fast'")
            sys.exit(1)

        if obj_type=='slow' or obj_type=='fast_checked':
            cat = 'NEW_'+tilename+".finalcut.cat"
        if obj_type=='modest':
            cat = 'pre_modest_'+tilename+".finalcut.cat"
        if obj_type=='fast':
            cat = 'pre_fast_'+tilename+".finalcut.cat"

        mktbl(cat,tilename,obj_type,obj_id)

        sys.exit(0)

    else:
        print('The format used for entries is... ')
        print('python object2detections.py  \
            "TILE_TITLE" "OBJECT_TYPE" OBJECT_ID')
        sys.exit(1)

    

    
    


