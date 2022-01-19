#!/usr/bin/env python
# Takes tile title, object type, and object id
# Returns subset of finalcut catalog associated with that object

from __future__ import print_function
import astropy.io.fits as pf
import numpy as np
import sys
import os
import argparse
from astropy.table import Table

if __name__=="__main__":
    if len(sys.argv)==4:
        tilename  = sys.argv[1]
        obj_type = sys.argv[2]
        obj_id = int(sys.argv[3])

        if obj_type not in ['slow','modest','fast']:
            print("Supported object types are: 'slow', 'modest', 'fast'")
            sys.exit(1)

        if obj_type=='slow':
            cat = tilename+".finalcut.cat"
        if obj_type=='modest':
            cat = 'pre_modest_'+tilename+".finalcut.cat"
        if obj_type=='fast':
            cat = 'pre_fast_'+tilename+".finalcut.cat"

        hdul_cat = pf.open(cat)
        data_cat = hdul_cat[1].data
        hdul_cat.close()

        objects = tilename+".finalcut_"+obj_type+"_movers.fits"

        hdul_obj = pf.open(objects)
        data_obj = hdul_obj[1].data
        hdul_obj.close()

        detections = tilename+".finalcut_"+obj_type+"_detections.fits"

        hdul_det = pf.open(detections)
        data_det = hdul_det[1].data
        hdul_det.close()

        obj_info = data_obj[obj_id]

        hdr = pf.Header()
        hdr['idx'] = obj_info['idx']
        hdr['mtype'] = obj_info['mtype']
        hdr['ra'] = obj_info['ra']
        hdr['raerr'] = obj_info['ra_err']
        hdr['dec'] = obj_info['dec']
        hdr['decerr'] = obj_info['dec_err']
        hdr['pm'] = obj_info['pm']
        hdr['pmra'] = obj_info['pmra']
        hdr['pmra_err'] = obj_info['pmra_err']
        hdr['pmdec'] = obj_info['pmdec']
        hdr['pmdecerr'] = obj_info['pmdec_err']
        hdr['parallax'] = obj_info['parallax']
        hdr['paraerr'] = obj_info['parallax_err']
        hdr['chisqTot'] = obj_info['chisqTotal']
        hdr['dof'] = obj_info['dof']

        empty_primary = pf.PrimaryHDU(header=hdr)

        obj_detections = data_det['detections'][np.argwhere(data_det['idx'] \
            ==obj_id)] 

        obj_clipped = data_det['clipped'][np.argwhere(data_det['idx']==obj_id)] 

        obj_det_data = data_cat[obj_detections]

        tbl = Table(obj_det_data)

        tbl.add_column(obj_clipped,name='clipped')

        table_hdu = pf.BinTableHDU(data=tbl,header=hdr)
        
        table_hdu.writeto(tilename+'_'+obj_type+'_'+str(obj_id) \
            +'_finalcut.cat',overwrite=True)

        sys.exit(0)

    else:
        print('The format used for entries is... ')
        print('python object2detections.py  \
            "TILE_TITLE" "OBJECT_TYPE" OBJECT_ID')
        sys.exit(1)

    

    
    


