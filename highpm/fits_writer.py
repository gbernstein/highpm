import numpy as np
import astropy.units as u
from astropy.table import vstack,QTable
from cat_reader import *
from gnomonic_plate2sky import *


def output_fits(pm_arr,filename,mtype,fittype='fit5d',outputname=None):
    # Writes a fits table containing data from the proper motion fitting

    if fittype!='fit5d':
        raise RuntimeError('This fitter is not yet supported')

    if fittype=='fit5d':

        column_names = [
            'idx',
            'mtype',
            'ra',
            'dec',
            'pm',
            'pmra',
            'pmdec',
            'parallax',
        #     'alpha',
            'chisqTotal',
        #     't',
            'dof',
            'nClip',
        #     'members',
        #     'clipped'
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

        idx   = range(len(pm_arr))
        ls_mtype = [mtype]*len(pm_arr)

        p_fits = np.vstack(pm_arr[:,0])
        cov = np.array([np.linalg.inv(i) for i in pm_arr[:,1]])

        xi = np.array((p_fits[:,0]*u.arcsec).to(u.deg))
        eta = np.array((p_fits[:,1]*u.arcsec).to(u.deg))

        header = read_cat_header(filename)

        ra0 = header['RA0']
        dec0 = header['DEC0']

        ra,dec = gnomonic_plate2sky(xi,eta,ra0,dec0)*u.deg
        pmra   = (p_fits[:,2]*u.arcsec/u.year).to(u.mas/u.year)
        pmdec  = (p_fits[:,3]*u.arcsec/u.year).to(u.mas/u.year)
        pm = np.hypot(pmra,pmdec)
        parallax = p_fits[:,4]



        c_xx   = cov[:,0,0]
        c_yy   = cov[:,1,1]
        c_vxvx = cov[:,2,2]
        c_vyvy = cov[:,3,3]
        c_pipi = cov[:,4,4]

        c_xy   = cov[:,0,1]
        c_xvx  = cov[:,0,2]
        c_xvy  = cov[:,0,3]
        c_xpi  = cov[:,0,4]

        c_yvx  = cov[:,1,2]
        c_yvy  = cov[:,1,3]
        c_ypi  = cov[:,1,4]

        c_vxvy = cov[:,2,3]
        c_vxpi = cov[:,2,4]

        c_vypi = cov[:,3,4]


        # ra_err = (np.sqrt(cov[:,0,0])*u.arcsec).to(u.deg)
        # dec_err = (np.sqrt(cov[:,1,1])*u.arcsec).to(u.deg)

        # pmra    = (p_fits[:,2]*u.arcsec/u.year).to(u.mas/u.year)
        # pmra_err = (np.sqrt(cov[:,2,2])*u.arcsec/u.year).to(u.mas/u.year)
        # pmdec   = (p_fits[:,3]*u.arcsec/u.year).to(u.mas/u.year)
        # pmdec_err = (np.sqrt(cov[:,3,3])*u.arcsec/u.year).to(u.mas/u.year)
        # pm = np.hypot(pmra,pmdec)
        # parallax = p_fits[:,4]
        # parallax_err = np.sqrt(cov[:,4,4])

        alpha = pm_arr[:,1]
        chisqTotal = np.array(pm_arr[:,2],dtype=np.float64)
        t = pm_arr[:,3]
        dof = np.array(pm_arr[:,4],dtype=np.float64)
        nClip = np.array(pm_arr[:,5],dtype=np.float64)
        members = pm_arr[:,6]
        clipped = pm_arr[:,7]

        data = [
            idx,
            ls_mtype,
            ra,
            dec,
            pm,
            pmra,
            pmdec,
            parallax,
        #     alpha,
            chisqTotal,
        #     t,
            dof,
            nClip,
        #     members,
        #     clipped
            c_xx,
            c_yy,
            c_vxvx,
            c_vyvy,
            c_pipi,

            c_xy,
            c_xvx,
            c_xvy,
            c_xpi,

            c_yvx,
            c_yvy,
            c_ypi,

            c_vxvy,
            c_vxpi,

            c_vypi
        ]

        tbl = QTable(names=column_names,data=data)
        
        detection_column_names = [
            'idx',
            'detections',
            'clipped'
        ]

        detection_tbl = QTable(names=detection_column_names,
            dtype=['i4','i4','i4'])
        for i in idx:
            clipbool = len(members[i])*[False] + len(clipped[i])*[True]
            detections = np.hstack((members[i],clipped[i]))
            temp_tbl = QTable(data=[[i]*len(detections),detections,clipbool], 
                names=detection_column_names,
                dtype=['i4','i4','i4'])
            
            detection_tbl = vstack([detection_tbl,temp_tbl])

        if outputname==None:
            tbl.write(filename[:-4]+'_'+mtype+'_movers.fits',overwrite=True)
            detection_tbl.write(filename[:-4]+'_'+mtype+'_detections.fits', \
                overwrite=True)
        else:
            tbl.write(outputname+'.fits',overwrite=True)
            detection_tbl.write(outputname+'_detections.fits', \
                overwrite=True)
    return