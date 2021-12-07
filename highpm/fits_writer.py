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
            'ra_err',
            'dec',
            'dec_err',
            'pm',
            'pmra',
            'pmra_err',
            'pmdec',
            'pmdec_err',
            'parallax',
            'parallax_err',
        #     'alpha',
            'chisqTotal',
        #     't',
            'dof',
            'nClip',
        #     'members',
        #     'clipped'
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

        ra_err = (np.sqrt(cov[:,0,0])*u.arcsec).to(u.deg)
        dec_err = (np.sqrt(cov[:,1,1])*u.arcsec).to(u.deg)

        pmra    = (p_fits[:,2]*u.arcsec/u.year).to(u.mas/u.year)
        pmra_err = (np.sqrt(cov[:,2,2])*u.arcsec/u.year).to(u.mas/u.year)
        pmdec   = (p_fits[:,3]*u.arcsec/u.year).to(u.mas/u.year)
        pmdec_err = (np.sqrt(cov[:,3,3])*u.arcsec/u.year).to(u.mas/u.year)
        pm = np.hypot(pmra,pmdec)
        parallax = p_fits[:,4]
        parallax_err = np.sqrt(cov[:,4,4])

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
            ra_err,
            dec,
            dec_err,
            pm,
            pmra,
            pmra_err,
            pmdec,
            pmdec_err,
            parallax,
            parallax_err,
        #     alpha,
            chisqTotal,
        #     t,
            dof,
            nClip,
        #     members,
        #     clipped
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