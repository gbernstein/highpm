#!/usr/bin/env python
# Takes file output by object2detections.py
# Returns plot of detections associated with that object

from __future__ import print_function
import numpy as np 
import matplotlib.pyplot as plt
import sys
import os
import argparse
from matplotlib import colors
from matplotlib import gridspec
from astropy.io import fits
from astropy.table import QTable
from astropy.time import Time
from astropy.coordinates import solar_system_ephemeris, EarthLocation
from astropy.coordinates import get_body
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
import astropy.units as u

def plotter(data,header,img=None,outdir=None):
    year = 365.2425
    ref_date = 57388.0
    deg2mas = 3600000

    RA = header['RA']
    DEC = header['DEC']

    position = SkyCoord(u.rad*RA*np.pi/180,u.rad*DEC*np.pi/180, frame='icrs')

    t_ephem = np.linspace(ref_date-3*year,ref_date+3*year,1000)

    loc = EarthLocation.of_site('Cerro Tololo Interamerican Observatory')

    with solar_system_ephemeris.set('builtin'):
        sol = get_body('sun', Time(t_ephem,format='mjd'), loc) 

    t = (t_ephem-ref_date)/year

    def F_ra(ra_star,R_earth,ra_sun,dec_ecliptic):
        F_ra = (R_earth * np.sin(ra_sun) 
            * np.cos(ra_star) * np.cos(dec_ecliptic) +
            R_earth * np.sin(ra_star) * np.cos(ra_sun))
        return F_ra

    def F_dec(ra_star,R_earth,ra_sun,dec_ecliptic,dec_star):
        F_dec = R_earth * ((np.sin(dec_ecliptic) * np.cos(dec_star) 
            - np.cos(dec_ecliptic) * np.sin(ra_star) * np.sin(dec_star)) 
            * np.sin(ra_sun)
            - np.cos(ra_star) * np.sin(dec_star) * np.cos(ra_sun))
        return F_dec

    params = np.array((np.repeat(RA,1000),
                    sol.distance,
                    sol.ra*np.pi/180,
                    sol.dec*np.pi/180,
                    np.repeat(DEC,1000)))
    f_ra = np.zeros(1000)
    f_dec= np.zeros(1000)
    for i in range(1000):
        f_ra[i] = F_ra(params[0,i],params[1,i],params[2,i],params[3,i])
        f_dec[i]= F_dec(params[0,i],params[1,i],
            params[2,i],params[3,i],params[4,i])
        
    ra_line = RA + header['PMRA']/(deg2mas)*t \
        + f_ra * header['PARALLAX']/(deg2mas)
    dec_line = header['PMDEC']/(deg2mas)*t+DEC \
        + f_dec * header['PARALLAX']/(deg2mas)


    color_ls = ['tab:green',
            'tab:red',
            'tab:purple',
            'tab:brown',
            'tab:gray']
    bands = ['g','r','i','z','Y']


    ra_fit_err = np.sqrt(
        header['C_XX'] 
        + 2*header['C_XVX']*t 
        + header['C_VXVX']*t**2
        + 2*header['C_XPI']*f_ra 
        + 2*header['C_VXPI']*t*f_ra 
        + header['C_PIPI']*f_ra**2
        )*1000

    dec_fit_err = np.sqrt(
        header['C_YY'] 
        + 2*header['C_YVY']*t 
        + header['C_VYVY']*t**2
        + 2*header['C_YPI']*f_dec 
        + 2*header['C_VYPI']*t*f_dec 
        + header['C_PIPI']*f_dec**2
        )*1000

    if img!=None:
        fig = plt.figure(figsize=(15,18))

        ax0 = plt.subplot2grid(shape=(3, 2), loc=(0, 0), colspan=1)
        ax1 = plt.subplot2grid(shape=(3, 2), loc=(0, 1), colspan=1)
        

    if img==None:
        fig = plt.figure(figsize=(15,6))
        ax0 = plt.subplot2grid(shape=(1, 2), loc=(0, 0), colspan=1)
        ax1 = plt.subplot2grid(shape=(1, 2), loc=(0, 1), colspan=1)
    

    for i in range(len(bands)):
        ax0.scatter((data['MJD'][data['BAND']==bands[i]]-ref_date)/year,
            deg2mas*(data['ALPHAWIN_J2000'][data['BAND']==bands[i]]-RA),
            marker='x',s=100,c=color_ls[i],label=bands[i])
        ax0.errorbar((data['MJD'][data['BAND']==bands[i]]-ref_date)/year,
            deg2mas*(data['ALPHAWIN_J2000'][data['BAND']==bands[i]]-RA),
            yerr=deg2mas*(data['ERRAWIN_WORLD'][data['BAND']==bands[i]]),
            marker='None',ls='None',c=color_ls[i])
    ax0.plot(t,deg2mas*(ra_line-RA),c='k')
    ax0.scatter(np.array((data['MJD']-ref_date)/year)[data['clipped']==1],
        np.array(deg2mas*(data['ALPHAWIN_J2000']-RA))[data['clipped']==1],
        edgecolors='r',facecolors='none',s=160)

    ax0.fill_between(t,
                    deg2mas*(ra_line-ra_fit_err/deg2mas-RA),
                    deg2mas*(ra_line+ra_fit_err/deg2mas-RA),
                    color='grey',alpha=0.15)
    ax0.set_xlabel('Time [yr]',fontsize=15)
    ax0.set_ylabel('RA [mas]',fontsize=15)
    ax0.grid()
    ax0.legend(fontsize=15)



    for i in range(len(bands)):
        ax1.scatter((data['MJD'][data['BAND']==bands[i]]-ref_date)/year,
            deg2mas*(data['DELTAWIN_J2000'][data['BAND']==bands[i]]-DEC),
            marker='x',s=100,c=color_ls[i],label=bands[i])
        ax1.errorbar((data['MJD'][data['BAND']==bands[i]]-ref_date)/year,
            deg2mas*(data['DELTAWIN_J2000'][data['BAND']==bands[i]]-DEC),
            yerr=deg2mas*(data['ERRBWIN_WORLD'][data['BAND']==bands[i]]),
            marker='None',ls='None',c=color_ls[i])
    ax1.plot(t,deg2mas*(dec_line-DEC),c='k')
    ax1.scatter(np.array((data['MJD']-ref_date)/year)[data['clipped']==1],
        np.array(deg2mas*(data['DELTAWIN_J2000']-DEC))[data['clipped']==1],
        edgecolors='r',facecolors='none',s=160)
    ax1.fill_between(t,
                    deg2mas*(dec_line-dec_fit_err/deg2mas-DEC),
                    deg2mas*(dec_line+dec_fit_err/deg2mas-DEC),
                    color='grey',alpha=0.15)

    ax1.set_xlabel('Time [yr]',fontsize=15)
    ax1.set_ylabel('DEC [mas]',fontsize=15)
    ax1.grid()



    pm_fit_err = np.sqrt(
        header['C_XX'] + 
        2*t*(header['C_XVX'] - header['C_XVY']) + 
        t**2*(header['C_VXVX'] - 2*header['C_VXVY'] + header['C_VYVY']) + 
        2*(f_ra - f_dec)*(header['C_XPI'] + 
            header['C_VXPI'] - header['C_VYPI']) +
        header['C_PIPI']*(f_ra**2-2*f_ra*f_dec+f_dec**2)
    )

    if img!=None:

        hdu_img = fits.open(img)
        header_img = hdu_img[1].header
        data_img = hdu_img[1].data

        wcs = WCS(header_img)

        if header['PM']<1000:
            cutout = Cutout2D(data_img, position, (40,40),wcs=wcs)
        if header['PM']>=1000:
            size = u.Quantity((10*header['PM'], 10*header['PM']), 
                u.milliarcsecond)
            cutout = Cutout2D(data_img, position, size,wcs=wcs)

        wcs_cutout = cutout.wcs

        ax2 = plt.subplot2grid(shape=(3, 2), loc=(1, 0), colspan=2, rowspan=2,
                            projection=wcs_cutout)

        detections = np.array([data['ALPHAWIN_J2000'],
            data['DELTAWIN_J2000']]).T

        detection_pix = wcs_cutout.all_world2pix(detections,0)

        clipped = (data['clipped']==1)

        ra_err = np.hypot(data['ERRAWIN_WORLD'],data['TURBERRA'])
        dec_err = np.hypot(data['ERRBWIN_WORLD'],data['TURBERRB'])

        detection_err = np.array([ra_err,dec_err]).T
        detection_err_pix = \
            wcs_cutout.all_world2pix(detection_err+detections,0)
        detection_err_pix = abs(detection_err_pix-detection_pix)

        pm_line = np.array([ra_line,dec_line]).T
        pm_line_pix = wcs_cutout.all_world2pix(pm_line,0)

        pm_fit_up = np.array([ra_line,dec_line+pm_fit_err/3600]).T
        pm_fit_up_pix = wcs_cutout.all_world2pix(pm_fit_up,0)
        pm_fit_down = np.array([ra_line,dec_line-pm_fit_err/3600]).T
        pm_fit_down_pix = wcs_cutout.all_world2pix(pm_fit_down,0)

        ax2.imshow(cutout.data+100,norm=colors.LogNorm(),
            cmap='binary',origin='lower')

        ax2.errorbar(detection_pix[:,0],detection_pix[:,1],
                    xerr = detection_err_pix[:,0], 
                    yerr = detection_err_pix[:,1],
                    ls='none',c='w')
        scatter = ax2.scatter(detection_pix[:,0],detection_pix[:,1],
                    c=(data['MJD']-ref_date)/year,zorder=10)
        ax2.scatter(detection_pix[:,0][clipped],detection_pix[:,1][clipped],
                    edgecolors='r',facecolors='none',s=160)
        ax2.plot(pm_line_pix[:,0],pm_line_pix[:,1],c='w')
        ax2.plot(pm_fit_up_pix[:,0],pm_fit_up_pix[:,1],c='w',ls='--')
        ax2.plot(pm_fit_down_pix[:,0],pm_fit_down_pix[:,1],c='w',ls='--')

        fig.colorbar(scatter, 
            ax=ax2, 
            shrink=0.6).set_label('Time [yr]',fontsize=15)

        ax2.set_xlabel('RA',fontsize=15)
        ax2.set_ylabel('DEC',fontsize=15)

    name = str(header['MTYPE'])+'_'+str(header['IDX'])

    fig.suptitle(str(header['MTYPE'])+' '+str(header['IDX']),
        fontsize=20,y=0.91)
    if outdir==None:
        plt.savefig(name+'.png',bbox_inches='tight')
    if outdir!=None:
        plt.savefig('./'+outdir+'/'+name+'.png',bbox_inches='tight')

    plt.close()
    return

if __name__=="__main__":
    if len(sys.argv)==2:
        fl = sys.argv[1]
        hdu = fits.open(fl)
        header = hdu[1].header
        data = QTable(hdu[1].data)

        plotter(data,header)

        sys.exit(0)


    if len(sys.argv)==3:
        fl = sys.argv[1]
        hdu = fits.open(fl)
        header = hdu[1].header
        data = QTable(hdu[1].data)

        img = sys.argv[2]

        plotter(data,header,img)

        sys.exit(0)

    else:
        print('The format used for entries is... ')
        # print('python object2plot.py  \
        #     "object2detections.py output"')
        sys.exit(1)