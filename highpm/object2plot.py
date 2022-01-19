#!/usr/bin/env python
# Takes file output by object2detections.py
# Returns plot of detections associated with that object

from __future__ import print_function
import numpy as np 
import matplotlib.pyplot as plt
import sys
import os
import argparse
from astropy.io import fits
from astropy.table import QTable

if __name__=="__main__":
    if len(sys.argv)==2:
        fl = sys.argv[1]
        hdu = fits.open(fl)
        header = hdu[1].header
        data = QTable(hdu[1].data)

        year = 365.2425
        ref_date = 57388.0

        RA = header['RA']
        DEC = header['DEC']

        t = np.linspace(-3,+3,1000)

        ra_line = header['PMRA']*t/(3600*year)+RA
        dec_line = header['PMDEC']*t/(3600*year)+DEC

        n = len(data['ALPHAWIN_J2000'])-np.sum(data['clipped'])

        s_yx = np.sqrt(np.sum((data['DELTAWIN_J2000']-DEC)**2)/(n-2)) 
        s = s_yx * \
            np.sqrt((1/n)+(ra_line-RA)**2
                /np.sum((data['ALPHAWIN_J2000']-RA)**2))


        plt.figure(figsize=(10,8))
        plt.axis('equal')
        plt.gca().invert_xaxis()
        plt.scatter(data['ALPHAWIN_J2000'],
                    data['DELTAWIN_J2000'],
                    c=(data['MJD']-ref_date)/year,marker='x',s=100)
        plt.scatter(ra_line,dec_line,c=t,s=1)
        plt.colorbar().set_label('MJD [yr]',fontsize=15)
        plt.scatter(data['ALPHAWIN_J2000'][data['clipped']],
                    data['DELTAWIN_J2000'][data['clipped']],
                    edgecolors='r',facecolors='none',s=160)
        plt.fill_between(ra_line,dec_line-s,dec_line+s,color='grey',alpha=0.15)

        plt.xlabel('RA [deg]',fontsize=15)
        plt.ylabel('DEC [deg]',fontsize=15)
        plt.grid()
        plt.savefig(fl[:-13]+'.png')
        plt.close()
        sys.exit(0)

    else:
        print('The format used for entries is... ')
        print('python object2plot.py  \
            "object2detections.py output"')
        sys.exit(1)