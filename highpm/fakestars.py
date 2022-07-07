#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import astropy.units as u
import astropy.io.fits as pf
from astropy.table import Table
import sys
import os
import argparse

from scipy import rand
from getFinalcutTile import *

def randoGen(low,high,n):
    return np.random.uniform(low,high,n)

if __name__=="__main__":
    help = "Still need to write the help section"

    if len(sys.argv)==2:
        if sys.argv[1]=='-h' or sys.argv[1]=='--help':
            print(help)
            sys.exit(1)
    if len(sys.argv)==3:
        tilename = sys.argv[1]
        pos = getTileBounds(tilename)
        n = int(sys.argv[2])
    if len(sys.argv)==4:
        ra = sys.argv[1]
        dec = sys.argv[2]
        n = int(sys.argv[3])
        tilename = getTileOf(ra,dec)
        pos = getTileBounds(tilename)
    
    idx = range(n)
    
    ramin = pos[0]
    ramax = pos[1]

    randRA = randoGen(ramin,ramax,n) * u.deg

    decmin = pos[2]
    decmax = pos[3]

    randDEC = randoGen(decmin,decmax,n) * u.deg

    randPM_slow = randoGen(0,100,int(0.1*n)) * u.mas/u.yr

    randPM_modest = randoGen(100,1000,int(0.1*n)) * u.mas/u.yr

    randPM_fast = randoGen(1000,18000,int(0.8*n)) * u.mas/u.yr

    randPM = np.hstack([randPM_slow, randPM_modest, randPM_fast])

    randTHETA = randoGen(0,2*np.pi,n)

    randPMRA = randPM * np.cos(randTHETA)
    randPMDEC = randPM * np.sin(randTHETA)


    def uniform_proposal(x, delta=2.0):
        return np.random.uniform(x - delta, x + delta)

    def p(x,low,high):
        if x>=low and x<high:
            return 1/(1+x**3) 
        else:
            return 0

    def metropolis_sampler(p, nsamples, proposal=uniform_proposal,domain=(0,10)):
        x = 1 # start somewhere
        low,high = domain
        for i in range(nsamples):
            trial = proposal(x) # random neighbour from the proposal distribution
            acceptance = p(trial,low,high)/p(x,low,high)

            # accept the move conditionally
            if np.random.uniform() < acceptance:
                x = trial

            yield x
            
    
        
    randPARALLAX = list(metropolis_sampler(p, n, domain=(0, 50))) 

    # randPARALLAX = randoGen(0,1000,n) * u.mas

    tbl = Table.read('/home/vwetzell/Documents/DES_HighPM/DES0043-3332/DES0043-3332.finalcut_movers.fits')

    tbl = tbl[tbl['nClip']==0]
    tbl = tbl[tbl['dof']>30]

    # tbl = tbl[tbl['Y_mag']>19]
    # tbl = tbl[tbl['z_mag']>19]
    # tbl = tbl[tbl['i_mag']>19]
    # tbl = tbl[tbl['r_mag']>19]
    # tbl = tbl[tbl['g_mag']>19]

    

    mag_dict = dict(zip(np.arange(len(tbl)),
        list(zip(tbl['g_mag'],tbl['r_mag'], tbl['i_mag'],tbl['z_mag'],tbl['Y_mag']))))

    randOFFSET = randoGen(-2,3,n)

    randSOURCE = np.random.randint(0,len(mag_dict),n)

    column_names = [
        "idx",
        "RA",
        "DEC",
        "PM",
        "PMRA",
        "PMDEC",
        "PARALLAX",
        "SOURCE",
        "OFFSET",
        "g_mag",
        "r_mag",
        "i_mag",
        "z_mag",
        "Y_mag"
    ]

    data = [
        idx,
        randRA,
        randDEC,
        randPM,
        randPMRA,
        randPMDEC,
        randPARALLAX,
        randSOURCE,
        randOFFSET,
        np.repeat(21.5,n),
        np.repeat(21.5,n),
        np.repeat(21.5,n),
        np.repeat(21.5,n),
        np.repeat(21.5,n),
        # np.array([mag_dict[i][0] for i in randSOURCE])+randOFFSET,
        # np.array([mag_dict[i][1] for i in randSOURCE])+randOFFSET,
        # np.array([mag_dict[i][2] for i in randSOURCE])+randOFFSET,
        # np.array([mag_dict[i][3] for i in randSOURCE])+randOFFSET,
        # np.array([mag_dict[i][4] for i in randSOURCE])+randOFFSET
    ]

    cat = Table(names=column_names,data=data)

    hdr = pf.Header()
    hdr['TILENAME'] = tilename

    table_hdu = pf.BinTableHDU(data=cat,header=hdr)

    table_hdu.writeto(tilename+'.fakes.cat',overwrite=True)

    sys.exit(0)




    