#!/usr/bin/env python

from __future__ import print_function
import sys
import os
import argparse
import numpy as np
import subprocess
import glob
from getFinalcutTile import *
from cat_reader import *



if __name__=='__main__':
    help = "Still need to write the help section"

    my_cores=os.cpu_count()

    if len(sys.argv)==2:
        if sys.argv[1]=='-h' or sys.argv[1]=='--help':
            print(help)
            sys.exit(1)
        catname = sys.argv[1]
    else:
        print(help)
        sys.exit(1)

    cat = read_cat_data(catname)

    ra = cat['ra']
    dec = cat['dec']

    tiles = []
    for i in range(len(ra)):
        print(i)
        try:
            tiles += [getTileOf(ra[i],dec[i])]
        except Exception:
            pass

    tiles = np.unique(tiles)

    print(len(tiles))
    print(tiles)

    for i in tiles:
        fl = glob.glob('./'+i+'*')
        if len(fl)==0:
            print(i)
            cmd = ['/home/vwetzell/git_repos/highpm/highpm/getFinalcutTile.py',str(i)]
            process = subprocess.Popen(cmd)
            process.wait()

    sys.exit(0)

