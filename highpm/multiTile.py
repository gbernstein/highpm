#!/usr/bin/env python

from __future__ import print_function
import sys
import os
import argparse
import numpy as np
import subprocess
import glob


if __name__=='__main__':
    help = "Still need to write the help section"

    my_cores=os.cpu_count()

    if len(sys.argv)==2:
        if sys.argv[1]=='-h' or sys.argv[1]=='--help':
            print(help)
            sys.exit(1)
        dir = sys.argv[1]
    else:
        print(help)
        sys.exit(1)

    fl = glob.glob(dir)
    # print(fl)

    for i in fl:
        print(i)
        cmd = ['/home/vwetzell/git_repos/highpm/highpm/DES_PM_v1.py',str(i)]
        subprocess.Popen(cmd).wait()
        


    sys.exit(0)