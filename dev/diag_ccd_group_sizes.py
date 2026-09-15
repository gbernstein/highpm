#!/usr/bin/env python
"""One-off diagnostic: for a given exposure, print how many matched
GPR/skim detections land on each CCD, so we can see whether the CCDs
with only 1 detection (which crash pixmappy's inverse) reflect a
matching bug or genuinely sparse data.

Run on the HPC login node (needs conda env `pm` + /data8):
    python dev/diag_ccd_group_sizes.py <expnum> [<expnum> ...]
"""
import sys

import numpy as np

from highpm.position_correction import (
    getGPRFile,
    getSkimFile,
    loadGPR,
    loadSkim,
    matchGPRToSkim,
)

SKIMS_PATH = "/data8/shared/decampm/[GRIZ]/"
GPR_PATH = "/data8/shared/decampm/GPR2/CAT/[griz]/"

for expnum in map(int, sys.argv[1:]):
    skimFile = getSkimFile(expnum, SKIMS_PATH)
    gprFile = getGPRFile(expnum, GPR_PATH)
    if gprFile is None:
        print(f"{expnum}: no GPR file found")
        continue

    skimData, _ = loadSkim(skimFile)
    gprData, _ = loadGPR(gprFile)
    matched = matchGPRToSkim(gprData, skimData)

    ccds, counts = np.unique(matched["CCDNUM"], return_counts=True)
    singleton_ccds = ccds[counts == 1]
    print(
        f"{expnum}: skim={len(skimData)} gpr={len(gprData)} matched={len(matched)} "
        f"ccds={len(ccds)} singleton_ccds={list(singleton_ccds)} "
        f"count_hist={dict(zip(*np.unique(counts, return_counts=True)))}"
    )
