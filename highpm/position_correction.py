import glob
import os
import re
from typing import List

import fitsio
import healpy as hp
import numpy as np
import numpy.lib.recfunctions as rfn
import pixmappy as pm
from astropy import units as u
from astropy.coordinates import SkyCoord, match_coordinates_sky


def loadSkim(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} does not exist.")

    data = fitsio.read(
        filename,
        ext=1,
        columns=[
            "OBJECT_NUMBER",
            "CCDNUM",
            "BAND",
            "XWIN_IMAGE",
            "YWIN_IMAGE",
            "SPREAD_MODEL",
            "SPREADERR_MODEL",
            "ERRAWIN_WORLD",
            "FLUX_PSF",
            "FLUXERR_PSF",
            "FLAGS",
            "IMAFLAGS_ISO",
        ],
    )
    header = fitsio.read_header(filename, ext=1)
    return data, header


def loadGPR(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} does not exist.")

    data: np.ndarray = fitsio.read(
        filename,
        ext=1,
        columns=[
            "id",
            "new_rd",
            "cov_model",
            "has_color",
        ],
    )

    data = data[(data["cov_model"][:, 0, 0] > 0.0) & (data["cov_model"][:, 1, 1] > 0.0)]

    newData = np.zeros(
        len(data),
        dtype=[
            ("OBJECT_NUMBER", "i4"),
            ("CCDNUM", "i4"),
            ("NEW_RA", "f8"),
            ("NEW_DEC", "f8"),
            ("NEW_RA_ERR", "f8"),
            ("NEW_DEC_ERR", "f8"),
            ("HAS_UNIQUE_COLOR", "i1"),
        ],
    )

    splitIds = np.array(np.char.split(data["id"], "_").tolist(), dtype=int)

    newData["OBJECT_NUMBER"] = splitIds[:, 1]
    newData["CCDNUM"] = splitIds[:, 0]
    newData["NEW_RA"] = data["new_rd"][:, 0]
    newData["NEW_DEC"] = data["new_rd"][:, 1]
    newData["NEW_RA_ERR"] = np.sqrt(data["cov_model"][:, 0, 0])
    newData["NEW_DEC_ERR"] = np.sqrt(data["cov_model"][:, 1, 1])
    newData["HAS_UNIQUE_COLOR"] = data["has_color"]

    header = fitsio.read_header(filename, ext=1)
    return newData, header


def loadCoadd(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} does not exist.")

    data = fitsio.read(
        filename,
        ext=1,
        columns=[
            "ALPHAWIN_J2000",
            "DELTAWIN_J2000",
            "MAG_AUTO_G",
            "MAG_AUTO_R",
            "MAG_AUTO_I",
            "MAG_AUTO_Z",
            "MAG_AUTO_Y",
        ],
    )
    header = fitsio.read_header(filename, ext=1)
    return data, header


def getSkimFile(expnum, skimsPath="./"):
    pattern = os.path.join(skimsPath, f"D*{expnum:08d}_*.fits")
    skimFile = glob.glob(pattern)
    if not skimFile:
        raise FileNotFoundError(f"No skim files found for exposure number {expnum}.")
    return skimFile[0]


def getGPRFile(expnum, gprPath="./"):
    """
    Locate a GPR file for a given exposure.

    New convention: files are named like 'gpr_*{expnum:07d}_{band}.fits'.
    If multiple bands exist for the same exposure, prefer a deterministic band order.
    """
    # Match any band suffix
    pattern = os.path.join(gprPath, f"gpr_*{expnum:07d}_*.fits")
    matches = glob.glob(pattern)
    if not matches:
        # Fall back to legacy naming without band (older data sets)
        legacy = glob.glob(os.path.join(gprPath, f"gpr_*{expnum:07d}.fits"))
        if not legacy:
            raise FileNotFoundError(
                f"No GPR files found for exposure number {expnum} in '{gprPath}'."
            )
        return legacy[0]

    # If more than one match, choose preferred band order
    preferred = ["r", "i", "z", "g"]
    by_band = {}
    for path in matches:
        base = os.path.basename(path)
        # Expect ..._{band}.fits at the end
        m = re.search(r"_[griz]\.fits$".replace(" ", ""), base)
        band = m.group(1) if m else None
        by_band.setdefault(band, path)

    for b in preferred:
        if b in by_band:
            return by_band[b]

    # As a last resort, return the first match sorted lexicographically
    return sorted(matches)[0]


def findCoaddFile(ra0, dec0, radius=1.1, nside=32, coaddPath="./") -> List[str]:
    pointing = hp.dir2vec(ra0, dec0, lonlat=True)
    ipix = hp.query_disc(nside, pointing, np.radians(radius), inclusive=True)
    npix = np.arange(hp.nside2npix(nside))
    coaddpix = npix[ipix]
    coaddFiles = [
        os.path.join(coaddPath, f"y6_gold_2_2_{pix:05d}.fits") for pix in coaddpix
    ]
    return coaddFiles


def combineCoadds(coaddFiles: List[str]):
    combinedData = []
    for coaddFile in coaddFiles:
        if not os.path.exists(coaddFile):
            print(f"Warning: Coadd file {coaddFile} does not exist.")
            continue
        data, _ = loadCoadd(coaddFile)
        combinedData.append(data)

    if not combinedData:
        raise ValueError("No valid coadd files found to combine.")

    return np.concatenate(combinedData)


def matchGPRToSkim(gprData, skimData):
    return rfn.join_by(("OBJECT_NUMBER", "CCDNUM"), gprData, skimData, jointype="outer")


def matchGPRToCoadd(coaddData, joinedSkimGPRData, radius=0.5):
    coaddCoords = SkyCoord(
        ra=coaddData["ALPHAWIN_J2000"],
        dec=coaddData["DELTAWIN_J2000"],
        unit="deg",
        frame="icrs",
    )

    coaddData = rfn.drop_fields(coaddData, ["ALPHAWIN_J2000", "DELTAWIN_J2000"])

    gprCoords = SkyCoord(
        ra=joinedSkimGPRData["NEW_RA"],
        dec=joinedSkimGPRData["NEW_DEC"],
        unit="deg",
        frame="icrs",
    )

    idx, d2d, _ = match_coordinates_sky(gprCoords, coaddCoords, nthneighbor=1)

    matchedCoaddData = coaddData[idx]
    matchedCoaddData[d2d > radius * u.arcsec] = -99.0  # type: ignore[attr-defined]

    matchedSkimGPRCoaddData = rfn.merge_arrays(
        [joinedSkimGPRData, matchedCoaddData],
        asrecarray=True,
        usemask=False,
        flatten=True,
    )

    return matchedSkimGPRCoaddData


def sky2bestSky(matchedSkimGPRCoaddData, expnum, ra0, dec0, defaultColor=1.0):
    maps = pm.DESMaps()

    exposureTable = fitsio.read(os.environ["DES_EXPOSURES"], ext=1)

    cra, sra = np.cos(ra0 * np.pi / 180.0), np.sin(ra0 * np.pi / 180.0)
    cdec, sdec = np.cos(dec0 * np.pi / 180.0), np.sin(dec0 * np.pi / 180.0)
    R_bl = np.array(
        [
            [-sra, cra, 0.0],
            [-cra * sdec, -sra * sdec, cdec],
            [cra * cdec, sra * cdec, sdec],
        ]
    )

    ccdnumArgsort = np.argsort(matchedSkimGPRCoaddData["CCDNUM"])
    ccdnumSort = matchedSkimGPRCoaddData["CCDNUM"][ccdnumArgsort]
    tmp = ccdnumSort[1:] != ccdnumSort[:-1]
    starts = np.concatenate(([0], np.where(tmp)[0] + 1, [len(ccdnumSort)]))

    tempData = np.zeros(
        len(matchedSkimGPRCoaddData),
        dtype=[
            ("MJD", "f8"),
            ("PAR_XI", "f8"),
            ("PAR_ETA", "f8"),
            ("NEW_XWIN_IMAGE", "f8"),
            ("NEW_YWIN_IMAGE", "f8"),
            ("BEST_RA", "f8"),
            ("BEST_DEC", "f8"),
            ("DRA_DCOLOR", "f8"),
            ("DDEC_DCOLOR", "f8"),
        ],
    )

    wcsArray = np.empty(len(matchedSkimGPRCoaddData), dtype=object)

    iExp = np.where(exposureTable["expnum"] == expnum)[0]

    if len(iExp) == 0:
        print(f"**Warning**: Exposure number {expnum} not found in exposure table.")
        mjd = -1
        tempData["MJD"] = mjd

    else:
        iExp = iExp[0]
        mjd = exposureTable["mjd"][iExp]
        observatory = exposureTable["observatory"][iExp]

        tmpPar = np.dot(R_bl, observatory)
        parX = -tmpPar[0]
        parY = -tmpPar[1]

        giColor = (
            matchedSkimGPRCoaddData["MAG_AUTO_G"]
            - matchedSkimGPRCoaddData["MAG_AUTO_I"]
        )

        for iStart in range(len(starts) - 1):
            iUse = ccdnumArgsort[starts[iStart] : starts[iStart + 1]]
            ccdnum = int(matchedSkimGPRCoaddData["CCDNUM"][iUse[0]])

            wcs = maps.getDESWCS(expnum, ccdnum)

            wcsArray[iUse] = wcs

        xywin = np.fromiter(
            (
                wcs.toPix(ra, dec, c=defaultColor)
                for wcs, ra, dec in zip(
                    wcsArray,
                    matchedSkimGPRCoaddData["NEW_RA"],
                    matchedSkimGPRCoaddData["NEW_DEC"],
                )
            ),
            dtype=[("xwin", "f8"), ("ywin", "f8")],
        )

        xwin = xywin["xwin"]
        ywin = xywin["ywin"]

        bestRADEC = np.fromiter(
            (
                wcs.toSky(x, y, c=c)
                for wcs, x, y, c in zip(
                    wcsArray,
                    xwin,
                    ywin,
                    giColor,
                )
            ),
            dtype=[("bestRA", "f8"), ("bestDEC", "f8")],
        )

        bestRA = bestRADEC["bestRA"]
        bestDEC = bestRADEC["bestDEC"]

        shiftRADEC = np.fromiter(
            (
                wcs.toSky(x, y, c=defaultColor - 1.0)
                for wcs, x, y in zip(wcsArray, xwin, ywin)
            ),
            dtype=[("shiftRA", "f8"), ("shiftDEC", "f8")],
        )

        shiftRA = shiftRADEC["shiftRA"]
        shiftDEC = shiftRADEC["shiftDEC"]

        dRAdColor = matchedSkimGPRCoaddData["NEW_RA"] - shiftRA
        dDECdColor = matchedSkimGPRCoaddData["NEW_DEC"] - shiftDEC

        tempData["MJD"] = mjd
        tempData["PAR_XI"] = parX
        tempData["PAR_ETA"] = parY
        tempData["NEW_XWIN_IMAGE"] = xwin
        tempData["NEW_YWIN_IMAGE"] = ywin
        tempData["BEST_RA"] = np.where(
            matchedSkimGPRCoaddData["HAS_UNIQUE_COLOR"],
            matchedSkimGPRCoaddData["NEW_RA"],
            bestRA,
        )
        tempData["BEST_DEC"] = np.where(
            matchedSkimGPRCoaddData["HAS_UNIQUE_COLOR"],
            matchedSkimGPRCoaddData["NEW_DEC"],
            bestDEC,
        )
        tempData["DRA_DCOLOR"] = np.where(
            matchedSkimGPRCoaddData["HAS_UNIQUE_COLOR"], 0.0, dRAdColor
        )
        tempData["DDEC_DCOLOR"] = np.where(
            matchedSkimGPRCoaddData["HAS_UNIQUE_COLOR"], 0.0, dDECdColor
        )

    updatedSkimGPRCoaddData = rfn.merge_arrays(
        [matchedSkimGPRCoaddData, tempData],
        asrecarray=True,
        usemask=False,
        flatten=True,
    )

    return updatedSkimGPRCoaddData


def process_exposure(expnum, skimsPath, gprPath, coaddPath, outputPath):
    print(f"Processing exposure number: {expnum}")

    skimFile = getSkimFile(expnum, skimsPath)
    gprFile = getGPRFile(expnum, gprPath)

    print(f"Skim file: {skimFile}")
    print(f"GPR file: {gprFile}")

    skimData, skimHeader = loadSkim(skimFile)
    gprData, gprHeader = loadGPR(gprFile)

    joinedSkimGPRData = matchGPRToSkim(gprData, skimData)

    coaddFiles = findCoaddFile(gprHeader["RA0"], gprHeader["DEC0"], coaddPath=coaddPath)

    coaddData = combineCoadds(coaddFiles)

    matchedData = matchGPRToCoadd(coaddData, joinedSkimGPRData)

    updatedData = sky2bestSky(
        matchedData,
        expnum,
        gprHeader["RA0"],
        gprHeader["DEC0"],
        defaultColor=gprHeader["DEF_COL"],
    )

    updatedHeader = gprHeader
    updatedHeader["EXPNUM"] = expnum

    outputFile = os.path.join(outputPath, f"updated_skim_gpr_coadd_{expnum:08d}.fits")
    fitsio.write(
        outputFile, updatedData, extname="DATA", header=updatedHeader, clobber=True
    )
    print(f"Updated data saved to {outputFile}")

    return
