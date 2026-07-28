import glob
import os
import re
from typing import List

import fitsio
import h5py
import healpy as hp
import numpy as np
import numpy.lib.recfunctions as rfn
import pixmappy as pm
from astropy import units as u
from astropy.coordinates import SkyCoord, match_coordinates_sky

from highpm.crossmatch import mutual_nearest_neighbor
from highpm.gnomonic_converter import projectGnomonic

MIN_ID_MATCH_FRACTION = 0.5  # below this (or on a join_by error), fall back to RA/Dec matching
POSITION_MATCH_TOLERANCE = 1.0  # arcsec, for the RA/Dec fallback match


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
            "ALPHAWIN_J2000",
            "DELTAWIN_J2000",
            "XWIN_IMAGE",
            "YWIN_IMAGE",
            "SPREAD_MODEL",
            "SPREADERR_MODEL",
            "ERRAWIN_WORLD",
            "ERRBWIN_WORLD",
            "ERRTHETAWIN_J2000",
            "FLUX_PSF",
            "FLUXERR_PSF",
            "FLUX_AUTO",
            "FLUXERR_AUTO",
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

    # ponytail: fitsio returns id as bytes/object depending on the file; normalize to str before splitting
    try:
        splitIds = np.array(
            [(s.decode() if isinstance(s, bytes) else str(s)).split("_") for s in data["id"]],
            dtype=int,
        )
        if splitIds.ndim != 2 or splitIds.shape[1] != 2:
            raise ValueError(f"expected 'CCDNUM_OBJECTNUMBER' ids, got shape {splitIds.shape}")
        newData["OBJECT_NUMBER"] = splitIds[:, 1]
        newData["CCDNUM"] = splitIds[:, 0]
    except ValueError as e:
        print(
            f"Warning: could not parse GPR ids in {filename} ({e}); "
            "OBJECT_NUMBER/CCDNUM left invalid, will fall back to RA/Dec match."
        )
        newData["OBJECT_NUMBER"] = -1
        newData["CCDNUM"] = -1
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
            "RA",
            "DEC",
            "MAG_AUTO_G",
            "MAG_AUTO_R",
            "MAG_AUTO_I",
            "MAG_AUTO_Z",
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
    gprPath may be a single directory or a list of directories to search.
    """
    gprPaths = [gprPath] if isinstance(gprPath, str) else gprPath

    # Match any band suffix
    matches = [
        m
        for path in gprPaths
        for m in glob.glob(os.path.join(path, f"gpr_*{expnum:07d}_*.fits"))
    ]
    if not matches:
        # Fall back to legacy naming without band (older data sets)
        legacy = [
            m
            for path in gprPaths
            for m in glob.glob(os.path.join(path, f"gpr_*{expnum:07d}.fits"))
        ]
        if not legacy:
            # raise FileNotFoundError(
            #     f"No GPR files found for exposure number {expnum} in '{gprPath}'."
            # )
            return None
        return legacy[0]

    # If more than one match, choose preferred band order
    preferred = ["r", "i", "z", "g"]
    by_band = {}
    for path in matches:
        base = os.path.basename(path)
        # Expect ..._{band}.fits at the end; capture band robustly and case-insensitively
        m = re.search(r"_([A-Za-z]+)\.fits$", base, re.IGNORECASE)
        band = m.group(1).lower() if m else None
        if band and band not in by_band:
            by_band[band] = path

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
        os.path.join(coaddPath, f"cat_hpx_{pix:05d}.fits") for pix in coaddpix
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


def matchGPRToSkimByPosition(gprData, skimData, tolerance=POSITION_MATCH_TOLERANCE):
    """Fallback for matchGPRToSkim: match by RA/Dec (mutual nearest neighbor,
    arcsec tolerance) instead of by (OBJECT_NUMBER, CCDNUM) id."""
    ra1, dec1 = gprData["NEW_RA"], gprData["NEW_DEC"]
    ra2, dec2 = skimData["ALPHAWIN_J2000"], skimData["DELTAWIN_J2000"]
    ra0, dec0 = np.mean(ra1), np.mean(dec1)

    xi1, eta1, *_ = projectGnomonic(ra1, dec1, np.zeros_like(ra1), np.zeros_like(ra1), ra0, dec0)
    xi2, eta2, *_ = projectGnomonic(ra2, dec2, np.zeros_like(ra2), np.zeros_like(ra2), ra0, dec0)

    pts1 = np.vstack([xi1, eta1]).T * 3600.0
    pts2 = np.vstack([xi2, eta2]).T * 3600.0

    matched, idx, _ = mutual_nearest_neighbor(pts1, pts2, tolerance)

    # Keep skimData's OBJECT_NUMBER/CCDNUM (always genuine) rather than gprData's
    # (which may be unparsed sentinels -- see loadGPR), since downstream per-CCD
    # WCS lookups need a real CCDNUM.
    matchedGpr = rfn.drop_fields(gprData[matched], ["OBJECT_NUMBER", "CCDNUM"])
    matchedSkim = rfn.drop_fields(skimData[idx[matched]], ["ALPHAWIN_J2000", "DELTAWIN_J2000"])
    return rfn.merge_arrays(
        [matchedGpr, matchedSkim], asrecarray=True, usemask=False, flatten=True
    )


def _hasDuplicateKeys(data, keys=("OBJECT_NUMBER", "CCDNUM")):
    combined = np.stack([data[k] for k in keys], axis=1)
    _, counts = np.unique(combined, axis=0, return_counts=True)
    return bool(np.any(counts > 1))


def matchGPRToSkim(gprData, skimData):
    """Join gprData to skimData by (OBJECT_NUMBER, CCDNUM) id; fall back to a 2D
    RA/Dec match if either side has duplicate keys (rfn.join_by silently misaligns
    rows rather than erroring on those, per its own docstring) or the id join
    matches too few rows (e.g. object numbering changed between the GPR run and
    this skim file).
    """
    if _hasDuplicateKeys(gprData) or _hasDuplicateKeys(skimData):
        print("Duplicate (OBJECT_NUMBER, CCDNUM) keys found; falling back to RA/Dec match.")
        return matchGPRToSkimByPosition(gprData, skimData)

    nIdMatched = len(rfn.join_by(("OBJECT_NUMBER", "CCDNUM"), gprData, skimData, jointype="inner"))

    if nIdMatched >= MIN_ID_MATCH_FRACTION * len(gprData):
        return rfn.join_by(("OBJECT_NUMBER", "CCDNUM"), gprData, skimData, jointype="outer")

    print(
        f"ID match between GPR and skim only matched {nIdMatched}/{len(gprData)} rows; "
        "falling back to RA/Dec match."
    )
    return matchGPRToSkimByPosition(gprData, skimData)


def matchGPRToCoadd(coaddData, joinedSkimGPRData, radius=0.5):
    coaddCoords = SkyCoord(
        ra=coaddData["RA"],
        dec=coaddData["DEC"],
        unit="deg",
        frame="icrs",
    )

    coaddData = rfn.drop_fields(coaddData, ["RA", "DEC"])

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
    maps = pm.DelveMaps()

    # delveExposures.hdf5 is an astropy Table dumped to a single compound dataset.
    with h5py.File(os.environ["DES_EXPOSURES"], "r") as f:
        exposureTable = f["__astropy_table__"][:]

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

    iExp = np.where(exposureTable["expnum"] == expnum)[0]

    if len(iExp) == 0:
        print(f"**Warning**: Exposure number {expnum} not found in exposure table.")
        mjd = -1
        tempData["MJD"] = mjd

    else:
        iExp = iExp[0]
        mjd = exposureTable["mjdmid"][iExp]
        observatory = exposureTable["obsicrs"][iExp]

        tmpPar = np.dot(R_bl, observatory)
        parX = -tmpPar[0]
        parY = -tmpPar[1]

        giColor = (
            matchedSkimGPRCoaddData["MAG_AUTO_G"]
            - matchedSkimGPRCoaddData["MAG_AUTO_I"]
        )

        new_ra = matchedSkimGPRCoaddData["NEW_RA"]
        new_dec = matchedSkimGPRCoaddData["NEW_DEC"]

        n = len(matchedSkimGPRCoaddData)
        xwin = np.empty(n, dtype="f8")
        ywin = np.empty(n, dtype="f8")
        bestRA = np.empty(n, dtype="f8")
        bestDEC = np.empty(n, dtype="f8")
        shiftRA = np.empty(n, dtype="f8")
        shiftDEC = np.empty(n, dtype="f8")

        # One WCS per CCD: transform each CCD's whole row group at once instead
        # of calling toPix/toSky per detection.
        for iStart in range(len(starts) - 1):
            iUse = ccdnumArgsort[starts[iStart] : starts[iStart + 1]]
            ccdnum = int(matchedSkimGPRCoaddData["CCDNUM"][iUse[0]])

            wcs = maps.getDelveWCS(expnum, ccdnum)

            x, y = wcs.toPix(new_ra[iUse], new_dec[iUse], c=defaultColor)
            bra, bdec = wcs.toSky(x, y, c=giColor[iUse])
            sra, sdec = wcs.toSky(x, y, c=defaultColor - 1.0)

            xwin[iUse] = x
            ywin[iUse] = y
            bestRA[iUse] = bra
            bestDEC[iUse] = bdec
            shiftRA[iUse] = sra
            shiftDEC[iUse] = sdec

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

    if gprFile is None:
        print(f"No GPR file found for exposure {expnum}. Skipping.")
        return

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

    outputFile = os.path.join(outputPath, f"position_corrected_{expnum:08d}.fits")
    fitsio.write(
        outputFile, updatedData, extname="DATA", header=updatedHeader, clobber=True
    )
    print(f"Updated data saved to {outputFile}")

    return
