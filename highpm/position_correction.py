import glob
import os
import re

import fitsio
import h5py
import numpy as np
import numpy.lib.recfunctions as rfn
import pixmappy as pm

from highpm.crossmatch import mutual_nearest_neighbor
from highpm.gnomonic_converter import projectGnomonic

MIN_ID_MATCH_FRACTION = 0.5  # below this (or on a join_by error), fall back to RA/Dec matching
POSITION_MATCH_TOLERANCE = 1.0  # arcsec, for the RA/Dec fallback match
COLOR_DERIVATIVE_STEP = 0.01  # mag, full width of the symmetric finite-difference step


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
            "color_source",
            "color",
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
            ("BEST_RA_ERR", "f8"),
            ("BEST_DEC_ERR", "f8"),
            ("BEST_RA_DEC_CORR", "f8"),
            ("COLOR_SOURCE", "i1"),
            ("COLOR", "f8"),
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
    # cov_model lives in the same tangent-plane (xieta) basis as uv_model, in
    # arcsec**2, not degrees**2 -- so despite the RA/DEC-flavored name, these
    # are arcsec, and downstream code (pmfit.err2cov) uses them as such.
    newData["BEST_RA_ERR"] = np.sqrt(data["cov_model"][:, 0, 0])
    newData["BEST_DEC_ERR"] = np.sqrt(data["cov_model"][:, 1, 1])
    newData["BEST_RA_DEC_CORR"] = data["cov_model"][:, 0, 1] / (
        newData["BEST_RA_ERR"] * newData["BEST_DEC_ERR"]
    )
    # color_source is a pixmappy ColorConverter color-system code (0: g-i, 3: g-r,
    # 4: Gaia bp-rp) identifying which known color the GPR fit used for that star's
    # position, or -1 if no unique color was available and the fit assumed the
    # default color instead. color is the actual (already-converted, g-i-equivalent)
    # value that was used in either case.
    newData["COLOR_SOURCE"] = data["color_source"]
    newData["COLOR"] = data["color"]

    header = fitsio.read_header(filename, ext=1)
    return newData, header


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
        return rfn.join_by(
            ("OBJECT_NUMBER", "CCDNUM"), gprData, skimData, jointype="inner", usemask=False, asrecarray=True
        )

    print(
        f"ID match between GPR and skim only matched {nIdMatched}/{len(gprData)} rows; "
        "falling back to RA/Dec match."
    )
    return matchGPRToSkimByPosition(gprData, skimData)


def sky2bestSky(matchedSkimGPRData, expnum, ra0, dec0):
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

    ccdnumArgsort = np.argsort(matchedSkimGPRData["CCDNUM"])
    ccdnumSort = matchedSkimGPRData["CCDNUM"][ccdnumArgsort]
    tmp = ccdnumSort[1:] != ccdnumSort[:-1]
    starts = np.concatenate(([0], np.where(tmp)[0] + 1, [len(ccdnumSort)]))

    tempData = np.zeros(
        len(matchedSkimGPRData),
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

        # The GPR fit already used each star's own listed COLOR (identified by
        # COLOR_SOURCE, a pixmappy ColorConverter system code, or -1 if no
        # unique color was available and the fit assumed the default color
        # instead) when solving for new_rd, so new_rd is already the best
        # position -- no detection-level position remap is needed here. The
        # same listed color is what we center the finite-difference color
        # derivative on below, so it's evaluated at the same color the GPR fit
        # actually used, not a coadd-derived stand-in.
        hasKnownColor = matchedSkimGPRData["COLOR_SOURCE"] != -1
        centerColor = matchedSkimGPRData["COLOR"]

        new_ra = matchedSkimGPRData["NEW_RA"]
        new_dec = matchedSkimGPRData["NEW_DEC"]

        n = len(matchedSkimGPRData)
        xwin = np.empty(n, dtype="f8")
        ywin = np.empty(n, dtype="f8")
        dRAdColor = np.empty(n, dtype="f8")
        dDECdColor = np.empty(n, dtype="f8")

        # One WCS per CCD: transform each CCD's whole row group at once instead
        # of calling toPix/toSky per detection.
        for iStart in range(len(starts) - 1):
            iUse = ccdnumArgsort[starts[iStart] : starts[iStart + 1]]
            ccdnum = int(matchedSkimGPRData["CCDNUM"][iUse[0]])

            wcs = maps.getDelveWCS(expnum, ccdnum)

            x, y = wcs.toPix(new_ra[iUse], new_dec[iUse], c=centerColor[iUse])

            # Exact color derivative at each detection's own center color: a
            # small symmetric step, rather than a 1-mag secant, so the slope
            # reflects the local (possibly curved, e.g. DCR) color response
            # instead of averaging over several tabulated segments.
            cPlus = centerColor[iUse] + COLOR_DERIVATIVE_STEP / 2.0
            cMinus = centerColor[iUse] - COLOR_DERIVATIVE_STEP / 2.0
            raPlus, decPlus = wcs.toSky(x, y, c=cPlus)
            raMinus, decMinus = wcs.toSky(x, y, c=cMinus)

            xwin[iUse] = x
            ywin[iUse] = y
            dRAdColor[iUse] = (raPlus - raMinus) / COLOR_DERIVATIVE_STEP
            dDECdColor[iUse] = (decPlus - decMinus) / COLOR_DERIVATIVE_STEP

        tempData["MJD"] = mjd
        tempData["PAR_XI"] = parX
        tempData["PAR_ETA"] = parY
        tempData["NEW_XWIN_IMAGE"] = xwin
        tempData["NEW_YWIN_IMAGE"] = ywin
        tempData["BEST_RA"] = matchedSkimGPRData["NEW_RA"]
        tempData["BEST_DEC"] = matchedSkimGPRData["NEW_DEC"]
        # Rows with a known color already had it used in the GPR fit, so
        # there's no remaining color ambiguity to model. Only rows that fell
        # back to the default color get a real (nonzero) derivative.
        tempData["DRA_DCOLOR"] = np.where(hasKnownColor, 0.0, dRAdColor)
        tempData["DDEC_DCOLOR"] = np.where(hasKnownColor, 0.0, dDECdColor)

    updatedSkimGPRData = rfn.merge_arrays(
        [matchedSkimGPRData, tempData],
        asrecarray=True,
        usemask=False,
        flatten=True,
    )

    return updatedSkimGPRData


def process_exposure(expnum, skimsPath, gprPath, outputPath):
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

    try:
        updatedData = sky2bestSky(
            joinedSkimGPRData,
            expnum,
            gprHeader["RA0"],
            gprHeader["DEC0"],
        )
    except ValueError as e:
        # e.g. pixmappy has no WCS solution for this exposure -- a permanent
        # failure, not a transient one. Skip it rather than letting the whole
        # multiprocessing chunk (and any other exposures batched with it) die.
        print(f"Skipping exposure {expnum}: {e}")
        return

    # NEW_RA/NEW_DEC are now redundant with BEST_RA/BEST_DEC (sky2bestSky just
    # copies them through, since the GPR fit already used each detection's
    # listed color), so don't carry the duplicate columns into the output file.
    updatedData = rfn.drop_fields(updatedData, ["NEW_RA", "NEW_DEC"])

    updatedHeader = gprHeader
    updatedHeader["EXPNUM"] = expnum

    outputFile = os.path.join(outputPath, f"position_corrected_{expnum:08d}.fits")
    fitsio.write(
        outputFile, updatedData, extname="DATA", header=updatedHeader, clobber=True
    )
    print(f"Updated data saved to {outputFile}")

    return
