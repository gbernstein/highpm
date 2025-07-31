import glob
import os

import fitsio
import healpy as hp
import numpy as np
import numpy.lib.recfunctions as rfn
import pixmappy as pm
from astropy import units as u
from astropy.coordinates import SkyCoord, match_coordinates_sky


def loadSkim(filename):
    """Load specific columns and header from a FITS file using fitsio.

    Parameters
    ----------
    filename : str
        Path to the FITS file to be read.
    Returns
    -------
    data : numpy.recarray
        Array containing the selected columns from the FITS file extension 1.
    header : fitsio.FITSHDR
        Header of the FITS file extension 1.
    Raises
    ------
    FileNotFoundError
        If the specified FITS file does not exist.
    Notes
    -----
    The function reads the following columns: 'OBJECT_NUMBER', 'CCDNUM', 'BAND',
    'XWIN_IMAGE', 'YWIN_IMAGE', 'SPREAD_MODEL', 'SPREADERR_MODEL', 'ERRAIWN_WORLD',
    'FLUX_PSF', 'FLUXERR_PSF', 'FLAGS', and 'IMAFLAGS_ISO'.
    """

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
            "ERRAIWN_WORLD",
            "FLUX_PSF",
            "FLUXERR_PSF",
            "FLAGS",
            "IMAFLAGS_ISO",
        ],
    )
    header = fitsio.read_header(filename, ext=1)
    return data, header


def loadGPR(filename):
    """Load and process GPR data from a FITS file. Reads a FITS file containing GPR
    data, filters out rows with invalid covariance matrices, and returns a structured
    NumPy array with selected columns and a FITS header.

    Parameters
    ----------
    filename : str
        Path to the FITS file to be loaded.
    Returns
    -------
    newData : numpy.ndarray
        Structured array containing the following fields:
        - OBJECT_NUMBER (int): Object number extracted from the 'id' column.
        - CCDNUM (int): CCD number extracted from the 'id' column.
        - NEW_RA (float): New right ascension value.
        - NEW_DEC (float): New declination value.
        - NEW_RA_ERR (float): Error in right ascension.
        - NEW_DEC_ERR (float): Error in declination.
        - HAS_UNIQUE_COLOR (int): Flag indicating unique color.
    header : fitsio.FITSHDR
        Header of the FITS file extension 1.
    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    """

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

    data = data[
        (data["cov_model"][:, 0, 0] > 0.0) & (data["cov_model"][:, 1, 1] > 0.0)
    ]  # Filter out rows with invalid covariance

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
    """Load specific columns and header from a FITS file using fitsio.

    Parameters
    ----------
    filename : str
        Path to the FITS file to be loaded.
    Returns
    -------
    data : numpy.ndarray
        Array containing the selected columns from the FITS file extension 1.
    header : fitsio.FITSHDR
        Header of the FITS file extension 1.
    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    Notes
    -----
    The function reads the following columns from the FITS file:
    'ALPHAWIN_J2000', 'DELTAWIN_J2000', 'MAG_AUTO_G', 'MAG_AUTO_R',
    'MAG_AUTO_I', 'MAG_AUTO_Z', and 'MAG_AUTO_Y'.
    """

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
    """Searches for a skim FITS file corresponding to a given exposure number.

    Parameters
    ----------
    expnum : int
        The exposure number to search for in the skim file names.
    skimsPath : str, optional
        The directory path where skim files are located. Defaults to the current
        directory ("./").
    Returns
    -------
    str
        The path to the first skim file found matching the exposure number.
    Raises
    ------
    FileNotFoundError
        If no skim files matching the exposure number are found in the specified
        directory.
    """

    skimFile = glob.glob(skimsPath + f"D*{expnum:08d}_*.fits")
    if not skimFile:
        raise FileNotFoundError(f"No skim files found for exposure number {expnum}.")
    return skimFile[0]


def getGPRFile(expnum, gprPath="./"):
    """Retrieve the path to a GPR FITS file corresponding to a given exposure number.

    Parameters
    ----------
    expnum : int
        The exposure number to search for in the GPR file name.
    gprPath : str, optional
        The directory path where GPR files are located. Defaults to the current
        directory ("./").
    Returns
    -------
    str
        The path to the first matching GPR FITS file.
    Raises
    ------
    FileNotFoundError
        If no GPR files matching the exposure number are found in the specified
        directory.
    """

    gprFile = glob.glob(gprPath + f"gpr_*{expnum:07d}.fits")
    if not gprFile:
        raise FileNotFoundError(f"No GPR files found for exposure number {expnum}.")
    return gprFile[0]


def findCoaddFile(ra0, dec0, radius=1.1, nside=32, coaddPath="./"):
    """Finds the list of coadd FITS files overlapping a given sky position.

    Parameters
    ----------
    ra0 : float
        Right ascension of the target position in degrees.
    dec0 : float
        Declination of the target position in degrees.
    radius : float, optional
        Search radius around the target position in degrees. Default is 1.1.
    nside : int, optional
        HEALPix NSIDE parameter defining the resolution. Default is 32.
    coaddPath : str, optional
        Path to the directory containing the coadd FITS files. Default is "./".
    Returns
    -------
    coaddFiles : list of str
        List of file paths to the coadd FITS files overlapping the specified
        sky region.
    """

    pointing = hp.dir2vec(ra0, dec0, lonlat=True)
    ipix = hp.query_disc(nside, pointing, np.radians(radius), inclusive=True)
    npix = np.arange(hp.nside2npix(nside))
    coaddpix = npix[ipix]
    coaddFiles = [coaddPath + f"y6_gold_2_2_{pix:05d}.fits" for pix in coaddpix]
    return coaddFiles


def combineCoadds(coaddFiles):
    """Combines data from multiple coadd files into a single NumPy array.

    Parameters
    ----------
    coaddFiles : list of str
        List of file paths to coadd files to be combined.
    Returns
    -------
    numpy.ndarray
        Concatenated data from all valid coadd files.
    Raises
    ------
    ValueError
        If no valid coadd files are found to combine.
    Notes
    -----
    Files that do not exist are skipped with a warning message.
    """

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
    """Matches GPR data to skim data by joining on 'OBJECT_NUMBER' and 'CCDNUM'.

    Parameters
    ----------
    gprData : numpy.ndarray or structured array
        The GPR data to be matched, expected to contain 'OBJECT_NUMBER' and
        'CCDNUM' fields.
    skimData : numpy.ndarray or structured array
        The skim data to be matched, expected to contain 'OBJECT_NUMBER' and
        'CCDNUM' fields.
    Returns
    -------
    numpy.ndarray
        The joined array containing data from both `gprData` and `skimData`,
        matched on 'OBJECT_NUMBER' and 'CCDNUM' using an outer join.
    """

    joinedSkimGPRData = rfn.join_by(
        ("OBJECT_NUMBER", "CCDNUM"), gprData, skimData, jointype="outer"
    )
    return joinedSkimGPRData


def matchGPRToCoadd(coaddData, joinedSkimGPRData, radius=0.5):
    """Matches GPR sources to coadd sources within a specified radius and merges data.

    Parameters
    ----------
    coaddData : numpy.recarray
        Record array containing coadd source data, including 'ALPHAWIN_J2000' and
        'DELTAWIN_J2000' fields for RA and Dec in degrees.
    joinedSkimGPRData : numpy.recarray
        Record array containing joined skim and GPR data, including 'NEW_RA' and
        'NEW_DEC' fields for RA and Dec in degrees.
    radius : float, optional
        Matching radius in arcseconds. Default is 0.5.
    Returns
    -------
    matchedSkimGPRCoaddData : numpy.recarray
        Record array with the original joinedSkimGPRData fields and the matched
        coaddData fields appended. If no match is found within the radius, the
        coadd fields are set to -99.0.
    """

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

    # Append coadd data to joined GPR and skim data
    matchedSkimGPRCoaddData = rfn.merge_arrays(
        [joinedSkimGPRData, matchedCoaddData],
        asrecarray=True,
        usemask=False,
        flatten=True,
    )

    return matchedSkimGPRCoaddData


def sky2bestSky(matchedSkimGPRCoaddData, expnum, ra0, dec0, defaultColor=1.0):
    """Transforms sky coordinates to "best" sky coordinates and computes chromatic
    derivatives for a set of matched objects, updating the input data with new
    astrometric and photometric information.

    Parameters
    ----------
    matchedSkimGPRCoaddData : numpy.recarray
        Input record array containing matched object data, including sky
        coordinates, CCD numbers, and photometric measurements.
    expnum : int
        Exposure number to identify the relevant exposure in the exposure table.
    ra0 : float
        Reference right ascension (in degrees) for the transformation.
    dec0 : float
        Reference declination (in degrees) for the transformation.
    defaultColor : float, optional
        Default color value to use for chromatic transformations (default is 1.0).
    Returns
    -------
    updatedSkimGPRCoaddData : numpy.recarray
        Record array with additional fields for MJD, parallactic angle components,
        transformed pixel coordinates, best-fit sky coordinates, and chromatic
        derivatives.
    Notes
    -----
    This function relies on external DESMaps and WCS transformation utilities, as
    well as an exposure table specified by the "DES_EXPOSURES" environment
    variable. The function assumes the input data contains specific fields such as
    "CCDNUM", "NEW_RA", "NEW_DEC", "MAG_AUTO_G", "MAG_AUTO_I", and
    "HAS_UNIQUE_COLOR".
    """

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
        iExp = iExp[0]
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

        print("Transforming back to pixel coordinates...")

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

        print("Calculating best RA and DEC with COADD color...")

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

        print("Calculating chromatic derivatives for RA and DEC...")

        shiftRADEC = np.fromiter(
            (
                wcs.toSky(x, y, c=defaultColor - 1.0)
                for wcs, x, y in zip(wcsArray, xwin, ywin)
            ),
            dtype=[("shiftRA", "f8"), ("shiftDEC", "f8")],
        )

        shiftRA = shiftRADEC["shiftRA"]
        shiftDEC = shiftRADEC["shiftDEC"]

        print("Pixmappy transformations complete.")

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
            matchedSkimGPRCoaddData["HAS_UNIQUE_COLOR"],
            0.0,
            dRAdColor,
        )
        tempData["DDEC_DCOLOR"] = np.where(
            matchedSkimGPRCoaddData["HAS_UNIQUE_COLOR"],
            0.0,
            dDECdColor,
        )

    updatedSkimGPRCoaddData = rfn.merge_arrays(
        [matchedSkimGPRCoaddData, tempData],
        asrecarray=True,
        usemask=False,
        flatten=True,
    )

    return updatedSkimGPRCoaddData


def process_exposure(expnum):
    """Processes a single exposure by matching and updating astronomical data. This
    function loads skim and GPR files for a given exposure number, matches their data,
    finds and combines relevant coadd files, and updates the matched data using sky
    coordinate corrections. The final updated data is saved as a new FITS file.

    Parameters
    ----------
    expnum : int
        The exposure number to process.
    Returns
    -------
    None
        This function does not return a value. The updated data is saved to disk.
    Notes
    -----
    - Requires several helper functions: getSkimFile, getGPRFile, loadSkim,
      loadGPR, matchGPRToSkim, findCoaddFile, combineCoadds, matchGPRToCoadd,
      and sky2bestSky.
    - Output FITS files are saved in a predefined directory.
    """

    skimsPath = "/home/vwetzell/Documents/Sculptor/Sculptor_skims/*/"
    gprPath = "/home/vwetzell/Documents/Sculptor/Sculptor_GPR/"
    coaddPath = "/home/vwetzell/Documents/Sculptor/Sculptor_coadds/"

    print(f"Processing exposure number: {expnum}")

    skimFile = getSkimFile(expnum, skimsPath)
    gprFile = getGPRFile(expnum, gprPath)

    print(f"Skim file: {skimFile}")
    print(f"GPR file: {gprFile}")

    skimData, skimHeader = loadSkim(skimFile)
    gprData, gprHeader = loadGPR(gprFile)

    joinedSkimGPRData = matchGPRToSkim(gprData, skimData)

    coaddFiles = findCoaddFile(gprHeader["RA0"], gprHeader["DEC0"], coaddPath=coaddPath)

    coaddFiles = [os.path.join(coaddPath, f) for f in coaddFiles]

    coaddData = combineCoadds(coaddFiles)

    matchedData = matchGPRToCoadd(coaddData, joinedSkimGPRData)

    updatedData = sky2bestSky(
        matchedData,
        expnum,
        gprHeader["RA0"],
        gprHeader["DEC0"],
        defaultColor=gprHeader["DEF_COL"],
    )

    outputPath = "/home/vwetzell/Documents/Sculptor/Sculptor_new_updated/"

    # Save the updated data to a new FITS file
    outputFile = outputPath + f"updated_skim_gpr_coadd_{expnum:08d}.fits"
    fitsio.write(outputFile, updatedData, extname="DATA", clobber=True)
    print(f"Updated data saved to {outputFile}")

    return


if __name__ == "__main__":
    exposures = np.load(
        "/home/vwetzell/Documents/Sculptor/Sculptor_exposures.npy",
        allow_pickle=True,
    )

    from multiprocessing import Pool

    with Pool(processes=24) as pool:
        pool.map(
            process_exposure,
            exposures,
        )
