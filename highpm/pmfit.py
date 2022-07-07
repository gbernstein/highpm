''' 
Fit full five-parameter PM/parallax model to observations.
'''

## ??? Need to gaurd against bad fits,  perhaps a v weak prior on PM

from cmath import exp
import numpy as np
from getFinalcutTile import *

def singleFit(xy_in,cov_xy, t,par_xy, parallax_prior=None):
    '''Fit 5-parameter model to stellar observations, where:
    `xy_in` is Nx2 array of observations of the star, in arcsec
    `cov` is Nx3 array giving (sig^2_x, sig^2_y, cov_xy) for each
    `t`  is time of observation of each (in yrs)
    `par_xy` are -1*(projected earth position components), in AU
    `parallax_prior` is prior sigma for parallax, in arcsec/yr
    Returns:
    `soln`  the 5-param covariance (x0, y0, vx, vy, parallax)
    `cov`   covariance matrix of this
    `chisq` vector of chisq of each observation.'''

    # Remove means from xy for numerical stability
    xyMean = np.mean(xy_in, axis=0)
    xy = xy_in - xyMean
    # Build matrices
    npts = xy.shape[0]
    one = np.ones(npts, dtype=float)
    zero = np.zeros(npts, dtype=float)

    # Build 2 x N x 5 matrix of coefficients
    A = np.array([[one, zero, t, zero, par_xy[:,0]],
                  [zero, one, zero, t, par_xy[:,1]]])
    A = np.swapaxes(A,1,2)
    
    # xy is the 2 x N "x" vector
    # Build 2 x 2 x N vector of inverse covariances:
    det = cov_xy[:,0]*cov_xy[:,1] - cov_xy[:,2]*cov_xy[:,2]
    invC = np.array( [[cov_xy[:,1], -cov_xy[:,2]],
                      [-cov_xy[:,2], cov_xy[:,0]]]) / det

    # Now the solution - contract over first two dimensions of A
    alpha = np.einsum("ikm,ijk,jkn",A,invC,A)
    beta = np.einsum("ikm,ijk,kj",A,invC,xy)
    # Add parallax prior
    if parallax_prior is not None:
        alpha[4,4] += parallax_prior**-2
    # Solve for 5 parameters
    p = np.linalg.solve(alpha, beta)

    # Calculate fit and chisq per point
    fit =  np.dot(A,p).T
    resid = xy - fit
    chisq = np.einsum("ki,ijk,kj->k",resid,invC,resid)

    # Put mean back into position
    p[:2] += xyMean

    return p, fit, chisq, alpha

def fit5d(indices,cat, time_sep=3, chisqClip=11., parallax_prior=0.15, 
            mjd_ref=57388.0,minPts=5,
            ra0=None,dec0=None):
    '''Execute 5d fit, with outlier rejection, on entries
    in the catalog at the rows specified by `indices`.
    Input catalog units are degrees but all parallax
    results come in arcsec / yrs.
    The catalog needs to have columns 
    `XI, ETA, ERRAWIN_WORLD, ERRBWIN_WORLD, ERRTHETAWIN_j2000,`
    `MJD`, `PAR_XI`, `PAR_ETA`
    `chisqClip`:  parameter gives clipping threshold
        for a 2d measurement (default value is p~0.004).
    `parallax_prior`: sigma for prior on parallax
    `mjd_ref` is epoch date for positions, defaults to 2016.0
    `err_floor` is sigma of additional (circular) error to add ###
    Returns:
    `p`: best-fit 5d parameters
    `alpha`: inverse-covariance matrix for fit
    `chisq`: total for fit
    `dof`: total for fit
    `nClip`: number of points clipped 
    Returns `None` if there are insufficient data for a fit.'''

    exposureTable = pf.getdata(os.environ['DES_EXPOSURES'],1)

    degree = 3600.   # in arcsec
    day = 1./365.2425  # in years

    temp_cat = cat[indices]

    color = 0.6

    # Extract data from catalog
    xy = np.array( [temp_cat['XI'],temp_cat['ETA']]).T * degree
    t = (np.array( temp_cat['MJD'] ) - mjd_ref) * day
    par_xy = np.array( [temp_cat['PAR_XI'],temp_cat['PAR_ETA']]).T

    # Put covariance into matrix form
    a = np.array(temp_cat['ERRAWIN_WORLD']) * degree
    b = np.array(temp_cat['ERRBWIN_WORLD']) * degree

    # a = np.hypot(a,0.1)
    # b = np.hypot(b,0.1)
    
    turb_aa = np.array(temp_cat['TURBERRA']) * 3600**2
    turb_bb = np.array(temp_cat['TURBERRB']) * 3600**2
    turb_ab = np.array(temp_cat['TURBERRAB']) * 3600**2

    turb_ee = turb_aa - turb_bb

    np.seterr(divide='ignore',invalid='ignore')

    turb_pa = 0.5 * np.arctan(2*np.divide(turb_ab , turb_ee))
    turb_pa[np.isnan(turb_pa)] = 0
    turb_sig_aa = 0.5 * (turb_aa + turb_bb \
        - np.sqrt(turb_ee**2 + 4*turb_ab**2))
    turb_sig_bb = 0.5 * (turb_aa + turb_bb \
        + np.sqrt(turb_ee**2 + 4*turb_ab**2))

    pa = np.array(temp_cat['ERRTHETAWIN_J2000']) \
        * np.pi / 180.  # in radians
    # Convert to cov
    ee = a*a - b*b
    cov_xy = np.array( [a*a+b*b + ee*np.cos(pa),
                        a*a+b*b - ee*np.cos(pa),
                        ee*np.sin(pa)]).T / 2.

    turb_cov_xy = np.array([turb_sig_aa+turb_sig_bb+turb_ee*np.cos(turb_pa),
                            turb_sig_aa+turb_sig_bb-turb_ee*np.cos(turb_pa),
                            turb_ee*np.sin(turb_pa)]).T 

    cov_xy = cov_xy + turb_cov_xy

    nClip = 0
    clips = []
    
    # Begin fit/clip loop
    while (xy.shape[0] >= minPts 
           and len(np.unique([round(i) for i in t]))>=time_sep): 
        p, fit, chisq, alpha = singleFit(xy, cov_xy, t, par_xy,
                                         parallax_prior=parallax_prior)
        # See if anything is clipped
        if xy.shape[0] > minPts and np.max(chisq)>chisqClip:
            iClip = np.argmax(chisq)
            clips += [indices[iClip]]
            temp_cat.remove_row(iClip)
            indices = np.delete(indices, iClip)
            
            t = np.delete(t, iClip)
            xy = np.delete(xy, iClip, axis=0)
            cov_xy = np.delete(cov_xy, iClip, axis=0)
            par_xy = np.delete(par_xy, iClip, axis=0)
            nClip = nClip + 1
        else:
            # Fit is finished
            chisqTotal = np.sum(chisq)
            dof = 2*xy.shape[0] - 5
            if chisqTotal/dof < chisqClip and (max(t)-min(t))>time_sep:

                g_flux = temp_cat[temp_cat['BAND']=='g']['FLUX_AUTO']
                g_mag = np.median(temp_cat[temp_cat['BAND']=='g']['ZEROPOINT'] \
                    -2.5*np.log10(g_flux))
                r_flux = temp_cat[temp_cat['BAND']=='r']['FLUX_AUTO']
                r_mag = np.median(temp_cat[temp_cat['BAND']=='r']['ZEROPOINT'] \
                    -2.5*np.log10(r_flux))
                i_flux = temp_cat[temp_cat['BAND']=='i']['FLUX_AUTO']
                i_mag = np.median(temp_cat[temp_cat['BAND']=='i']['ZEROPOINT'] \
                    -2.5*np.log10(i_flux))
                z_flux = temp_cat[temp_cat['BAND']=='z']['FLUX_AUTO']
                z_mag = np.median(temp_cat[temp_cat['BAND']=='z']['ZEROPOINT'] \
                    -2.5*np.log10(z_flux))
                Y_flux = temp_cat[temp_cat['BAND']=='Y']['FLUX_AUTO']
                Y_mag = np.median(temp_cat[temp_cat['BAND']=='Y']['ZEROPOINT'] \
                    -2.5*np.log10(Y_flux))

                if len(temp_cat[temp_cat['BAND']=='g']['FLUX_AUTO'])!=0 \
                    and len(temp_cat[temp_cat['BAND']=='i']['FLUX_AUTO'])!=0:

                    color = g_mag - i_mag

                if color!=0.6 and ra0!=None and dec0!=None :

                    temp_cat.remove_columns(['XI','ETA','MJD','PAR_XI','PAR_ETA'])

                    projectCatalog(temp_cat, \
                        ra0,dec0, \
                        exposureTable,color)

                    xy = np.array( [temp_cat['XI'],temp_cat['ETA']]).T * degree
                    t = (np.array( temp_cat['MJD'] ) - mjd_ref) * day
                    par_xy = np.array( [temp_cat['PAR_XI'],temp_cat['PAR_ETA']]).T

                    p, fit, chisq, alpha = singleFit(xy, cov_xy, t, par_xy,
                                                    parallax_prior=parallax_prior)

                g_n = len(temp_cat[temp_cat['BAND']=='g'])
                r_n = len(temp_cat[temp_cat['BAND']=='r'])
                i_n = len(temp_cat[temp_cat['BAND']=='i'])
                z_n = len(temp_cat[temp_cat['BAND']=='z'])
                Y_n = len(temp_cat[temp_cat['BAND']=='Y'])

                g_spread = np.median(temp_cat['SPREAD_MODEL'][temp_cat['BAND']=='g'])
                r_spread = np.median(temp_cat['SPREAD_MODEL'][temp_cat['BAND']=='r'])
                i_spread = np.median(temp_cat['SPREAD_MODEL'][temp_cat['BAND']=='i'])
                z_spread = np.median(temp_cat['SPREAD_MODEL'][temp_cat['BAND']=='z'])
                Y_spread = np.median(temp_cat['SPREAD_MODEL'][temp_cat['BAND']=='Y'])

                g_spread_err = np.median(temp_cat['SPREADERR_MODEL'][temp_cat['BAND']=='g'])
                r_spread_err = np.median(temp_cat['SPREADERR_MODEL'][temp_cat['BAND']=='r'])
                i_spread_err = np.median(temp_cat['SPREADERR_MODEL'][temp_cat['BAND']=='i'])
                z_spread_err = np.median(temp_cat['SPREADERR_MODEL'][temp_cat['BAND']=='z'])
                Y_spread_err = np.median(temp_cat['SPREADERR_MODEL'][temp_cat['BAND']=='Y'])
                

                

                return p, alpha, chisqTotal, t, dof, nClip, \
                    np.array(indices,dtype=np.int32), np.array(clips,dtype=np.int32), \
                    g_mag, r_mag, i_mag, z_mag, Y_mag, color, \
                    g_n, r_n, i_n, z_n, Y_n, \
                    g_spread, r_spread, i_spread, z_spread, Y_spread, \
                    g_spread_err, r_spread_err, i_spread_err, z_spread_err, Y_spread_err
            else:
                return None

    # Get here if fit did not start with enough points
    return None