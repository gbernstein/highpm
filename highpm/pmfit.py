''' 
Fit full five-parameter PM/parallax model to observations.
'''

## ??? Need to gaurd against bad fits,  perhaps a v weak prior on PM

import numpy as np

def singleFit(xy,cov_xy, t,par_xy, parallax_prior=None):
    '''Fit 5-parameter model to stellar observations, where:
    `xy` is Nx2 array of observations of the star, in arcsec
    `cov` is Nx3 array giving (sig^2_x, sig^2_y, cov_xy) for each
    `t`  is time of observation of each (in yrs)
    `par_xy` are -1*(projected earth position components), in AU
    `parallax_prior` is prior sigma for parallax, in arcsec/yr

    Returns:
    `soln`  the 5-param covariance (x0, y0, vx, vy, parallax)
    `cov`   covariance matrix of this
    `chisq` vector of chisq of each observation.'''

    # Remove means from xy for numerical stability
    xyMean = np.mean(xy, axis=0)
    xy -= xyMean
    # Build matrices
    npts = xy.shape[0]
    one = np.ones(npts, dtype=float)
    zero = np.zeros(npts, dtype=float)

    # Build 2 x N x 5 matrix of coefficients
    A = np.array( [[one, zero, t, zero, par_xy[:,0]],
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

                   
def fit5d(cat, indices, chisqClip=11., parallax_prior=0.15, mjd_ref=57388.0,
              err_floor = 0.01):
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

    degree = 3600.   # in arcsec
    day = 1./365.2425  # in years

    # Extract data from catalog
    xy = np.array( [cat['XI'][indices],cat['ETA'][indices]]).T * degree
    t = (np.array( cat['MJD'][indices] ) - mjd_ref) * day
    par_xy = np.array( [cat['PAR_XI'][indices],cat['PAR_ETA'][indices]]).T

    # Put covariance into matrix form
    a = np.array(cat['ERRAWIN_WORLD'][indices]) * degree
    b = np.array(cat['ERRBWIN_WORLD'][indices]) * degree
    a = np.hypot(a,err_floor)
    b = np.hypot(b,err_floor)
    pa = np.array(cat['ERRTHETAWIN_J2000'][indices]) * np.pi / 180.  # in radians
    # Convert to cov
    ee = a*a - b*b
    cov_xy = np.array( [a*a+b*b + ee*np.cos(pa),
                        a*a+b*b - ee*np.cos(pa),
                        ee*np.sin(pa)]).T / 2.
    nClip = 0
    minPts = 4
   
    # Begin fit/clip loop
    while xy.shape[0] >= minPts:
        p, fit, chisq, alpha = singleFit(xy, cov_xy, t, par_xy,
                                         parallax_prior=parallax_prior)
        # See if anything is clipped
        if xy.shape[0] > minPts and np.max(chisq)>chisqClip:
            iClip = np.argmax(chisq)
            xy = np.delete(xy, iClip, axis=0)
            t = np.delete(t, iClip)
            par_xy = np.delete(par_xy, iClip, axis=0)
            cov_xy = np.delete(cov_xy, iClip, axis=0)
            nClip = nClip + 1
        else:
            # Fit is finished
            chisqTotal = np.sum(chisq)
            dof = 2*xy.shape[0] - 5
            return p, alpha, chisqTotal, dof, nClip

    # Get here if fit did not start with enough points
    return None
