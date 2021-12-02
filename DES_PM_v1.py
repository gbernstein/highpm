
#!/usr/bin/env python
# Establishes proper motions for stars in the DES footprint


from __future__ import print_function
import numpy as np
import astropy.io.fits as pf
from astropy.table import vstack,QTable
import astropy.units as u
import sys
import os
import argparse
from scipy import spatial
import scipy.spatial as sps
import scipy.ndimage
from scipy.ndimage.filters import gaussian_filter
from multiprocessing import Pool

def read_cat_header(filename):
    # Takes catalog produced by getFinalcutTile.py and returns the 
    # corresponding header 
    hdul = pf.open(filename)
    header = hdul[1].header
    hdul.close()
    return header

def read_cat_data(filename):
    # Takes catalog produced by getFinalcutTile.py and returns the 
    # corresponding data table
    hdul = pf.open(filename)
    cat_data = hdul[1].data
    hdul.close()
    cat = QTable(cat_data)
    return cat


def clean_cat(catname):
    # Removes detections with "FLAGS"!=0 and "IMAFLAGS!=0"
    cleancat = catname[np.logical_and(catname["FLAGS"==0], \
                                      catname["IMAFLAGS_ISO"]==0)]
    return cleancat

# =============================================================================
# These functions are from Pedro's code
# =============================================================================

def find_friend(data, length, cores=None):
    '''
    Uses scipy's kDTree functionalities to find all friends 
    (points within a given distance of each other)
    '''
    if cores == None:
        return data.query_ball_tree(data, length)
    else:
        return data.query_ball_point(data.data, length, workers=cores)


def friends_of_friends(list_friends):
    '''
    Main function of the code. Using a mutating list of friends, 
    finds all sets that overlap with each other and joins them together. 
    New version, for loop changed for set operations, significantly faster
    '''
    todo = set(range(len(list_friends)))
    result = []
    while len(todo) > 0:
        i = todo.pop()
        new_set = set([i])
        fresh_friends = set(list_friends[i])
        fresh_friends.remove(i)
        while len(fresh_friends) > 0:
            next_friend = fresh_friends.pop()
            new_set.add(next_friend)
            if next_friend in todo:
                fof = set(list_friends[next_friend])
                todo.remove(next_friend)
                fof = fof - new_set
                fresh_friends = fresh_friends | fof 
        result.append(list(new_set))

    return result

# =============================================================================

def multithreader(func,lol,cores=1):
    # Distributes a function acting on a list of lists to multiple cores
    pool = Pool(cores)
    ls_out = pool.map(func,lol)
    pool.close()
    pool.join()
    return ls_out


def multi_fit5d(fitter,detections_groups,cores=1):
    # Multithreaded application of a 5D fitter to groups of detections
    pm_list = multithreader(fitter,detections_groups,cores)

    # Remove fits that return None
    clean_pm_list = [i for i in pm_list if i!=None]
    
    # Convert list to array and discard list
    clean_pm_arr = np.array(clean_pm_list,dtype=object)
    clean_pm_list.clear()

    return clean_pm_arr


def gnomonic_plate2sky(xi,eta,ra0,dec0):
    # Converts plate coordinates (xi,eta)[deg] to sky coordinates (ra,dec)[deg]
    # assuming (xi,eta) are the gnomonic projections of (ra,dec)
    # centered at (ra0,dec0)

    xi_rad = xi * np.pi/180.
    eta_rad = eta * np.pi/180.

    ra0_rad = ra0 * np.pi/180.
    dec0_rad = dec0 * np.pi/180.

    rho = np.hypot(xi_rad,eta_rad)
    c = np.arctan(rho)

    ra_rad = ra0_rad+np.arctan(xi_rad*np.sin(c)
            /(rho*np.cos(dec0_rad)*np.cos(c)-eta_rad*np.sin(dec0_rad)*np.sin(c)))

    dec_rad = np.arcsin(np.cos(c)*np.sin(dec0_rad)
        +eta_rad*np.sin(c)*np.cos(dec0_rad)/rho)

    ra = ra_rad * 180./np.pi
    dec = dec_rad * 180./np.pi

    return ra,dec


def output_fits(pm_arr,filename,mtype,fittype='fit5d',outputname=None):
    # Writes a fits table containing data from the proper motion fitting

    if fittype!='fit5d':
        raise RuntimeError('This fitter is not yet supported')

    if fittype=='fit5d':

        column_names = [
            'idx',
            'mtype',
            'ra',
            'ra_err',
            'dec',
            'dec_err',
            'pm',
            'pmra',
            'pmra_err',
            'pmdec',
            'pmdec_err',
            'parallax',
            'parallax_err',
        #     'alpha',
            'chisqTotal',
        #     't',
            'dof',
            'nClip',
        #     'members',
        #     'clipped'
        ]

        idx   = range(len(pm_arr))
        ls_mtype = [mtype]*len(pm_arr)

        p_fits = np.vstack(pm_arr[:,0])
        cov = np.array([np.linalg.inv(i) for i in pm_arr[:,1]])

        xi = np.array((p_fits[:,0]*u.arcsec).to(u.deg))
        eta = np.array((p_fits[:,1]*u.arcsec).to(u.deg))

        header = read_cat_header(filename)

        ra0 = header['RA0']
        dec0 = header['DEC0']

        ra,dec = gnomonic_plate2sky(xi,eta,ra0,dec0)*u.deg

        ra_err = (np.sqrt(cov[:,0,0])*u.arcsec).to(u.deg)
        dec_err = (np.sqrt(cov[:,1,1])*u.arcsec).to(u.deg)

        pmra    = (p_fits[:,2]*u.arcsec/u.year).to(u.mas/u.year)
        pmra_err = (np.sqrt(cov[:,2,2])*u.arcsec/u.year).to(u.mas/u.year)
        pmdec   = (p_fits[:,3]*u.arcsec/u.year).to(u.mas/u.year)
        pmdec_err = (np.sqrt(cov[:,3,3])*u.arcsec/u.year).to(u.mas/u.year)
        pm = np.hypot(pmra,pmdec)
        parallax = p_fits[:,4]
        parallax_err = np.sqrt(cov[:,4,4])

        alpha = pm_arr[:,1]
        chisqTotal = np.array(pm_arr[:,2],dtype=np.float64)
        t = pm_arr[:,3]
        dof = np.array(pm_arr[:,4],dtype=np.float64)
        nClip = np.array(pm_arr[:,5],dtype=np.float64)
        members = pm_arr[:,6]
        clipped = pm_arr[:,7]

        data = [
            idx,
            ls_mtype,
            ra,
            ra_err,
            dec,
            dec_err,
            pm,
            pmra,
            pmra_err,
            pmdec,
            pmdec_err,
            parallax,
            parallax_err,
        #     alpha,
            chisqTotal,
        #     t,
            dof,
            nClip,
        #     members,
        #     clipped
        ]

        tbl = QTable(names=column_names,data=data)
        
        detection_column_names = [
            'idx',
            'detections',
            'clipped'
        ]

        detection_tbl = QTable(names=detection_column_names,
            dtype=['i4','i4','i4'])
        for i in idx:
            clipbool = len(members[i])*[False] + len(clipped[i])*[True]
            detections = np.hstack((members[i],clipped[i]))
            temp_tbl = QTable(data=[[i]*len(detections),detections,clipbool], 
                names=detection_column_names,
                dtype=['i4','i4','i4'])
            
            detection_tbl = vstack([detection_tbl,temp_tbl])

        if outputname==None:
            tbl.write(filename[:-4]+'_'+mtype+'_movers.fits',overwrite=True)
            detection_tbl.write(filename[:-4]+'_'+mtype+'_detections.fits', \
                overwrite=True)
        else:
            tbl.write(outputname+'.fits',overwrite=True)
            detection_tbl.write(outputname+'_detections.fits', \
                overwrite=True)
    return

# =============================================================================
# Modest mover algorithm
# =============================================================================


def normal_dist(x , mean , sd):
    prob_density = 1/(np.sqrt(2*np.pi)*sd) * np.exp(-0.5*((x-mean)/sd)**2)
    return prob_density

def centroid(data):
        h,w = np.shape(data)   
        x = np.arange(0,w)
        y = np.arange(0,h)

        X,Y = np.meshgrid(x,y)

        cx = np.sum(X*data)/np.sum(data)
        cy = np.sum(Y*data)/np.sum(data)

        return cx,cy
    
def peak_finder(velocity_image):
    velocity_image_thresh = velocity_image.copy()
    velocity_image_blur = gaussian_filter(velocity_image_thresh,sigma=2)
    velocity_image_thresh[velocity_image_blur \
        < 0.85*np.max(velocity_image_blur)]=0
        
    #now find the objects
    labeled_image, number_of_objects = scipy.ndimage.label(
        velocity_image_thresh
        )

    peak_slices = scipy.ndimage.find_objects(labeled_image)

    centroids = np.zeros((0,2))

    for peak_slice in peak_slices:
        dy,dx  = peak_slice
        x,y = dx.start, dy.start
        cx,cy = centroid(velocity_image_thresh[peak_slice])
        centroids = np.vstack((centroids, \
            np.array([((x+cx)-50)/25,((y+cy)-30)/20])))
        
    return centroids

def filter_list(L):
        return [x for x in L if not any(set(x)<=set(y) \
            for y in L if x is not y)]
    
def object_finder(sample,t_i,pos_i,pos_err_i):
    velocity_image = np.zeros((0,100))
    for v in np.linspace(-1.5,1.5,60):
        sample_gaussians = np.zeros((0,100))
        pos = np.linspace(-2,2,100)
        for i in range(len(pos_i)):
            norm_i = normal_dist(pos,pos_i[i]+t_i[i]*v,pos_err_i[i])
            sample_gaussians = np.vstack((sample_gaussians,norm_i))
        sample_distribution = np.sum(sample_gaussians,axis=0)
        velocity_image = np.vstack((velocity_image,sample_distribution))
        
        
    centroids = peak_finder(velocity_image)

    obj_list = []
    for i in centroids:
        residuals = abs(pos_i + i[1]*t_i - i[0])
        obj_list += [[sample[j] for j in range(len(sample)) \
            if residuals[j]<3*pos_err_i[j]]]
        
    return centroids,obj_list
    
    
def modest_movers(sample):
    sample = list(sample)
    if len(sample)>4:
        t_0 = np.median(cat['MJD'][sample])
        t_i = (cat['MJD'][sample] - t_0)/365.25

        # =============== RA =========================

        ra_0 = np.mean(cat['XI'][sample])

        ra_i = (cat['XI'][sample] - ra_0)
        ra_err_i = 3600*(cat['ERRAWIN_WORLD'][sample])


        ra_centroids,ra_obj_list = object_finder(sample,t_i,ra_i,ra_err_i)

        # =============== DEC ========================

        dec_0 = np.mean(cat['ETA'][sample])

        dec_i = (cat['ETA'][sample] - dec_0)
        dec_err_i = 3600*(cat['ERRBWIN_WORLD'][sample])


        dec_centroids,dec_obj_list = object_finder(sample,t_i,dec_i,dec_err_i)


        obj_list = []
        for i in ra_obj_list:
            for j in dec_obj_list:
                obj_list += [list(set(i).intersection(set(j)))]

        obj_list = filter_list(obj_list)

        return obj_list
    
    else:
        return None

def modest_fitter(cat,linklength=1./3600.,cores=1):
    xi = cat['XI']
    eta = cat['ETA']

    modest_tree = arborist(xi,eta)
    modest_friends = find_friend(modest_tree,linklength,cores)
    modest_groups = friends_of_friends(modest_friends)

    modest_pm_ls = multithreader(modest_movers,modest_groups,cores)

    modest_pm_ls = [i for i in modest_pm_ls if i!=None]
    modest_pm_obj = []
    for i in modest_pm_ls:
        for j in i:
            modest_pm_obj += [j]

    modest_pm_obj = [i for i in modest_pm_obj if len(i)>=5]

    modest_pm_arr = multi_fit5d(fit5d,modest_pm_obj,cores)

    return modest_pm_arr

# =============================================================================


# =============================================================================
# Fast mover algorithm
# =============================================================================

def posvel(pair):
    first,second=pair
    time_sep = (cat['MJD'][second]-cat['MJD'][first])/365.25
    pair_posvel=np.zeros((7))
    if time_sep > 0.7:
        avg_ra   = np.mean(cat['XI'][[first,second]])
        avg_dec  = np.mean(cat['ETA'][[first,second]])
        vel_ra   = (cat['XI'][second]-cat['XI'][first])/time_sep
        vel_dec  = (cat['ETA'][second]-cat['ETA'][first])/time_sep
        
        pair_posvel = np.array([avg_ra,avg_dec,
                                vel_ra,vel_dec,
                                time_sep,first,second])
    return pair_posvel

def fast_movers(cat,pairlength=60./3600.,linklength=12./3600.,cores=1, \
    min_pairs=10):
    xi = cat['XI']
    eta = cat['ETA']

    fast_tree = arborist(xi,eta)
    print('fast_tree built...')
    fast_pairs = fast_tree.query_pairs(r=pairlength)
    print('fast_pairs generated...')
    fast_posvel = multithreader(posvel,fast_pairs,cores)
    print('fast_posvel completed...')
    fast_posvel = np.vstack(fast_posvel)
    print('fast_posvel stacked...')
    fast_posvel = fast_posvel[~np.all(fast_posvel == 0, axis=1)]
    print('same detection pairs removed...')
    pm_keep = np.logical_and(
        np.hypot(fast_posvel[:,2], fast_posvel[:,3])>1/3600,
        np.hypot(fast_posvel[:,2], fast_posvel[:,3])<linklength
        )
    print('keepers found...')
    fast_posvel = fast_posvel[pm_keep]
    print('keepers kept...')
    fast_data = np.hstack((fast_posvel[:,:2],(6)*fast_posvel[:,2:4]))
    print('prepped for 4d tree...')
    fast_4dtree = sps.cKDTree(fast_data)
    print('4d tree planted...')
    fast_friends = find_friend(fast_4dtree,linklength,cores)
    print('fast_friends found...')
    fast_objects = friends_of_friends(fast_friends)
    print('friends of friends found...')
    hipm_object_sets = [i for i in fast_objects if len(i)>min_pairs]
    print('high pm objects cleaned...')
    fast_obj = []
    for i in hipm_object_sets:
        temp_object = []
        for j in i:
            detection_pair = [int(fast_posvel[j,-2]),int(fast_posvel[j,-1])]
            temp_object += detection_pair
        fast_obj += [list(np.unique(temp_object))]
    print('fast objects grouped...')
    fast_obj = np.unique(fast_obj)
    print('unique fast objects complete!')
    return fast_obj
    

# =============================================================================

def arborist(xi,eta):
    tree = spatial.KDTree(np.array([xi,eta]).transpose())
    return tree

# =============================================================================
# Slow mover algorithm
# =============================================================================

def slow_movers(cat,linklength=0.1/3600.,cores=1):
    xi = cat['XI']
    eta = cat['ETA']

    slow_tree = arborist(xi,eta)
    slow_friends = find_friend(slow_tree,linklength,cores)
    slow_groups = friends_of_friends(slow_friends)
    return slow_groups

# =============================================================================



def detections_for_removal(pm_arr,pm_lim,pm_err_lim):

    p_fits = np.vstack(pm_arr[:,0])
    cov = np.array([np.linalg.inv(i) for i in pm_arr[:,1]])

    pmra    = (p_fits[:,2]*u.arcsec/u.year).to(u.mas/u.year)
    pmra_err = (cov[:,2,2]*u.arcsec/u.year).to(u.mas/u.year)
    pmdec   = (p_fits[:,3]*u.arcsec/u.year).to(u.mas/u.year)
    pmdec_err = (cov[:,3,3]*u.arcsec/u.year).to(u.mas/u.year)

    removals = np.concatenate([pm_arr[i][-2] for i in range(len(pm_arr)) \
        if np.logical_and(abs(pmra_err[i])<pm_err_lim*u.mas/u.year, \
            abs(pmdec_err[i])<pm_err_lim*u.mas/u.year) \
        and np.hypot(pmra[i],pmdec[i])<pm_lim*u.mas/u.year])

    return removals

# =============================================================================
# 5D Fitter
# =============================================================================

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

def fit5d(indices, time_sep=0.7, chisqClip=11., parallax_prior=0.15, 
            mjd_ref=57388.0, err_floor = 0.01):
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
    pa = np.array(cat['ERRTHETAWIN_J2000'][indices]) \
        * np.pi / 180.  # in radians
    # Convert to cov
    ee = a*a - b*b
    cov_xy = np.array( [a*a+b*b + ee*np.cos(pa),
                        a*a+b*b - ee*np.cos(pa),
                        ee*np.sin(pa)]).T / 2.
    nClip = 0
    clips = []
    minPts = 5
   
    # Begin fit/clip loop
    while xy.shape[0] >= minPts and max(t)-min(t)>time_sep:
        p, fit, chisq, alpha = singleFit(xy, cov_xy, t, par_xy,
                                         parallax_prior=parallax_prior)
        # See if anything is clipped
        if xy.shape[0] > minPts and np.max(chisq)>chisqClip:
            iClip = np.argmax(chisq)
            clips += [indices[iClip]]
            indices = np.delete(indices, iClip)
            xy = np.delete(xy, iClip, axis=0)
            t = np.delete(t, iClip)
            par_xy = np.delete(par_xy, iClip, axis=0)
            cov_xy = np.delete(cov_xy, iClip, axis=0)
            nClip = nClip + 1
        else:
            # Fit is finished
            chisqTotal = np.sum(chisq)
            dof = 2*xy.shape[0] - 5
            if chisqTotal/dof < chisqClip and (max(t)-min(t))>time_sep:
                return p, alpha, chisqTotal, t, dof, nClip, \
                    np.array(indices), np.array(clips)
            else:
                return None

    # Get here if fit did not start with enough points
    return None

# =============================================================================

if __name__=='__main__':
    help = "Still need to write the help section"

    my_cores=24

    if len(sys.argv)==2:
        if sys.argv[1]=='-h' or sys.argv[1]=='--help':
            print(help)
            sys.exit(1)
        catname = sys.argv[1]
    else:
        print(help)
        sys.exit(1)

    cat = read_cat_data(catname)

    slow_detection_groups = slow_movers(cat,cores=my_cores)

    slow_pm_arr = multi_fit5d(fit5d,slow_detection_groups,cores=my_cores)

    print('Writing slow movers...')
    output_fits(slow_pm_arr,catname,'slow')

    slow_detections = detections_for_removal(slow_pm_arr,100,10)

    cat.remove_rows(slow_detections)
    cat.write('pre_modest_' + catname,format='fits',overwrite=True)

    modest_pm_arr = modest_fitter(cat,cores=my_cores)

    print('Writing modest movers...')
    output_fits(modest_pm_arr,catname,'modest')

    modest_detections = detections_for_removal(modest_pm_arr,1000,10)

    cat.remove_rows(modest_detections)
    cat.write('pre_fast_' + catname,format='fits',overwrite=True)

    fast_detection_groups = fast_movers(cat,cores=my_cores)

    fast_pm_arr = multi_fit5d(fit5d,fast_detection_groups,cores=my_cores)

    print('Writing fast movers...')
    output_fits(fast_pm_arr,catname,'fast')

    print('Done!')


    sys.exit(0)
