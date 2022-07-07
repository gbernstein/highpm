#!/usr/bin/env python
# Establishes proper motions for stars in the DES footprint

from __future__ import print_function
import numpy as np
import astropy.units as u
import sys
import os
import argparse
import scipy.spatial as sps
import scipy.ndimage
from scipy.ndimage import gaussian_filter
from multiprocessing import Pool
from functools import partial
from astropy.table import Column
from pixmappy import DESMaps
from pmfit import *
from friends_of_friends import *
from fits_writer import *
from cat_reader import *
import tqdm

n_detections = 5

def arborist(xi,eta):
    """
    Creates KD-Tree for possitions in tile coordinates.

    Parameters
    ---------
    xi : astropy.table.column.Column
        RA tile coordinates in pixels.
    eta : astropy.table.column.Column
        DEC tile coordinates in pixels.

    Returns
    -------
    scipy.spatial.kdtree.KDTree
        KD-Tree for all detections in a tile.

    """

    tree = spatial.KDTree(np.array([xi,eta]).transpose())
    return tree


def detections_for_removal(pm_arr,pm_lim,pm_err_lim):
    """
    Removes detections that have been included in a fit.

    Parameters
    ----------
    pm_arr : 
    pm_lim : 
    pm_err_lim : 

    Returns
    -------
    removals : numpy.ndarray
    """

    p_fits = np.vstack(pm_arr[:,0])
    cov = np.array([np.linalg.inv(i) for i in pm_arr[:,1]])

    pmra    = (p_fits[:,2]*u.arcsec/u.year).to(u.mas/u.year)
    pmra_err = (cov[:,2,2]*u.arcsec/u.year).to(u.mas/u.year)
    pmdec   = (p_fits[:,3]*u.arcsec/u.year).to(u.mas/u.year)
    pmdec_err = (cov[:,3,3]*u.arcsec/u.year).to(u.mas/u.year)

    removals = np.concatenate([pm_arr[i][6] for i in range(len(pm_arr)) \
        if np.logical_and(abs(pmra_err[i])<pm_err_lim*u.mas/u.year, \
            abs(pmdec_err[i])<pm_err_lim*u.mas/u.year) \
            and np.hypot(pmra[i],pmdec[i])<pm_lim*u.mas/u.year])
    return removals


# =============================================================================
# Slow mover algorithm
# =============================================================================

def slow_movers(cat,linklength=0.1/3600.,cores=1):
    """
    Identifies slow move candidates.

    Parameters
    ----------
    cat : 
    linklength : 
    cores :

    Returns
    -------
    slow_groups : list of lists
    """

    xi = cat['XI']
    eta = cat['ETA']

    slow_tree = arborist(xi,eta)
    slow_friends = find_friend(slow_tree,linklength,cores)
    slow_groups = friends_of_friends(slow_friends)
    slow_obj = [i for i in slow_groups if len(i)>=n_detections]
    return slow_obj

# =============================================================================


# =============================================================================
# Modest mover algorithm
# =============================================================================


def normal_dist(x , mean , sd):
    """
    Normal Distribution

    Parameters
    ----------
    x : 
    mean : 
    sd :

    Returns
    -------
    prob_density : numpy.ndarray
    
    """
    prob_density = 1/(np.sqrt(2*np.pi)*sd) * np.exp(-0.5*((x-mean)/sd)**2)
    return prob_density

def centroid(data):
    """
    Centroid finders.

    Parameters
    ----------
    data : 

    Returns
    -------
    cx : 
    cy : 
    
    """
    h,w = np.shape(data)   
    x = np.arange(0,w)
    y = np.arange(0,h)

    X,Y = np.meshgrid(x,y)

    cx = np.sum(X*data)/np.sum(data)
    cy = np.sum(Y*data)/np.sum(data)

    return cx,cy
    
def peak_finder(velocity_image):
    """
    Peak finding algorithm.

    Paramters
    ---------
    veloity_image : 

    Returns
    -------
    centroids : 
    
    """

    velocity_image_thresh = velocity_image.copy()
    velocity_image_blur = gaussian_filter(velocity_image_thresh,sigma=1)
    velocity_image_thresh[velocity_image_blur \
        < 0.9*np.max(velocity_image_blur)]=0
        
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
            np.array([((x+cx)-100)/50,((y+cy)-60)/40])))
        
    return centroids

def filter_list(L):
    """
    List filter removes lists that are subsets of other lists.

    Parameters
    ----------
    L : list of lists

    Returns
    -------
    Cleaned list
    
    """

    return [x for x in L if not any(set(x)<=set(y) \
        for y in L if x is not y)]

def filter_list1(L):
    sets={frozenset(e) for e in L}  
    us=[]
    for e in sets:
        if any(e < s for s in sets):
            continue
        else:
            us.append(list(e))   
    return us 
    
def object_finder(sample,t_i,pos_i,pos_err_i):
    """
    Identifies modest moving objects.

    Parameters
    ----------
    sample : 
    t_i : 
    pos_i : 
    pos_err_i : 

    Returns
    -------
    centroids : 
    obj_list : list of lists

    """
    
    velocity_image = np.zeros((0,200))
    for v in np.linspace(-1.5,1.5,120):
        sample_gaussians = np.zeros((0,200))
        pos = np.linspace(-2,2,200)
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
    
    
def modest_movers(sample,cat):
    """
    Modest mover algorithm.

    Parameters
    ----------
    sample : 
    cat : 
    
    Returns
    -------
    obj_list : list of lists
    """

    sample = list(sample)
    if len(sample)>=n_detections:
        t_0 = np.median(cat['MJD'][sample])
        t_i = (cat['MJD'][sample] - t_0)/365.25

        a = np.array(cat['ERRAWIN_WORLD'][sample]) * 3600.
        b = np.array(cat['ERRBWIN_WORLD'][sample]) * 3600.

        a = np.hypot(a,0.1)
        b = np.hypot(b,0.1)

        turb_aa = np.array(cat['TURBERRA'][sample]) * 3600**2
        turb_bb = np.array(cat['TURBERRB'][sample]) * 3600**2
        turb_ab = np.array(cat['TURBERRAB'][sample]) * 3600**2

        turb_ee = turb_aa - turb_bb

        np.seterr(divide='ignore',invalid='ignore')

        turb_pa = 0.5 * np.arctan(2*np.divide(turb_ab , turb_ee))
        turb_pa[np.isnan(turb_pa)] = 0
        turb_sig_aa = 0.5 * (turb_aa + turb_bb \
            - np.sqrt(turb_ee**2 + 4*turb_ab**2))
        turb_sig_bb = 0.5 * (turb_aa + turb_bb \
            + np.sqrt(turb_ee**2 + 4*turb_ab**2))

        pa = np.array(cat['ERRTHETAWIN_J2000'][sample]) \
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

        # =============== RA =========================

        ra_0 = np.mean(cat['XI'][sample])

        ra_i = 3600*(cat['XI'][sample] - ra_0)
        ra_err_i = np.sqrt(cov_xy[:,0])

        ra_centroids,ra_obj_list = object_finder(sample,t_i,ra_i,ra_err_i)

        # =============== DEC ========================

        dec_0 = np.mean(cat['ETA'][sample])

        dec_i = 3600*(cat['ETA'][sample] - dec_0)
        dec_err_i = np.sqrt(cov_xy[:,1])

        dec_centroids,dec_obj_list = object_finder(sample,t_i,dec_i,dec_err_i)

        obj_list = []
        for i in ra_obj_list:
            for j in dec_obj_list:
                temp_obj_list = list(set(i).intersection(set(j)))
                if len(temp_obj_list)>=n_detections:
                    obj_list += temp_obj_list

        obj_list = [list(obj_list)]
        # print(obj_list)
        obj_list = filter_list1(obj_list)
        # print(obj_list)
        return obj_list
    
    else:
        return None

def modest_fitter(cat,fitter,linklength=1./3600.,cores=1):
    """
    Modest mover fitting.

    Parameters
    ----------
    cat : 
    linklength : 
    cores : 

    Returns
    -------
    modest_pm_arr : 
    """

    xi = cat['XI']
    eta = cat['ETA']

    modest_tree = arborist(xi,eta)
    modest_friends = find_friend(modest_tree,linklength,cores)
    modest_groups = friends_of_friends(modest_friends)

    modest_groups = [i for i in modest_groups if len(i)>=n_detections]

    modest_pm_ls = multithreader(modest_movers,modest_groups,cat,
        cores,chunksize=int(len(modest_groups)/cores)+1)

    modest_pm_ls = [i for i in modest_pm_ls if i!=None]
    modest_pm_obj = []
    for i in modest_pm_ls:
        for j in i:
            modest_pm_obj += [j]

    modest_pm_obj = [i for i in modest_pm_obj if len(i)>=n_detections]

    modest_pm_arr = multi_fit5d(fitter,modest_pm_obj,cat,cores, \
        chunksize=int(len(modest_pm_obj)/cores)+1)

    return modest_pm_arr

# =============================================================================


# =============================================================================
# Fast mover algorithm
# =============================================================================

def posvel(pair,cat):
    """
    Calculates position and velocity for (xi,eta) pairs.   

    Parameters
    ----------
    pairs : 
    cat : 

    Returns
    -------
    pair_posvel : numpy.ndarray

    """

    mjd_ref=57388.0
    first,second=pair
    time_sep = (cat['MJD'][second]-cat['MJD'][first])/365.2425
    pair_posvel=np.zeros((7))
    if time_sep != 0:
        vel_ra   = (cat['XI'][second]-cat['XI'][first])/time_sep
        vel_dec  = (cat['ETA'][second]-cat['ETA'][first])/time_sep
        avg_ra   = np.mean(cat['XI'][[first,second]]) \
                    - vel_ra * (np.mean(cat['MJD'][[first,second]]) \
                    - mjd_ref)/365.2425
        avg_dec  = np.mean(cat['ETA'][[first,second]]) \
                    - vel_dec * (np.mean(cat['MJD'][[first,second]]) \
                    - mjd_ref)/365.2425
        
        pair_posvel = np.array([avg_ra,avg_dec,
                                vel_ra,vel_dec,
                                time_sep,first,second])
    return pair_posvel

def fast_movers(cat,linklength=2./3600.,cores=1, \
    min_pairs=5):
    """
    Fast movers algorithm.

    Parameters
    ----------
    cat : 
    linklength : 
    cores : 
    min_pairs : 

    Returns
    -------
    fast_obj : list of lists
    """

    xi = cat['XI']
    eta = cat['ETA']

    pairlength = 5*linklength

    fast_tree = arborist(xi,eta)
    print('fast_tree built...')
    fast_pairs = fast_tree.query_pairs(r=pairlength)
    print('fast_pairs generated...')
    fast_posvel = multithreader(posvel,fast_pairs,cat,cores,chunksize=100000)
    print('fast_posvel completed...')
    fast_posvel = np.vstack(fast_posvel)
    print('fast_posvel stacked...')
    fast_posvel = fast_posvel[~np.all(fast_posvel == 0, axis=1)]
    print('same detection pairs removed...')
    pm_keep = np.logical_and(
        np.hypot(fast_posvel[:,2], fast_posvel[:,3])>1/3600,
        np.hypot(fast_posvel[:,2], fast_posvel[:,3])<20/3600
        )
    print('keepers found...')
    fast_posvel = fast_posvel[pm_keep]
    print('keepers kept...')
    fast_data = np.hstack((fast_posvel[:,:2],(2)*fast_posvel[:,2:4]))
    print('prepped for 4d tree...')
    fast_4dtree = sps.cKDTree(fast_data)
    print('4d tree planted...')
    fast_friends = find_friend(fast_4dtree,linklength,cores)
    print(str(len(fast_friends))+' fast_friends found...')
    fast_objects = friends_of_friends(fast_friends)
    print(str(len(fast_objects))+' friends of friends found...')
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
    # fast_obj = np.unique(np.asanyarray(fast_obj,dtype=object))
    fast_obj = [i for i in fast_obj if len(i)>n_detections]
    fast_sets = [set(i) for i in fast_obj]
    print('fast objects culled...')
    fast_obj = [l for l,s in zip(fast_obj, fast_sets) if not any(s < other for other in fast_sets)]
    # fast_obj = list(set(tuple(x) for x in fast_obj))
    # print('fast objects prepped for filter...')
    # fast_obj = filter_list1(fast_obj)
    print('unique fast objects complete!')
    return fast_obj


def mkimg(t_i,pos_i,pos_err_i,w,h,res):
    """
    Creates position - velocity image.

    Parameters
    ----------
    t_i : 
    pos_i : 
    pos_err_i : 
    w : 
    h : 
    res : 

    Returns
    -------
    velocity_image : 
    """

    velocity_image = np.zeros((0,w))
    for v in np.linspace(-h/(2*res),h/(2*res),h):
        sample_gaussians = np.zeros((0,w))
        pos = np.linspace(-w/(2*res),w/(2*res),w)
        for i in range(len(pos_i)):
            norm_i = normal_dist(pos,pos_i[i]+t_i[i]*v,pos_err_i[i])
            sample_gaussians = np.vstack((sample_gaussians,norm_i))
        sample_distribution = np.sum(sample_gaussians,axis=0)
        velocity_image = np.vstack((velocity_image,sample_distribution))

    velocity_image = gaussian_filter(velocity_image,sigma=2)
    return velocity_image

def fast_checker(idx,cat,fast_cat):
    """
    Fast mover checking algorithm.

    Parameters
    ----------
    idx : 
    cat : 
    fast_cat :

    Returns
    -------
    obj_lol : list of lists

    """

    temp_idx_circ = (cat['XI']-fast_cat['xi'][idx])**2+(cat['ETA']-fast_cat['eta'][idx])**2 \
                <((3*u.yr/u.mas*fast_cat['pm'][idx]+1000)/3600000)**2
    temp_cat_circ = cat[temp_idx_circ]
    
    slope = fast_cat['pmdec'][idx]/fast_cat['pmra'][idx]
    temp_idx = np.abs(slope*(temp_cat_circ['XI']-fast_cat['xi'][idx])
                      +fast_cat['eta'][idx]-temp_cat_circ['ETA']) \
                      /np.sqrt(1+slope**2) < 2/3600.
    temp_cat = temp_cat_circ[temp_idx]
    
    pm = (fast_cat['pm'][idx]/1000).value
    
    
    
    res = 30
    if pm<0.1:
        res = 300

    w = round(2*3*res*(pm))+2*res
    h = round(2*res*(pm))+2*res

    ra_0 = np.mean(temp_cat['XI'])
    dec_0 = np.mean(temp_cat['ETA'])

    obj_lol = []

    for _ in range(10):
    
        if len(temp_cat)>n_detections:
            sample = np.arange(len(temp_cat))
            t_0 = np.median(temp_cat['MJD'])
            t_i = (temp_cat['MJD'] - t_0)/365.25
            
            a = np.array(temp_cat['ERRAWIN_WORLD']) * 3600.
            b = np.array(temp_cat['ERRBWIN_WORLD']) * 3600.

            a = np.hypot(a,0.1)
            b = np.hypot(b,0.1)

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
            
            
            # =============== RA =========================


            ra_i = 3600*(temp_cat['XI'] - ra_0)
            ra_err_i = np.sqrt(cov_xy[:,0])
            
            ra_velocity_image = mkimg(t_i,ra_i,ra_err_i,w,h,res)
            
            ra_centroids = np.array(np.unravel_index(np.argmax(ra_velocity_image, axis=None), 
                                            ra_velocity_image.shape))
            ra_centroids_1 = (ra_centroids[1]-w/2)/res
            ra_centroids_0 = (ra_centroids[0]-h/2)/res

            ra_residuals = abs(ra_i + ra_centroids_0*t_i - ra_centroids_1)
            ra_obj_list = [sample[j] for j in range(len(sample)) \
                if ra_residuals[j]<3*ra_err_i[j]]

            # =============== DEC ========================


            dec_i = 3600*(temp_cat['ETA'] - dec_0)
            dec_err_i = np.sqrt(cov_xy[:,1])

            dec_velocity_image = mkimg(t_i,dec_i,dec_err_i,w,h,res)

            dec_centroids = np.array(np.unravel_index(np.argmax(dec_velocity_image, axis=None), 
                                            dec_velocity_image.shape))
            dec_centroids_1 = (dec_centroids[1]-w/2)/res
            dec_centroids_0 = (dec_centroids[0]-h/2)/res  
            
            dec_residuals = abs(dec_i + dec_centroids_0*t_i - dec_centroids_1)
            dec_obj_list = [sample[j] for j in range(len(sample)) \
                if dec_residuals[j]<3*dec_err_i[j]]
            
            if len(ra_obj_list)==0 and len(dec_obj_list)==0:
                break
            
            obj_list = ra_obj_list
            if np.max(ra_velocity_image)/(len(ra_obj_list)+1) \
                <np.max(dec_velocity_image)/(len(dec_obj_list)+1) or \
                len(ra_obj_list)==0:
                obj_list = dec_obj_list
            
            obj_lol.append(list(temp_cat['ID'][obj_list]))

            temp_cat.remove_rows(obj_list)
            
                        
    return obj_lol

# =============================================================================


# =============================================================================
# Multithreading Functions
# =============================================================================

def multithreader(func,lol,cat,cores=1,chunksize=1000):
    """
    General purpose multithreader with progress bar.

    Parameters
    ----------
    func : 
    lol : 
    cat : 
    cores : 

    Returns
    -------
    ls_out : 

    """

    ls_out = []
    partial_func = partial(func,cat=cat)
    with Pool(processes=cores) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(partial_func,lol,
                    chunksize=chunksize),
                total=len(lol)):
            ls_out.append(_)
            pass
        # ls_out = pool.map(partial_func,lol)
    pool.close()
    pool.join()
    return ls_out


def multi_fit5d(fitter,detections_groups,cat,cores=1,chunksize=1000):
    """
    5-dimensional fitting multithreader.

    Parameters
    ----------
    fitter : 
    detections_groups : 
    cat : 
    cores :

    Returns
    -------
    clean_pm_arr : 

    """

    # Multithreaded application of a 5D fitter to groups of detections
    pm_list = multithreader(fitter,detections_groups,cat,cores,chunksize)

    # Remove fits that return None
    clean_pm_list = [i for i in pm_list if i!=None]
    
    # Convert list to array and discard list
    clean_pm_arr = np.array(clean_pm_list,dtype=object)
    clean_pm_list.clear()

    return clean_pm_arr

# =============================================================================


if __name__=='__main__':
    help = "Still need to write the help section"

    my_cores=22 # os.cpu_count()

    if len(sys.argv)==2:
        if sys.argv[1]=='-h' or sys.argv[1]=='--help':
            print(help)
            sys.exit(1)
        catname = sys.argv[1]
    else:
        print(help)
        sys.exit(1)

    header = read_cat_header(catname)

    cat = read_cat_data(catname)

    cat = clean_cat(cat)

    maps = DESMaps()

    tile_exposures = np.unique(cat['EXPNUM'])
    turb = np.array([maps.getCovariance(i) for i in tile_exposures])
    turb_a = turb[:,0,0]/(1000*3600)**2
    turb_b = turb[:,1,1]/(1000*3600)**2
    turb_ab = turb[:,0,1]/(1000*3600)**2
   
    
    turberr_a = np.zeros(len(cat))
    turberr_b = np.zeros(len(cat))
    turberr_ab = np.zeros(len(cat))
    for i in range(len(tile_exposures)):
        turberr_a[np.argwhere(cat['EXPNUM']==tile_exposures[i])] \
            = turb_a[i]
        turberr_b[np.argwhere(cat['EXPNUM']==tile_exposures[i])] \
            = turb_b[i]
        turberr_ab[np.argwhere(cat['EXPNUM']==tile_exposures[i])] \
            = turb_ab[i]

    turberr_a_col = Column(name='TURBERRA',data=turberr_a)
    turberr_b_col = Column(name='TURBERRB',data=turberr_b)
    turberr_ab_col = Column(name='TURBERRAB',data=turberr_ab)
    
    cat.add_column(turberr_a_col)
    cat.add_column(turberr_b_col)
    cat.add_column(turberr_ab_col)

    idx_col = Column(name='ID',data=range(len(cat)))
    cat.add_column(idx_col)

    zeropoint = read_cat_data("/home/vwetzell/git_repos/highpm/highpm/data/zeropoint1.fits")
    zp_dict = dict(zip(list(zip(
        zeropoint['CCDNUM'],
        zeropoint['EXPNUM'])),
        zip(zeropoint['MAG_ZERO'],zeropoint['SIGMA_MAG_ZERO']
        )))

    zp = []
    zp_err = []
    for i in zip(cat['CCDNUM'],cat['EXPNUM']):
        zp_tup = zp_dict[i]
        zp += [zp_tup[0]]
        zp_err += [zp_tup[1]]

    zeropoint_col = Column(name='ZEROPOINT',data=zp)
    zeropointerr_col = Column(name='SIGMA_ZEROPOINT_ERR',data=zp_err)

    cat.add_column(zeropoint_col)
    cat.add_column(zeropointerr_col)

    cat_copy = cat.copy()

    del tile_exposures
    del turb
    del turb_a
    del turb_b
    del turb_ab
    del turberr_a
    del turberr_b
    del turberr_ab
    del turberr_a_col
    del turberr_b_col
    del turberr_ab_col
    del idx_col
    del zeropoint
    del zp_dict
    del zp
    del zeropoint_col
    del zeropointerr_col

    slow_detection_groups = slow_movers(cat,cores=my_cores)

    part_fit5d = partial(fit5d, \
        ra0=header['RA0'],dec0=header['DEC0'])

    slow_pm_arr = multi_fit5d(part_fit5d,slow_detection_groups,cat,
        cores=my_cores,chunksize=int(len(slow_detection_groups)/my_cores)+1)

    if len(slow_pm_arr)!=0:
        print('Writing slow movers...')
        slow_tbl = output_fits(slow_pm_arr,catname,'slow')

        slow_detections = detections_for_removal(slow_pm_arr,100,10)

        cat.remove_rows(slow_detections)
    cat.write('pre_modest_' + catname,format='fits',overwrite=True)

    modest_pm_arr = modest_fitter(cat,part_fit5d,cores=my_cores)

    if len(modest_pm_arr)!=0:
        print('Writing modest movers...')
        modest_tbl = output_fits(modest_pm_arr,catname,'modest')

        modest_detections = detections_for_removal(modest_pm_arr,1000,10)

        cat.remove_rows(modest_detections)
    cat.write('pre_fast_' + catname,format='fits',overwrite=True)

    fast_detection_groups = fast_movers(cat,cores=my_cores)

    fast_pm_arr = multi_fit5d(part_fit5d,fast_detection_groups,cat,
        cores=my_cores,chunksize=int(len(fast_detection_groups)/my_cores)+1)

    
    if len(fast_pm_arr)!=0:
        print('Writing fast movers...')
        fast_tbl = output_fits(fast_pm_arr,catname,'fast')

    # fast_tbl = fast_tbl[fast_tbl['pm']>100*u.mas/u.yr]

    ls_out = []
    part_fast_checker = partial(fast_checker,cat=cat_copy,fast_cat=fast_tbl)
    with Pool(processes=my_cores) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(part_fast_checker,
            np.arange(len(fast_tbl)),
                    chunksize=int(len(fast_tbl)/my_cores)+1),
                total=len(fast_tbl)):
            ls_out.append(_)
            pass
    pool.close()
    pool.join()

    fast_checked_lol = [i for j in ls_out for i in j]

    fast_checked_arr = multi_fit5d(part_fit5d,fast_checked_lol,cat_copy,
        cores=my_cores,chunksize=int(len(fast_checked_lol)/my_cores)+1)

    
    if len(fast_checked_arr)!=0:
        print('Writing checked fast movers...')
        fast_checked_tbl = output_fits(fast_checked_arr,catname,'fast_checked')

    cat_copy.write('NEW_'+catname,format='fits',overwrite=True)

    print('Done!')


    sys.exit(0)
