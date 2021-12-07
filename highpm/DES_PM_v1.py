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
from scipy.ndimage.filters import gaussian_filter
from multiprocessing import Pool
from functools import partial
from pmfit import *
from friends_of_friends import *
from fits_writer import *
from cat_reader import *



def arborist(xi,eta):
    tree = spatial.KDTree(np.array([xi,eta]).transpose())
    return tree


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
    
    
def modest_movers(sample,cat):
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

    modest_pm_ls = multithreader(modest_movers,modest_groups,cat,cores)

    modest_pm_ls = [i for i in modest_pm_ls if i!=None]
    modest_pm_obj = []
    for i in modest_pm_ls:
        for j in i:
            modest_pm_obj += [j]

    modest_pm_obj = [i for i in modest_pm_obj if len(i)>=5]

    modest_pm_arr = multi_fit5d(fit5d,modest_pm_obj,cat,cores)

    return modest_pm_arr

# =============================================================================


# =============================================================================
# Fast mover algorithm
# =============================================================================

def posvel(pair,cat):
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
    fast_posvel = multithreader(posvel,fast_pairs,cat,cores)
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


# =============================================================================
# Multithreading Functions
# =============================================================================

def multithreader(func,lol,cat,cores=1):
    partial_func = partial(func,cat=cat)
    with Pool(processes=cores) as pool:
        ls_out = pool.map(partial_func,lol)
    pool.close()
    pool.join()
    return ls_out


def multi_fit5d(fitter,detections_groups,cat,cores=1):
    # Multithreaded application of a 5D fitter to groups of detections
    pm_list = multithreader(fitter,detections_groups,cat,cores)

    # Remove fits that return None
    clean_pm_list = [i for i in pm_list if i!=None]
    
    # Convert list to array and discard list
    clean_pm_arr = np.array(clean_pm_list,dtype=object)
    clean_pm_list.clear()

    return clean_pm_arr

# =============================================================================


if __name__=='__main__':
    help = "Still need to write the help section"

    my_cores=os.cpu_count()

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

    slow_pm_arr = multi_fit5d(fit5d,slow_detection_groups,cat,cores=my_cores)

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

    fast_pm_arr = multi_fit5d(fit5d,fast_detection_groups,cat,cores=my_cores)

    print('Writing fast movers...')
    output_fits(fast_pm_arr,catname,'fast')

    print('Done!')


    sys.exit(0)
