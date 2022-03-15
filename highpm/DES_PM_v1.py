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
from astropy.table import Column
from pixmappy import DESMaps
from pmfit import *
from friends_of_friends import *
from fits_writer import *
from cat_reader import *
import tqdm



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

        ra_i = 3600*(cat['XI'][sample] - ra_0)
        ra_err_i = 3600*(cat['ERRAWIN_WORLD'][sample])
        ra_err_i = np.hypot(ra_err_i,cat['TURBERRA'][sample])


        ra_centroids,ra_obj_list = object_finder(sample,t_i,ra_i,ra_err_i)

        # =============== DEC ========================

        dec_0 = np.mean(cat['ETA'][sample])

        dec_i = 3600*(cat['ETA'][sample] - dec_0)
        dec_err_i = 3600*(cat['ERRBWIN_WORLD'][sample])
        dec_err_i = np.hypot(dec_err_i,cat['TURBERRB'][sample])


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
    mjd_ref=57388.0
    first,second=pair
    time_sep = (cat['MJD'][second]-cat['MJD'][first])/365.2425
    pair_posvel=np.zeros((7))
    if time_sep > 0.7:
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
    min_pairs=10):
    xi = cat['XI']
    eta = cat['ETA']

    pairlength = 5*linklength

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
        np.hypot(fast_posvel[:,2], fast_posvel[:,3])<18/3600
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
    fast_obj = np.unique(fast_obj)
    print('unique fast objects complete!')
    return fast_obj


def mkimg(t_i,pos_i,pos_err_i,w,h,res):
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

    for _ in range(10):#range(len(fast_cat)):
        
        if len(temp_cat)>5:
            sample = np.arange(len(temp_cat))
            t_0 = np.median(temp_cat['MJD'])
            t_i = (temp_cat['MJD'] - t_0)/365.25
            
            a = np.array(temp_cat['ERRAWIN_WORLD']) * 3600.
            b = np.array(temp_cat['ERRBWIN_WORLD']) * 3600.

            turb_aa = np.array(temp_cat['TURBERRA'])
            turb_bb = np.array(temp_cat['TURBERRB'])
            turb_ab = np.array(temp_cat['TURBERRAB'])

            turb_ee = turb_aa - turb_bb

            np.seterr(invalid='ignore')

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
            
            
            # =============== RA =========================


            ra_i = 3600*(temp_cat['XI'] - ra_0)
            ra_err_i = np.sqrt(a*a+b*b + ee*np.cos(pa) 
                            + turb_sig_aa+turb_sig_bb+turb_ee*np.cos(turb_pa))

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
            dec_err_i = np.sqrt(a*a+b*b - ee*np.cos(pa) 
                            + turb_sig_aa+turb_sig_bb-turb_ee*np.cos(turb_pa))

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

def multithreader(func,lol,cat,cores=1):
    ls_out = []
    partial_func = partial(func,cat=cat)
    with Pool(processes=cores) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(partial_func,lol,
                    chunksize=1000),
                total=len(lol)):
            ls_out.append(_)
            pass
        # ls_out = pool.map(partial_func,lol)
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

    cat.write('NEW_'+catname,format='fits',overwrite=True)
    cat_copy = cat.copy()

    slow_detection_groups = slow_movers(cat,cores=my_cores)

    slow_pm_arr = multi_fit5d(fit5d,slow_detection_groups,cat,cores=my_cores)

    print('Writing slow movers...')
    slow_tbl = output_fits(slow_pm_arr,catname,'slow')

    slow_detections = detections_for_removal(slow_pm_arr,100,10)

    cat.remove_rows(slow_detections)
    cat.write('pre_modest_' + catname,format='fits',overwrite=True)

    modest_pm_arr = modest_fitter(cat,cores=my_cores)

    print('Writing modest movers...')
    modest_tbl = output_fits(modest_pm_arr,catname,'modest')

    modest_detections = detections_for_removal(modest_pm_arr,1000,10)

    cat.remove_rows(modest_detections)
    cat.write('pre_fast_' + catname,format='fits',overwrite=True)

    fast_detection_groups = fast_movers(cat,cores=my_cores)

    fast_pm_arr = multi_fit5d(fit5d,fast_detection_groups,cat,cores=my_cores)

    print('Writing fast movers...')
    fast_tbl = output_fits(fast_pm_arr,catname,'fast')

    # fast_tbl = fast_tbl[fast_tbl['pm']>100*u.mas/u.yr]

    ls_out = []
    part_fast_checker = partial(fast_checker,cat=cat,fast_cat=fast_tbl)
    with Pool(processes=my_cores) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(part_fast_checker,
            np.arange(len(fast_tbl)),
                    chunksize=10),
                total=len(fast_tbl)):
            ls_out.append(_)
            pass
    pool.close()
    pool.join()

    fast_checked_lol = filter_list([i for j in ls_out for i in j])

    fast_checked_arr = multi_fit5d(fit5d,fast_checked_lol,cat_copy,cores=my_cores)

    print('Writing checked fast movers...')
    fast_checked_tbl = output_fits(fast_checked_arr,catname,'fast_checked')



    print('Done!')


    sys.exit(0)
