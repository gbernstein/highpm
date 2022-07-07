from time import time
from turtle import pos
import destnosim
import numpy as np 
import astropy.table as tb 
from astropy.time import Time
from astropy.coordinates import solar_system_ephemeris, EarthLocation
from astropy.coordinates import get_body
from astropy.coordinates import SkyCoord
from scipy.spatial import cKDTree
from itertools import chain

def F_ra(ra_star,R_earth,ra_sun,dec_ecliptic):
    F_ra = (R_earth * np.sin(ra_sun) 
        * np.cos(ra_star) * np.cos(dec_ecliptic) +
        R_earth * np.sin(ra_star) * np.cos(ra_sun))
    return F_ra

def F_dec(ra_star,R_earth,ra_sun,dec_ecliptic,dec_star):
    F_dec = R_earth * ((np.sin(dec_ecliptic) * np.cos(dec_star) 
        - np.cos(dec_ecliptic) * np.sin(ra_star) * np.sin(dec_star)) 
        * np.sin(ra_sun)
        - np.cos(ra_star) * np.sin(dec_star) * np.cos(ra_sun))
    return F_dec


def project_to_images(solution, survey, t_0):
	'''
	Projects all proper motion solutions to all DES images, and checks which ones fall inside a CCD in an approximate way
	Arguments:
	- solution: (n x 5) array with the 5D proper motion solution for n stars. Assumes order is (ra_0, dec_0, pm_ra, pm_dec, parallax). Values in degrees, mas/yr, 1/parsec (?)
	- survey: (destnosim.DES) survey object containing information of all exposures in DES
	- ra_0: (float, degrees) RA center of the tile
	- dec_0 (float, degrees) DEC center of the tile
	- t_0 (float, mjd) reference time for the proper motion solutions
	'''
	year = 365.2425
	ref_date = t_0
	deg2mas = 3600000
	
	theta_x = np.zeros((len(solution), len(survey)))
	theta_y = np.zeros((len(solution), len(survey)))

	x_Earth = survey.exp['observatory'][:,0]
	y_Earth = survey.exp['observatory'][:,1]
	times = (survey.exp['mjd_mid'] - t_0) / year 

	## project to all exposures	
	
	#theta_x = solution[:,0] + np.outer(solution[:,2], times) - np.outer(solution[:,4], x_Earth)
	#theta_y = solution[:,1] + np.outer(solution[:,3], times) - np.outer(solution[:,4], y_Earth)

	#ra, dec = invert_gnomonic(theta_x, theta_y, ra_0, dec_0)

	ones = np.ones_like(times)

	
	RA = solution[:,0]
	DEC = solution[:,1]

	t_ephem = survey.exp['mjd_mid']

	loc = EarthLocation.of_site('Cerro Tololo Interamerican Observatory')

	with solar_system_ephemeris.set('builtin'):
		sol = get_body('sun', Time(t_ephem,format='mjd'), loc) 

	t = (t_ephem-ref_date)/year

	params = np.array([np.repeat(RA,len(t_ephem)),
                    np.array(sol.distance),
                    np.array(sol.ra*np.pi/180),
                    np.array(sol.dec*np.pi/180),
                    np.repeat(DEC,len(t_ephem))],dtype=object)
	f_ra = np.zeros(len(times))
	f_dec= np.zeros(len(times))
	for i in range(len(times)):
		f_ra[i] = F_ra(params[0][i],params[1][i],params[2][i],params[3][i])
		f_dec[i]= F_dec(params[0][i],params[1][i],
            params[2][i],params[3][i],params[4][i])
        
	ra = np.outer(RA,ones) + np.outer(solution[:,2]/(np.cos(DEC*np.pi/180)*deg2mas),t) \
        + np.outer(solution[:,4],f_ra)/(deg2mas)
	dec = np.outer(DEC,ones)+np.outer(solution[:,3]/(deg2mas),t) \
        + np.outer(solution[:,4],f_dec)/(deg2mas)

	# ra  = np.outer(solution[:,0], ones) + np.outer(solution[:,2] / 3600 / 1000, times) - np.outer(solution[:,4]/3600000, x_Earth)
	# dec = np.outer(solution[:,1], ones) + np.outer(solution[:,3] / 3600 / 1000, times) - np.outer(solution[:,4]/3600000, y_Earth)
	## now turns this into a table to facilitate things

	tables = [] 

	#proper structure?
	for i in range(len(solution)):
		tab = tb.Table()
		tab['RA'] = ra[i]
		tab['DEC'] = dec[i]
		tab['MJD'] = 365.25*times + t_0
		tab['EXPNUM'] = survey.expnum
		tab['STARID'] = solution[i,5]
		tables.append(tab)

	tables = tb.vstack(tables)


	### adapted from DESTNOSIM's space rocks extension
	exp = tb.Table()
	exp['EXPNUM'] = np.array(survey.expnum)
	exp['RA_CENTER'] = np.array(survey.ra)
	exp['RA_CENTER'][exp['RA_CENTER'] > 180] -= 360
	exp['DEC_CENTER'] = np.array(survey.dec) 
	exp['BAND'] = np.array(survey.band)

	t = tb.join(tables, exp)
	t['DELTA'] = np.sqrt( (( t['RA'] - t['RA_CENTER']) * np.cos(t['DEC_CENTER'] * np.pi/180))**2 +  (t['DEC'] - t['DEC_CENTER'])**2)

	t = t[t['DELTA'] < 1.5]

	theta = destnosim.bulk_gnomonic(np.array(t['RA']), np.array(t['DEC']), np.array(t['RA_CENTER']), np.array(t['DEC_CENTER']))
	#rescale for kD tree
	theta[:,1] *= 2

	ccd_tree, ccd_keys = destnosim.create_ccdtree()

	tree = cKDTree(theta)
	# kD tree ccd checker
	inside_CCD = ccd_tree.query_ball_tree(tree, 0.149931 * 1.001, p = np.inf)
	
	if inside_CCD != None: 
		ccd_id = [len(inside_CCD[i])*[destnosim.ccdnums[ccd_keys[i]]] for i in range(len(inside_CCD)) if len(inside_CCD[i]) > 0]
		inside_CCD = np.array(list(chain(*inside_CCD)))
		if len(inside_CCD) > 0:
			ccdlist = list(chain(*ccd_id))
		else:
			print('No stars inside CCDs!')
			return None
	else:
		print('No stars inside CCDs!')
		return None
	
	t = t[inside_CCD]
	t['CCDNUM'] = ccdlist

	t.sort(['STARID', 'EXPNUM'])
	return t['RA', 'DEC','MJD','EXPNUM', 'CCDNUM', 'BAND', 'STARID']




def simulate_detections(positions, survey):
	'''
	After positions are derived, uses information from magnitudes to check which detections would
	be recovered by a given DES exposure and uses the combination of shot noise + turbulence
	to generate positional offsets for all detections
	Arguments:
	- positions (astropy.table.Table): table created by project_to_images
	- magnitudes (astropy.table.Table): table with magnitudes and same indexing as the proper motion solutions
	- survey: (destnosim.DES) survey object containing information of all exposures in DES
	'''

	tab = [] 
	unif = destnosim.Uniform(0,1)
	for i in np.unique(positions['EXPNUM']):
		t = positions[positions['EXPNUM'] == i]
		prob = survey[i].probDetection(t['MAG'])
		# print(prob[0],positions['BAND'][positions['EXPNUM']==i][0])

		rand = unif.sample(len(t))

		mask = prob > rand 

		t = t[mask]

		if len(t)==0:
			continue

		err,err_cov,errtheta = survey[i].samplePosError(t['SHOTNOISE'], len(t))

		t['ALPHAWIN_J2000'] = t['RA'] + err[:,0]
		t['DELTAWIN_J2000'] = t['DEC'] + err[:,1]
		# t['ERRAWIN_WORLD'] = err_cov[:,0,0]
		# t['ERRBWIN_WORLD'] = err_cov[:,1,1]
		t['ERRAWIN_WORLD'] = t['SHOTNOISE']
		t['ERRBWIN_WORLD'] = t['SHOTNOISE']
		t['ERRTHETAWIN_J2000'] = errtheta


		tab.append(t)

	st =  tb.vstack(tab)
	st.sort(['STARID', 'EXPNUM'])

	return st[
		'STARID',
		'RA', 
		'DEC', 
		'EXPNUM',
		'CCDNUM', 
		'ALPHAWIN_J2000', 
		'DELTAWIN_J2000', 
		'ERRAWIN_WORLD',
		'ERRBWIN_WORLD',
		'ERRTHETAWIN_J2000',
		'MJD',
		'BAND',
		'FLUX_PSF',
		'MAG'
		]




	