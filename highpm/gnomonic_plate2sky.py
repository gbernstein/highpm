import numpy as np

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
