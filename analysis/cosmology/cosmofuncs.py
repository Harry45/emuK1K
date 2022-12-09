# Author: Arrykrishna Mootoovaloo
# Collaborators: Prof. Alan Heavens, Prof. Andrew Jaffe, Dr. Florent Leclercq
# Email : arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk
# Affiliation : Imperial Centre for Inference and Cosmology
# Status : Under Development (adapted from KiDS-1000 likelihood)

"""Some basic functions for spectra and likelihood calculation"""

import numpy as np

# our scripts
import setemu as se


def one_dim_index(b1: int, b2: int, nzbins: int) -> int:
    """This function is used to convert 2D sums over the two indices (b1, b2) of an N*N symmetric matrix into 1D sums
    over one index with N(N+1)/2 possible values.

    Args: b1 (int): the first index b2 (int): the second index nzbins (int): number of redshift bins (for KiDS-1000,
        nzbins = 5)

    Returns: int: a single index (based on the convention adopted by KiDS-1000 group)
    """

    if b1 <= b2:
        return b2 + nzbins * b1 - (b1 * (b1 + 1)) // 2
    else:
        return b1 + nzbins * b2 - (b2 * (b2 + 1)) // 2


def sigma_eight(cosmo: dict) -> float:
    """Calculates sigma_8 from a dictionary containing the cosmological parameters

    Args:
        cosmo (dict): A dictionary containing omega_cdm, omega_b, S_8, h

    Returns:
        float: the value of sigma_8
    """

    # omega_cdm
    cdm = cosmo['omega_cdm']

    # omega_b
    bar = cosmo['omega_b']

    # omega_nu
    nu = cosmo['m_ncdm'] / 93.14

    # omega_matter (contains factor h**2)
    omega_matter = cdm + bar + nu

    # actual omega_matter
    omega_matter /= cosmo['h']**2

    return cosmo['S_8'] * np.sqrt(0.3 / omega_matter)


def bar_fed(k: np.ndarray, z: np.ndarray, a_bary: float = 0.0) -> np.ndarray:
    """Fitting formula for baryon feedback following equation 10 and Table 2 from J. Harnois-Deraps et al. 2014 (arXiv.1407.4301)

    Args:
        k (np.ndarray): the wavevector
        z (np.ndarray): the redshift
        a_bary (float, optional): the free amplitude for baryon feedback . Defaults to 0.0.

    Returns:
        np.ndarray: b^2(k,z): bias squared
    """

    k = np.atleast_2d(k).T

    z = np.atleast_2d(z)

    bm = se.baryon_model

    # k is expected in h/Mpc and is divided in log by this unit...
    x_wav = np.log10(k)

    # calculate a
    a_factor = 1. / (1. + z)

    # a squared
    a_sqr = a_factor * a_factor

    a_z = se.cst[bm]['A2'] * a_sqr + se.cst[bm]['A1'] * a_factor + se.cst[bm]['A0']
    b_z = se.cst[bm]['B2'] * a_sqr + se.cst[bm]['B1'] * a_factor + se.cst[bm]['B0']
    c_z = se.cst[bm]['C2'] * a_sqr + se.cst[bm]['C1'] * a_factor + se.cst[bm]['C0']
    d_z = se.cst[bm]['D2'] * a_sqr + se.cst[bm]['D1'] * a_factor + se.cst[bm]['D0']
    e_z = se.cst[bm]['E2'] * a_sqr + se.cst[bm]['E1'] * a_factor + se.cst[bm]['E0']

    # original formula:
    # bias_sqr = 1.-A_z*np.exp((B_z-C_z)**3)+D_z*x*np.exp(E_z*x)
    # original formula with a free amplitude A_bary:
    bias_sqr = 1. - a_bary * (a_z * np.exp((b_z * x_wav - c_z)**3) - d_z * x_wav * np.exp(e_z * x_wav))

    return bias_sqr


def get_factor_ia(quant: dict, redshift: np.ndarray, amplitude: float = 1.0, exponent: float = 0.0) -> np.ndarray:
    """Calculates F(chi) - equation 23 in Kohlinger et al. 2017.

    Args:
        quant (dict): a dictionary containingthe critical density, omega matter, linear groth rate, Hubble parameter
        redshift (np.ndarray): a vector for the redshift
        amplitude (float, optinonal): the amplitude due to intrinsic alignment Defaults to 1.0.
        exponent (float, optional): an exponential factor - not used in inference. Defaults to 0.0.

    Returns:
        np.ndarray: the evaluated function F(chi)
    """

    # critical density
    rc = quant['rc']

    # omega matter
    om = quant['omega_m']

    # linear growth rate
    lgr = quant['lgr']

    # Hubble parameter
    h = quant['small_h']

    # in Mpc^3 / M_sol
    const = 5E-14 / h**2

    # arbitrary convention
    redshift_0 = 0.3

    # additional term for the denominator (not in paper)
    denom = ((1. + redshift) / (1. + redshift_0))**exponent

    # factor = (-1. * amplitude * const * rc * om) / (lgr * denom)
    factor = (amplitude * const * rc * om) / (lgr * denom)

    return factor


def get_critical_density(small_h: float) -> float:
    """The critical density of the Universe at redshift 0.

    Args:
        small_h (float): the Hubble parameter

    Returns:
        float: the critical density at redshift zero
    """

    # Mpc to cm conversion
    mpc_cm = 3.08568025e24

    # Mass of Sun in grams
    mass_sun_g = 1.98892e33

    # Gravitational constant
    grav_const_mpc_mass_sun_s = mass_sun_g * (6.673e-8) / mpc_cm**3.

    # in s^-1 (to check definition of this)
    h_100_s = 100. / (mpc_cm * 1.0e-5)

    rho_crit_0 = 3. * (small_h * h_100_s)**2. / (8. * np.pi * grav_const_mpc_mass_sun_s)

    return rho_crit_0


def delete_module(module):
    """Delete Class module - accumulates memory unnecessarily

    Args:
        module (classy.Class): the Class module
    """

    module.struct_cleanup()

    module.empty()

    del module
