# Author: Arrykrishna Mootoovaloo
# Collaborators: Prof. Alan Heavens, Prof. Andrew Jaffe, Dr. Florent Leclercq
# Email : arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk
# Affiliation : Imperial Centre for Inference and Cosmology
# Status : Under Development

"""Non-linear matter power spectrum calculation (and other quantities) using CLASS"""

from typing import Tuple
import numpy as np
from classy import Class

# our scripts
import cosmology.redshift as cr
import cosmology.cosmofuncs as cf
import utils.common as uc
import setemu as se


class pkclass(cr.nzDist):

    def __init__(self, nz_method='som'):

        cr.nzDist.__init__(self, nz_method)

        # set up the basic arguments to pass to CLASS
        self.class_args = {'z_max_pk': self.nz['zmax'],
                           'output': 'mPk',
                           'non linear': 'halofit',
                           'P_k_max_h/Mpc': se.k_max_h_by_Mpc,
                           'nonlinear_min_k_max': se.min_k_max,
                           'halofit_k_per_decade': se.halofit_k_per_decade,
                           'halofit_sigma_precision': se.halofit_sigma_precision,
                           'k_pivot': se.k_pivot,
                           'sBBN file': se.bbn
                           }

        # the values of redshift
        self.z_grid = np.linspace(se.zmin, se.zmax, se.nz, endpoint=True)

        # the values of k
        self.k_grid = np.geomspace(se.k_min_h_by_Mpc, se.kmax, se.nk, endpoint=True)

    def class_compute(self, parameters: dict) -> object:
        """Use CLASS to compute the basic quantities

        Args: parameters (dict): The input cosmological parameters, with the following parameters for the KiDS-1000
            analysis: {omega_cdm, omega_b, S_8, n_s, h}. The S_8 value will be converted to sigma8 first, before passing
            it to CLASS, simply because CLASS, following the explanatory.ini file, takes as input the following, A_s,
            ln10^{10}A_s, sigma8.

        Returns: object: The CLASS module
        """

        # create a new dictionary with the cosmological parameter inputs
        cosmology = {**parameters, **se.neutrino}

        # Calculate sigma8 from the dictionary
        sigma8 = cf.sigma_eight(cosmology)

        # update the dictionary with the sigma8 value
        cosmology.update({'sigma8': sigma8})

        # CLASS does not accept S_8
        cosmology = uc.removekey(cosmology, 'S_8')

        # instantiate Class
        class_module = Class()

        # set cosmology
        class_module.set(cosmology)

        # set basic configurations for Class
        class_module.set(self.class_args)

        # compute the important quantities
        class_module.compute()

        return class_module

    def pk_nonlinear(self, parameters: dict, a_bary: float = 0.0) -> Tuple[np.ndarray, dict]:
        """Calculate the 3D matter power spectrum based on the emulator setting file and also returns some important
           calculated quantities

        Args:
            parameters (dict): inputs to calculate the matter power spectrum
            a_bary (float, optional): The value of the baryon feedback parameter. Defaults to 0.0.

        Returns:
            Tuple[np.ndarray, dict]: the 3D matter power spectrum (with baryon feedback) and some important quantities

        """

        # baryon feedback
        bf = cf.bar_fed(self.k_grid / parameters['h'], self.z_grid, a_bary)

        # Calculate the 3D matter power spectrum
        class_module = self.class_compute(parameters)

        # Get the Hubble parameter
        h = class_module.h()

        # Get power spectrum P(k=l/r,z(r)) from cosmological module
        pk_matter = np.zeros((se.nk, se.nz), 'float64')

        for k in range(se.nk):
            for z in range(se.nz):

                # get the matter power spectrum
                pk_matter[k, z] = class_module.pk(self.k_grid[k] * h, self.z_grid[z])

        # we can also compute other important quantities

        # Omega_matter
        omega_m = class_module.Omega_m()

        # h parameter
        small_h = class_module.h()

        # critical density
        rc = cf.get_critical_density(small_h)

        # derive the linear growth factor D(z)
        lgr = np.zeros_like(self.nz['z'])

        for iz, red in enumerate(self.nz['z']):

            # compute linear growth rate
            lgr[iz] = class_module.scale_independent_growth_factor(red)

            # normalise linear growth rate at redshift = 0
            lgr /= class_module.scale_independent_growth_factor(0.)

        # get distances from cosmo-module
        chi, dzdr = class_module.z_of_r(self.nz['z'])

        # numerical stability for chi
        chi += 1E-10

        # pre-factor A in weak lensing
        a_fact = (3. / 2.) * omega_m * small_h**2 / 2997.92458**2

        # important quantities
        quant = {}
        quant.update({'omega_m': omega_m})
        quant.update({'chi': chi})
        quant.update({'dzdr': dzdr})
        quant.update({'lgr': lgr})
        quant.update({'rc': rc})
        quant.update({'a_fact': a_fact})
        quant.update({'small_h': small_h})

        # clean class_module to prevent memory issue
        cf.delete_module(class_module)

        return bf * pk_matter, quant
