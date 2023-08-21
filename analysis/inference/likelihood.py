# Author: Arrykrishna Mootoovaloo
# Collaborators: Prof. Alan Heavens, Prof. Andrew Jaffe, Dr. Florent Leclercq
# Email : arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk
# Affiliation : Imperial Centre for Inference and Cosmology
# Status : Under Development (adapted from KiDS-1000 likelihood)

"""Script for calculating the likelihood"""

import numpy as np
from scipy.linalg import solve_triangular

# our scripts
import cosmology.configurations as cg
import cosmology.weaklensing as wl
import utils.common as uc
import setemu as se
import setpriors as sp
import inference.priors as pr


class sampling_dist(wl.spectra, cg.setup):

    def __init__(self, nz_method='som', stats='PeeE', save=True):
        """Module to calculate the log-likelihood

        Args:
            nz_method (str, optional): the n(z) method to use. Defaults to 'som'.
            stats (str, optional): the statistics to use. Defaults to 'PeeE'.
            save (bool, optional): save the data vector and covariance matrix. Defaults to True.
        """

        # initialise module for the basic configurations
        cg.setup.__init__(self, stats)

        # get the theory module
        self.theory_module = cg.setup.config_theory(self)

        # get the basic quantities (band powers, covariance and scale cut modules)
        self.data_block = cg.setup.apply_scale_cut(self, save)

        # initialise module for calculating the weak lensing power spectra
        wl.spectra.__init__(self, nz_method)

        # prior due to the cosmology
        self.cosmo_prior = pr.all_entities(sp.cosmo)

        # prior due to the nuisance parameters
        self.nuisa_prior = pr.all_entities(sp.nuisance)

        # prior due to the shifts parameters
        self.shift_prior = pr.all_entities(sp.shifts)

        print('LOADED')

    def logprior(self, cosmology: dict, nuisance: dict) -> float:

        lp = pr.log_prod_pdf(self.cosmo_prior, cosmology)

        lp += pr.log_prod_pdf(self.nuisa_prior, nuisance)

        # lp += pr.log_prod_pdf(self.shift_prior, shifts)

        return lp

    def theory_calculation(self, total_shear: np.ndarray, keys: list) -> np.ndarray:
        """Calculates the theoretical band powers using the cosmosis routine

        Args:
            total_shear (np.ndarray): the total shear power spectra (auto- and cross-)
            keys (list): the keys (str) for each power spectrum, bin_i_j

        Returns:
            np.ndarray: the expected (theoretical) band powers
        """

        # create input dict for datablock:
        input_theory = {}

        input_theory['shear_cl'] = {'nbin_a': se.nzbins, 'nbin_b': se.nzbins}

        # now add the vals for 'ell' and 'bin_1_1', 'bin_2_1', ... 'bin_n_n'
        input_theory['shear_cl'].update({'ell': self.ells})

        # add the keys and the total shear power spectrum
        input_theory['shear_cl'].update(dict(zip(keys, total_shear)))

        datablock = cg.dict_to_datablock(input_theory)

        # excecute the data block
        self.theory_module.execute(datablock)

        # silence the scale_cuts module during likelihood evaluations
        uc.block_print()

        # apply the scale cuts to the shear power spectra
        self.data_block['scm'].execute(datablock)

        # re-enable print statements again
        uc.enable_print()

        # get the theory vector
        theory_vec = np.asarray(datablock['likelihood_bp', 'theory'])

        return theory_vec

    def bp_theory_calc(self, cosmology: dict, shifts: dict, nuisance: dict) -> np.ndarray:
        """Calculates the theoretical band powers given the set of input parameters

        Args:
            cosmology (dict): the cosmological parameters (omega_cdm, omega_b, S_8, n_s, h)
            shifts (dict): the shifts in the SOM n(z) distribution (d1, d2, d3, d4, d5)
            nuisance (dict): the two additional parameters (A_IA, A_bary)

        Returns:
            np.ndarray: the predicted band powers
        """

        # calculate the total shear
        shear_ps, keys = self.total_shear(cosmology, shifts, nuisance)

        # get the theoretical band powers
        theory_vec = self.theory_calculation(shear_ps, keys)

        return theory_vec

    def loglike(self, cosmology: dict, shifts: dict, nuisance: dict) -> float:
        """Calculates the log-likelihood value

        Args:
            cosmology (dict): the cosmological parameters (omega_cdm, omega_b, S_8, n_s, h)
            shifts (dict): the shifts in the SOM n(z) distribution (d1, d2, d3, d4, d5)
            nuisance (dict): the two additional parameters (A_IA, A_bary)

        Returns:
            float: the log-likelihood value
        """

        log_prior = self.logprior(cosmology, nuisance)

        log_prior_shifts = pr.log_prod_pdf(self.shift_prior, shifts)

        if not np.isfinite(log_prior):
            chi_sqr = 1E32

        else:

            # calculate the predicted band powers based on the cosmology and nuisance
            theory = self.bp_theory_calc(cosmology, shifts, nuisance)

            # the difference vector
            difference = theory - self.data_block['x']

            # check if all elements in the difference vector is finite
            if np.any(np.isinf(difference)) or np.any(np.isnan(difference)):
                chi_sqr = 1E32

            # calculate chi_square
            else:
                y = solve_triangular(self.data_block['cholesky'], difference, lower=True)
                chi_sqr = np.sum(y * y)

        return -0.5 * chi_sqr + log_prior_shifts
