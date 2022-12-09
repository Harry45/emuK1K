# Author: Arrykrishna Mootoovaloo
# Collaborators: Prof. Alan Heavens, Prof. Andrew Jaffe, Dr. Florent Leclercq
# Email : arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk
# Affiliation : Imperial Centre for Inference and Cosmology
# Status : Under Development (adapted from KiDS-1000 likelihood)

"""Script for sampling the posterior distribution"""

import numpy as np
import emcee

# our scripts
import inference.likelihood as lk
import utils.common as uc
import utils.helpers as hp
import setpriors as sp


class mcmc(lk.sampling_dist):

    def __init__(self, nz_method='som', stats='PeeE', save=True):

        # the likelihood routine
        lk.sampling_dist.__init__(self, nz_method, stats, save)

    def logliketest(self, parameters):
        # inputs are dictionaries
        params = uc.mk_dict(sp.marg_names, parameters)

        # the cosmology dictionary
        cosmo_dict = uc.subset_dict(params, sp.cosmo_names)

        # the nuisance dictionary
        nuisa_dict = uc.subset_dict(params, sp.nuisance_names)

        # the shifts dictionary
        shift_dict = uc.subset_dict(params, sp.shifts_names)

        # calculate the log-likelihood
        log_like = self.loglike(cosmo_dict, shift_dict, nuisa_dict)

        return log_like

    def logpost(self, parameters: np.ndarray) -> float:

        # inputs are dictionaries
        params = uc.mk_dict(sp.marg_names, parameters)

        # the cosmology dictionary
        cosmo_dict = uc.subset_dict(params, sp.cosmo_names)

        # the nuisance dictionary
        nuisa_dict = uc.subset_dict(params, sp.nuisance_names)

        # the shifts dictionary
        shift_dict = uc.subset_dict(params, sp.shifts_names)

        # calculate the log-likelihood
        log_like = self.loglike(cosmo_dict, shift_dict, nuisa_dict)

        # calculate the log-prior
        log_prior = self.logprior(cosmo_dict, nuisa_dict)

        if not np.isfinite(log_prior):
            return -np.inf

        else:
            lpost = log_like + log_prior
            return lpost

    def posterior_sampling(self, sampler_name: str = None) -> None:
        '''
        Perform posterior sampling

        Arguments:
            starting_point (np.ndarray) : the starting point for sampling

            sampler_name (str): if sa sampler name has been specified, the samples will be saved in the samples/ folder

        Returns:
            sampler (EMCEE module) :
        '''

        # we need the values from the dictionary to start sampling
        start = np.array([sp.mean[k] for k in sp.marg_names])

        # get the step size from the setting file
        eps = np.array(sp.eps)

        # the number of dimension
        ndim = len(start)

        # perturb the initial position
        pos = [start + eps * np.random.randn(ndim) for i in range(sp.n_walkers)]

        sampler = emcee.EnsembleSampler(sp.n_walkers, ndim, self.logpost)

        sampler.run_mcmc(pos, sp.n_samples, progress=True)

        # delete these objects - otherwise pickle complains
        del self.data_block
        del self.theory_module

        if sampler_name:
            hp.store_pkl_file(sampler, 'samples', sampler_name)

        return sampler
