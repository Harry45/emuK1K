# Author: Arrykrishna Mootoovaloo
# Collaborators: Prof. Alan Heavens, Prof. Andrew Jaffe, Dr. Florent Leclercq
# Email : arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk
# Affiliation : Imperial Centre for Inference and Cosmology
# Status : Under Development (adapted from KiDS-1000 likelihood)

"""All the settings for the priors"""

import numpy as np

# mean for the shifts
MU = [0.000, 0.002, 0.013, 0.011, -0.006]

# load the covariance matrix for the shifts in n(z) distributions
COV = np.loadtxt('data/SOM_covariance.asc')

# this is factor by which the covariance matrix is multiplied
# in the KiDS-1000 analysis, factor = 4
# however, other factors, for example, 2, 3 have been tested and reported in the paper
factor = 4.0
# -----------------------------------------------------------------------------

# Priors
# specs are according to the scipy.stats. See documentation:
# https://docs.scipy.org/doc/scipy/reference/stats.html

# cosmo is used for building the emulator as well
# and nuisance and shifts are used together with cosmo in parameter inference
# we have a multivariate normal distribution for the shifts

# For example, if we want uniform prior between 1.0 and 5.0, then
# it is specified by loc and loc + scale, where scale=4.0
# distribution = scipy.stats.uniform(1.0, 4.0)

cosmo = {

    'omega_cdm': {'distribution': 'uniform', 'specs': [0.051, 0.204]},
    'omega_b': {'distribution': 'uniform', 'specs': [0.019, 0.007]},
    'S_8': {'distribution': 'uniform', 'specs': [0.10, 1.30]},
    'n_s': {'distribution': 'uniform', 'specs': [0.84, 0.26]},
    'h': {'distribution': 'uniform', 'specs': [0.64, 0.18]}
}

nuisance = {
    'A_bary': {'distribution': 'uniform', 'specs': [2.00, 1.31]},
    'A_IA': {'distribution': 'uniform', 'specs': [-6.00, 12.0]}
}

shifts = {'deltas': {'distribution': 'multivariate_normal', 'specs': [MU, factor * COV]}}

# parameter names (cosmology)
cosmo_names = ['omega_cdm', 'omega_b', 'S_8', 'n_s', 'h']

# names for the nuisance parameters
nuisance_names = ['A_bary', 'A_IA']

# the calculation of the log-pdf takes in a dictionary
# hence we split the vector into the different elements
shifts_names = ['d1', 'd2', 'd3', 'd4', 'd5']

# all the parameter names
marg_names = cosmo_names + shifts_names + nuisance_names

# starting point for sampling
mean = {'omega_cdm': 0.153,
        'omega_b': 0.0225,
        'S_8': 0.763,
        'n_s': 1.062,
        'h': 0.755,
        'd1': 0.0,
        'd2': 0.002,
        'd3': 0.013,
        'd4': 0.011,
        'd5': -0.006,
        'A_bary': 2.5,
        'A_IA': 0.557
        }

# step sizes for Monte Carlo (EMCEE) run

# Experiment 1
# eps = [0.02, 0.0004, 0.03, 0.02, 0.02, 0.005, 0.003, 0.003, 0.003, 0.003, 0.15, 0.25]

# Standard Deviations from Results of Experiment 1
# [0.04642438, 0.00200991, 0.02405769, 0.04971041, 0.04547013, 0.00951432, 0.01109829, 0.010743 , 0.00845959, 0.00970721, 0.22027285, 0.41482501]

# Experiment 2
eps = [0.01, 5E-4, 0.01, 0.01, 0.01, 1E-4, 1E-4, 1E-4, 1E-4, 1E-4, 0.1, 0.5]

# number of walkers
n_walkers = 24

# number of samples
n_samples = 5 #10000
