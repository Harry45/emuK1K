import setemu as se
import numpy as np
from classy import Class
import inference.likelihood as lk

class_args = {'z_max_pk': 2.0,
                           'output': 'mPk',
                           'non linear': 'halofit',
                           'P_k_max_h/Mpc': 1,
                           'nonlinear_min_k_max': se.min_k_max,
                           'halofit_k_per_decade': se.halofit_k_per_decade,
                           'halofit_sigma_precision': se.halofit_sigma_precision,
                           'k_pivot': se.k_pivot,
                           'sBBN file': se.bbn
                           }


cosmology = {'omega_cdm':0.12, 'omega_b': 0.020,
         'sigma8': 0.75, 'n_s': 0.97, 'h': 0.70}

# instantiate Class
class_module = Class()

# set cosmology
class_module.set(cosmology)

# set basic configurations for Class
class_module.set(class_args)

# compute the important quantities
class_module.compute()

print('pass')


testing = lk.sampling_dist('grouping', 'PeeE', True)
cosmo = {'omega_cdm':0.10802979609861982, 'omega_b': 0.02332493567194329, 'S_8': 0.815371910192108, 'n_s': 0.9704551766630282, 'h': 0.6887657808123486}
shifts = {'d1': 0.0, 'd2': 0.002, 'd3': 0.013, 'd4': 0.011, 'd5': -0.006}
nuisa = {'A_bary': 2.5, 'A_IA': 1.0}
print(testing.loglike(cosmo, shifts, nuisa))