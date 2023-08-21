import setemu as se
import numpy as np
from classy import Class

class_args = {'z_max_pk': 2.0,
                           'output': 'mPk',
                           'non linear': 'halofit',
                           'P_k_max_h/Mpc': se.k_max_h_by_Mpc,
                        #    'nonlinear_min_k_max': se.min_k_max,
                        #    'halofit_k_per_decade': se.halofit_k_per_decade,
                        #    'halofit_sigma_precision': se.halofit_sigma_precision,
                        #    'k_pivot': se.k_pivot,
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