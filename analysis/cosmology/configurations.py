# Author: Arrykrishna Mootoovaloo
# Collaborators: Prof. Alan Heavens, Prof. Andrew Jaffe, Dr. Florent Leclercq
# Email : arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk
# Affiliation : Imperial Centre for Inference and Cosmology
# Status : Under Development (adapted from KiDS-1000 likelihood)

"""Base configurations for the likelihood"""

import os
from scipy.linalg import cholesky
import cosmosis.runtime.module as crm
import cosmosis.datablock

# our scripts
import setemu as se
import utils.helpers as hp


def dict_to_datablock(dictionary: dict) -> list:
    """Convert dictionary to the data block format for cosmosis.

    Args:
        dictionary (dict): A dictionary with the settings.

    Returns:
        list: A data block generated in the cosmosis format.
    """

    block = cosmosis.datablock.DataBlock()
    for section in dictionary.keys():
        for name, value in dictionary[section].items():
            block[section, name] = value
    return block


class setup(object):

    def __init__(self, stats: str = 'PeeE'):
        """Setup the basic configurations for the data, theory and covariance.

        Args:
            stats (str, optional): The statistics to use, for example, band powers. Defaults to 'PeeE'.
        """

        # the stats to use - PeeE
        self.stats = stats

    def config_theory(self):
        """Configure the base setup in order to calculate the theory vector.

        Returns:
            cosmosis module: The theory is calculated using the cosmosis module.
        """

        # Initialize the BandPowers module from CosmoSIS:
        config_theory = {'bandpowers': {'theta_min': se.theta_min,
                                        'theta_max': se.theta_max,
                                        'l_min': se.ell_bin_min,
                                        'l_max': se.ell_bin_max,
                                        'nBands': se.nbins,
                                        'type': 'cosmic_shear_e',
                                        'Apodise': int(se.apodise),
                                        'Delta_x': se.delta_x,
                                        'Response_function_type': se.response_function,
                                        'Analytic': int(se.analytic),
                                        'input_section_name': 'shear_cl',
                                        'Output_FolderName': os.path.join(se.data_dir, 'bp_outputs')
                                        }}

        if hasattr(se, 'ell_bin_file'):
            config_theory.update({'l_min_l_max_file': se.ell_bin_file})

        print('Creating theory module')
        # needs to point down to one of the '.so' files in '../kcap/cosebis/':
        fname = os.path.join(se.kcap_dir, 'cosebis/libbandpower.so')

        # set up the theory module
        theory_module = crm.Module(module_name='bandpowers', file_path=fname)
        theory_module.setup(dict_to_datablock(config_theory))
        return theory_module

    def apply_scale_cut(self, save: bool = True) -> dict:
        """Here, we apply the scale cuts and generate some of the input quantities (data, covariance) for the likelihood calculation.

        Args:
            save (bool, optional): Save the generated band powers and the covariance matrix. Defaults to True.

        Returns:
            dict: A dictionary with the following keys: scm, x, cov, cholesky where
            - scm refers to the scale cut module by cosmosis
            - x refers to the band power data vector
            - cov refers to the data covariance matrix
            - cholesky refers to the Cholesky factor of cov
        """

        # set up KCAP's scale cuts module here.
        # initialize the scale cuts module from CosmoSIS.
        # path where the data is located
        data_loc = os.path.join(se.data_dir, se.bp_data)

        # a dictionary for the scale cut configuration
        config = {'scale_cuts': {'data_and_covariance_fits_filename': data_loc,
                                 'use_stats': self.stats,
                                 'output_section_name': 'likelihood_bp',
                                 'bandpower_e_cosmic_shear_extension_name': self.stats,
                                 'bandpower_e_cosmic_shear_section_name': 'bandpower'}}

        # for now we only look for these two keywords:
        if hasattr(se, 'keep_ang_PeeE'):
            config['scale_cuts'].update({'keep_ang_PeeE': se.keep_ang_PeeE})

        if hasattr(se, 'cut_pair_PeeE'):
            config['scale_cuts'].update({'cut_pair_PeeE': se.cut_pair_PeeE})

        # import scale_cuts as CosmoSIS module
        # scale cut python file
        sc_py = os.path.join(se.kcap_dir, 'modules/scale_cuts/scale_cuts.py')

        # apply the scale cut module in this case
        # scale cut module
        scm = crm.Module(module_name='scale_cuts', file_path=sc_py)

        # during set up the module stores the cut data vec and covmat in its data
        scm.setup(dict_to_datablock(config))

        # data vector
        data_vec = scm.data['data']

        # covariance matrix
        covmat = scm.data['covariance']

        # calculate the Cholesky factor
        cholesky_factor = cholesky(covmat, lower=True)

        if save:
            hp.store_arrays(data_vec, 'data', 'bandpowers')
            hp.store_arrays(covmat, 'data', 'covariance')

        # an empty dictionary to store all important quantities
        quantities = {}

        # the scale cut module
        quantities.update({'scm': scm})

        # the generated data vector
        quantities.update({'x': data_vec})

        # the covariance matrix
        quantities.update({'cov': covmat})

        # the Cholesky factor
        quantities.update({'cholesky': cholesky_factor})

        return quantities
