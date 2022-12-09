# Author: Arrykrishna Mootoovaloo
# Collaborators: Prof. Alan Heavens, Prof. Andrew Jaffe, Dr. Florent Leclercq
# Email : arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk
# Affiliation : Imperial Centre for Inference and Cosmology
# Status : Under Development


"""
Module for important calculations involving the prior. For example,

- when scaling the Latin Hypercube samples to the appropriate prior range

- when calculating the posterior if the emulator is connected with an MCMC sampler
"""

import scipy.stats
import numpy as np


def entity(dictionary: dict) -> dict:
    """Creates a probability distribution object using scipy.stats

    Args:
        dictionary (dict): a dictionary containing information for each parameter, that is,

            - distribution, specified by the key 'distribution'

            - specifications, specified by the key 'specs'
    Returns:
        dict: the distribution generated using scipy
    """

    dist = eval('scipy.stats.' +
                dictionary['distribution'])(*dictionary['specs'])

    return dist


def all_entities(dict_params: dict) -> dict:
    """Generate all the priors once we have specified them.

    Args:
        dict_params (dict): a list containing the description for each parameter and each description (dictionary) contains the following information:

        - distribution, specified by the key 'distribution'

        - parameter name, specified by the key 'parameter'

        - specifications, specified by the key 'specs'
    Returns:
        dict: a dictionary containing the prior for each parameter, that is, each element contains the following information:

        - parameter name, specified by the key 'parameter'

        - distribution, specified by the key 'distribution'
    """
    # create an empty list to store the distributions
    record = {}

    for c in dict_params:
        record[c] = entity(dict_params[c])

    return record


def log_prod_pdf(desc: dict, parameters: dict) -> float:
    """Calculate the log-product for a set of parameters given the priors

    Args:
        desc (dict): dictionary of parameters
        parameters (dict): an array of parameters

    Returns:
        float: the log-probability when the pdf of each parameter is multiplied with another
    """

    # initialise log_sum to 0.0
    log_sum = 0.0

    if len(desc) == 1:
        # get the key names for the dictionary, for example, deltas
        key = list(desc.keys())[0]

        # get the parameter values
        values = np.array(list(parameters.values()))

        # calculate the log-probability
        log_sum += desc[key].logpdf(values)

    else:
        # calculate the log-pdf for each parameter
        for p in parameters:
            log_sum += desc[p].logpdf(parameters[p])

    return log_sum
