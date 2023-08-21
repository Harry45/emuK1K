# Author: Arrykrishna Mootoovaloo
# Collaborators: Alan Heavens, Andrew Jaffe, Florent Leclercq
# Email : a.mootoovaloo17@imperial.ac.uk
# Affiliation : Imperial Centre for Inference and Cosmology
# Status : Under Development

'''
Perform all additional operations such as interpolations
'''

import sys
import os
import logging
import numpy as np
import scipy.interpolate as itp
from typing import Tuple


def block_print():
    sys.stdout = open(os.devnull, 'w')


def enable_print():
    sys.stdout = sys.__stdout__

def subset_dict(d: dict, keys: list) -> dict:
    """Create a subset dictionary given an extended dictionary

    Args:
        d (dict): The extended dictionary
        keys (list): The small list of keys (str)

    Returns:
        dict: The smaller dictionary extracted from the larger dictionary
    """

    return dict((k, d[k]) for k in keys if k in d)


def mk_dict(l1: list, l2: list) -> dict:
    """Create a dictionary given a list of string and a list of numbers

    Args:
        l1 (list): a list of names for the parameters
        l2 (list): a list of values for the parameters

    Raises:
        ValueError: Size of inputs must match

    Returns:
        dict: a dictionary built using the keys and values
    """

    # Create a dictionary given a list of string and a list of numbers

    # :param: l1 (list) - list of string (parameter names)

    # :param: l2 (list) - list of values (values of each parameter)

    # :return: d (dict) - a dictionary consisting of the keys and values

    if len(l1) != len(l2):
        raise ValueError('Mis-match between parameter names and values.')

    d = dict(zip(l1, l2))

    return d


def indices(nzmax: int) -> Tuple[list, tuple]:
    '''
    Generates indices for double sum power spectra

    : param: nzmax(int) - the maximum number of redshifts(assuming first redshift is zero)

    : return: di_ee(list), idx_gi(tuple) - double indices for EE and indices for GI
    '''

    # create emty lists to recod all indices
    # for EE power spectrum
    di_ee = []

    # for GI power spectrum
    # ab means alpha, beta
    Lab_1 = []
    Lab_2 = []

    Lba_1 = []
    Lba_2 = []

    for i in range(1, nzmax + 1):
        for j in range(1, nzmax + 1):

            di_ee.append(np.min([i, j]))

            if i > j:
                Lab_1.append(i)
                Lab_2.append(j)

            elif j > i:
                Lba_1.append(i)
                Lba_2.append(j)

    Lab_1 = np.asarray(Lab_1)
    Lab_2 = np.asarray(Lab_2)
    Lba_1 = np.asarray(Lba_1)
    Lba_2 = np.asarray(Lba_2)

    di_ee = np.asarray(di_ee)

    idx_gi = (Lab_1, Lab_2, Lba_1, Lba_2)

    return di_ee, idx_gi


def removekey(d: dict, key: str) -> dict:
    """Remove a specific key and its content from a dictionary

    Args:
        d (dict): the dictionary
        key (str): the key we want to remove

    Returns:
        dict: the updated dictionary
    """
    # make a copy of the dictionary
    r = dict(d)

    # delete the key
    del r[key]

    return r


def dvalues(d: dict) -> np.ndarray:
    '''
    Returns an array of values instead of dictionary format

    : param: d(dict) - a dictionary with keys and values

    : return: v(np.ndarray) - array of values
    '''
    v = np.array(list(d.values()))

    return v


def like_interp_2d(inputs: list, int_type: str = 'cubic') -> object:
    '''
    We want to predict the function for any new point of k and z(example)

    : param: inputs(list) - a list containing x, y, f(x, y)

    : param: int_type(str) - interpolation type(default: 'cubic')

    : return: f(object) - the interpolator
    '''

    k, z, f_kz = np.log(inputs[0]), inputs[1], inputs[2]

    inputs_trans = [k, z, f_kz]

    f = itp.interp2d(*inputs_trans)

    return f


def two_dims_interpolate(inputs: list, grid: list) -> np.ndarray:
    '''
    Function to perform 2D interpolation using interpolate.interp2d

    : param: inputs(list): inputs to the interpolation module, that is, we need to specify the following:
        - x
        - y
        - f(x, y)
        - 'linear', 'cubic', 'quintic'

    : param: grid(list): a list containing xnew and ynew

    : return: pred_new(np.ndarray): the predicted values on the 2D grid
    '''

    # check that all elements are greater than 0 for log-transformation to be used
    condition = np.all(inputs[2] > 0)

    if condition:
        # transform k and f to log
        k, z, f_kz, int_type = np.log(inputs[0]), inputs[1], np.log(inputs[2]), inputs[3]

    else:
        # transform in k to log
        k, z, f_kz, int_type = np.log(inputs[0]), inputs[1], inputs[2], inputs[3]

    inputs_trans = [k, z, f_kz, int_type]

    # tranform the grid to log
    knew, znew = np.log(grid[0]), grid[1]

    grid_trans = [knew, znew]

    f = itp.interp2d(*inputs_trans)

    if condition:
        pred_new = np.exp(f(*grid_trans))

    else:
        pred_new = f(*grid_trans)

    return pred_new


def interpolate(inputs: list) -> np.ndarray:
    '''
    Function to interpolate the power spectrum along the redshift axis

    : param: inputs(list or tuple): x values, y values and new values of x

    : return: ynew(np.ndarray): an array of the interpolated power spectra
    '''

    x, y, xnew = inputs[0], inputs[1], inputs[2]

    spline = itp.splrep(x, y)

    ynew = itp.splev(xnew, spline)

    return ynew


def get_logger(name: str, log_name: str, folder_name: str = 'logs'):
    '''
    Create a log file for each Python scrip

    : param: name(str) - name of the Python script

    : param: log_name(str) - name of the output log file
    '''
    # create the folder if it does not exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    log_format = '%(asctime)s  %(name)8s  %(levelname)5s  %(message)s'

    logging.basicConfig(level=logging.DEBUG,
                        format=log_format,
                        filename=folder_name + '/' + log_name + '.log',
                        filemode='w')

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(name).addHandler(console)

    return logging.getLogger(name)
