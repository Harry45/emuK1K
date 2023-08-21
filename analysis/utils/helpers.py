# Author: (Dr to be) Arrykrishna Mootoovaloo
# Collaborators: Prof. Alan Heavens, Prof. Andrew Jaffe, Dr. Florent Leclercq
# Email : a.mootoovaloo17@imperial.ac.uk
# Affiliation : Imperial Centre for Inference and Cosmology
# Status : Under Development

"""
Important functions to store/load files in a compressed format.
"""

import os
import numpy as np
import pickle
from typing import Any


def save_excel(df, folder_name, file_name):
    """
    Given a folder name and file name, we will save a pandas dataframe to a excel file.

    :param: df (pd.DataFrame) - pandas dataframe

    :param: folder name (str) - name of the folder

    :param: file name (str) - name of the file output
    """

    # create the folder if it does not exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    df.to_excel(excel_writer=folder_name + "/" + file_name + ".xlsx")


def load_arrays(folder_name, file_name):
    """
    Given a folder name and file name, we will load
    the array

    :param: folder_name (str) - the name of the folder

    :param: file_name (str) - name of the file

    :return: matrix (np.ndarray) - array
    """

    matrix = np.load(folder_name + "/" + file_name + ".npz")["arr_0"]

    return matrix


def store_arrays(array, folder_name, file_name):
    """
    Given an array, folder name and file name, we will store the
    array in a compressed format.

    :param: array (np.ndarray) - array which we want to store

    :param: folder_name (str) - the name of the folder

    :param: file_name (str) - name of the file
    """

    # create the folder if it does not exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # use compressed format to store data
    np.savez_compressed(folder_name + "/" + file_name + ".npz", array)


def pickle_save(file: list, folder: str, fname: str) -> None:
    """Stores a list in a folder.
    Args:
        list_to_store (list): The list to store.
        folder_name (str): The name of the folder.
        file_name (str): The name of the file.
    """

    # create the folder if it does not exist
    os.makedirs(folder, exist_ok=True)

    # use compressed format to store data
    path = os.path.join(folder, fname)
    with open(path + ".pkl", "wb") as dummy:
        pickle.dump(file, dummy)


def pickle_load(folder: str, fname: str) -> Any:
    """Reads a list from a folder.
    Args:
        folder_name (str): The name of the folder.
        file_name (str): The name of the file.
    Returns:
        pd.DataFrame: a pandas dataframe
    """
    path = os.path.join(folder, fname)
    with open(path + ".pkl", "rb") as dummy:
        file = pickle.load(dummy)
    return file
