# Author: Arrykrishna Mootoovaloo
# Collaborators: Prof. Alan Heavens, Prof. Andrew Jaffe, Dr. Florent Leclercq
# Email : arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk
# Affiliation : Imperial Centre for Inference and Cosmology
# Status : Under Development (adapted from KiDS-1000 likelihood)

"""n(z) redshift distributions"""

from typing import Tuple
import os
import numpy as np
from astropy.io import fits
from scipy import interpolate as itp

# our scripts
import setemu as se
import utils.common as uc
from utils.helpers import pickle_load


def load_som() -> Tuple[np.ndarray, np.ndarray]:
    """Load the redshifts (mid) and the n(z) distributions, as provided by KiDS-1000 group.

    Returns:
        Tuple[np.ndarray, np.ndarray]: the redhifts and the heights
    """

    # path of the file name
    fname = os.path.join(se.data_dir, se.bp_data)

    # load the fits file
    data_tables = fits.open(fname)

    # extract the n(z) distribution
    nofz = data_tables["NZ_SOURCE"].data

    # get the redshifts
    redshifts = np.concatenate((np.zeros(1), nofz["Z_MID"]))

    # heights for the n(z) distribution
    heights = []

    for zbin in range(se.nzbins):
        hist_pz = nofz["BIN{:}".format(zbin + 1)]
        heights += [np.concatenate((np.zeros(1), hist_pz))]

    heights = np.asarray(heights)
    return redshifts, heights


def load_grouping(mean_nz: bool = True) -> Tuple[np.ndarray, dict]:
    """
    Load the samples or mean of the n(z). Each bin name is given by:
    - BIN1, BIN2...BIN5

    Args:
        mean_nz (bool, optional): Option to use either the mean or samples of n(z). Defaults to True.

    Returns:
        Tuple[np.ndarray, dict]: The redshifts and the heights of the redshift distribution.
    """

    if mean_nz:
        file = pickle_load(se.data_dir, "grouping_mean")
    else:
        file = pickle_load(se.data_dir, "grouping_samples")
    redshifts = file["redshifts"]

    # this is of shape Np x Nsamples for samples
    hist_pz = file["heights"]
    return redshifts, hist_pz


def sample_grouping(
    redshifts: np.ndarray, hist_pz: dict, mean_nz: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample a redshift distribution from the n(z). We will concatenate a zero for both redshift
    and the height of the n(z). The final shape of the n(z) will be 5 x 120.

    Args:
        redshifts (np.ndarray): the redshift axis
        hist_pz (dict): a dictionary (keys: BIN1...BIN5) containing the samples of n(z)
        mean_nz (bool): whether we are using the mean of the n(z) distribution

    Returns:
        Tuple[np.ndarray, np.ndarray]: the redshift and the sample of n(z)
    """

    z_samples = np.concatenate((np.zeros(1), redshifts))
    heights = []
    for i in range(se.nzbins):
        if mean_nz:
            h_chosen = hist_pz[f"BIN{i+1}"]
        else:
            nsamples = hist_pz[f"BIN{i+1}"].shape[1]
            idx = np.random.choice(nsamples, 1).item()
            h_chosen = hist_pz[f"BIN{i+1}"][:, idx]
        heights += [np.concatenate((np.zeros(1), h_chosen))]
    heights = np.asarray(heights)
    return z_samples, heights


def quantities_nz(z_samples: np.ndarray, heights: np.ndarray) -> dict:
    """
    Generate a dictionary of quantities for working with the n(z) distributions.

    Args:
        z_samples (np.ndarray): the redshifts
        heights (np.ndarray): the heights of the n(z)

    Returns:
        dict: a dictionary of containing the different quantities.
    """

    nredshifts = len(z_samples)

    # prevent undersampling of histograms!
    if se.nzmax < (nredshifts - 1):
        msg1 = "\nYou are trying to integrate at lower resolution than supplied by the n(z) histograms."
        msg2 = "\nIncrease nzmax greater or equal to {:}. Aborting now"
        print(msg1)
        print(msg2.format(nredshifts - 1))
        exit()

    elif se.nzmax == (nredshifts - 1):
        nzmax = len(z_samples)
        redshifts = z_samples
        print("\nIntegrations performed at resolution of histogram.")

    else:
        nzmax = se.nzmax + 1
        redshifts = np.linspace(min(z_samples), max(z_samples), nzmax)
        msg = "\nIntegration performed at set nzmax={:} resolution."
        print(msg.format(nzmax - 1))

    if redshifts[0] == 0:
        redshifts[0] = 1e-4

    zmax = max(redshifts)
    splines_pz = []
    for zbin in range(se.nzbins):
        # we assume that the z-spacing is the same for each histogram
        spline = itp.interp1d(z_samples, heights[zbin, :], kind=se.nz_interp)
        splines_pz.append(spline)

    quantities = {}
    quantities.update({"zs": z_samples})
    quantities.update({"z": redshifts})
    quantities.update({"h": heights})
    quantities.update({"zmax": zmax})
    quantities.update({"nzmax": nzmax})
    quantities.update({"splines": splines_pz})
    return quantities


class nzDist:
    """Module for setting up the redshift distribution. Currently, using the SOM technique provided by KiDS-1000 group.

    Args:
        technique (str, optional): The method we want to use for the n(z) distribution. Defaults to 'SOM'.
    """

    def __init__(self, technique: str = "SOM"):
        # use lower case letters in the string
        self.technique = technique.lower()

        print(f'Using method: {technique}')

        if self.technique == "som":
            z_samples, heights = load_som()
            self.nz = quantities_nz(z_samples, heights)
        elif self.technique == 'grouping':
            redshifts, hist_pz = load_grouping(mean_nz=True)
            z_samples, heights = sample_grouping(redshifts, hist_pz, mean_nz=True)
            self.nz = quantities_nz(z_samples, heights)

    def shift_redshifts(
        self, shifts: dict, sample: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """The Self-Organising Map (SOM) method requires shifting the n(z) distribution to the right and to the left.

        Args:
            shifts (dict): A dictionary of size 5, which is added to the redshift to shift the n(z) distribution
            sample (bool): Draw a new sample

        Returns:
            Tuple[np.ndarray, np.ndarray]: The unnormalised n(z) distributions and their normalisation factors
        """
        shifts_vals = uc.dvalues(shifts)
        pz_dist = np.zeros((self.nz["nzmax"], se.nzbins))
        pz_norm = np.zeros(se.nzbins, "float64")

        if sample and self.technique == "grouping":
            redshifts, hist_pz = load_grouping(mean_nz=False)
            z_samples, heights = sample_grouping(redshifts, hist_pz, mean_nz=False)
            self.nz = quantities_nz(z_samples, heights)

        for zbin in range(se.nzbins):
            # apply the shift to the redshift
            z_modified = self.nz["z"] + shifts_vals[zbin]

            # get the spline module specific to zbin
            spline_pz = self.nz["splines"][zbin]

            # perform masking
            mask_min = z_modified >= min(self.nz["zs"])
            mask_max = z_modified <= max(self.nz["zs"])
            mask = mask_min & mask_max

            # points outside the z-range of the histograms are set to 0.
            pz_dist[mask, zbin] = spline_pz(z_modified[mask])

            # Normalize selection functions
            deltaz = self.nz["z"][1:] - self.nz["z"][:-1]

            # area due to the small elements
            delta_area = 0.5 * (pz_dist[1:, zbin] + pz_dist[:-1, zbin]) * deltaz

            # normalisation factor
            pz_norm[zbin] = np.sum(delta_area)

        return pz_dist, pz_norm
