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
import setpriors as sp
import utils.common as uc


class nzDist(object):

    def __init__(self, technique: str = 'SOM'):
        """Module for setting up the redshift distribution. Currently, using the SOM technique provided by KiDS-1000 group.

        Args:
            technique (str, optional): The method we want to use for the n(z) distribution. Defaults to 'SOM'.
        """

        # use lower case letters in the string
        technique = technique.lower()

        if technique == 'som':
            self.nz = self.nz_som()

    def nz_som(self) -> dict:
        """Perform some preprocessing steps on the SOM redshift distributions.

        Returns:
            dict: A dictionary with the following keys: zs, z, h, zmax, nzmax, splines where
            - zs refers to the original redshift provided
            - z is a vector of redshift (on a new grid) depending on the nzmax supplied in the setting file
            - h are the heights of the n(z) distributions
            - zmax is the maximum redshift, to be used, with non-linear matter power spectrum (CLASS)
            - nzmax, the number of redshift samples we have (updated)
            - splines, the spline modules for each redshift distribution
        """

        z_samples, heights = self.load_som()

        # nredshifts = 121
        nredshifts = len(z_samples)

        # prevent undersampling of histograms!
        if se.nzmax < (nredshifts - 1):
            msg1 = "\nYou are trying to integrate at lower resolution than supplied by the n(z) histograms."
            msg2 = "\nIncrease nzmax greater or equal to {:}. Aborting now"
            print(msg1)
            print(msg2.format(nredshifts - 1))
            exit()

        # if that's the case, we want to integrate at histogram resolution and need to account for
        # the extra zero entry added
        elif se.nzmax == (nredshifts - 1):

            nzmax = len(z_samples)

            # requires that z-spacing is always the same for all bins
            redshifts = z_samples

            print('\nIntegrations performed at resolution of histogram.')

        # if we interpolate anyway at arbitrary resolution the extra 0 doesn't matter
        else:
            nzmax = se.nzmax + 1
            redshifts = np.linspace(min(z_samples), max(z_samples), nzmax)
            msg = '\nIntegration performed at set nzmax={:} resolution.'
            print(msg.format(nzmax - 1))

        if redshifts[0] == 0:
            redshifts[0] = 1E-4

        zmax = max(redshifts)

        splines_pz = []

        for zbin in range(se.nzbins):

            # we assume that the z-spacing is the same for each histogram
            spline = itp.interp1d(z_samples, heights[zbin, :], kind=se.nz_interp)

            splines_pz.append(spline)

        quantities = {}

        # z samples
        quantities.update({'zs': z_samples})

        # the redshifts
        quantities.update({'z': redshifts})

        # the p(z) heights
        quantities.update({'h': heights})

        # the maximum redshifts
        quantities.update({'zmax': zmax})

        # the maximum number of redshifts
        quantities.update({'nzmax': nzmax})

        # the spline interpolator
        quantities.update({'splines': splines_pz})

        return quantities

    def load_som(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load the redshifts (mid) and the n(z) distributions, as provided by KiDS-1000 group.

        Returns:
            Tuple[np.ndarray, np.ndarray]: the redhifts and the heights
        """

        # path of the file name
        fname = os.path.join(se.data_dir, se.bp_data)

        # load the fits file
        data_tables = fits.open(fname)

        # extract the n(z) distribution
        nofz = data_tables['NZ_SOURCE'].data

        # get the redshifts
        redshifts = np.concatenate((np.zeros(1), nofz['Z_MID']))

        # heights for the n(z) distribution
        heights = []

        for zbin in range(se.nzbins):

            hist_pz = nofz['BIN{:}'.format(zbin + 1)]

            heights += [np.concatenate((np.zeros(1), hist_pz))]

        heights = np.asarray(heights)

        return redshifts, heights

    def shift_som(self, shifts: dict) -> Tuple[np.ndarray, np.ndarray]:
        """The Self-Organising Map (SOM) method requires shifting the n(z) distribution to the right and to the left.

        Args:
            shifts (dict): A dictionary of size 5, which is added to the redshift to shift the n(z) distribution

        Returns:
            Tuple[np.ndarray, np.ndarray]: The unnormalised n(z) distributions and their normalisation factors
        """

        shifts_vals = uc.dvalues(shifts)

        pz_dist = np.zeros((self.nz['nzmax'], se.nzbins))

        pz_norm = np.zeros(se.nzbins, 'float64')

        for zbin in range(se.nzbins):

            # apply the shift to the redshift
            z_modified = self.nz['z'] + shifts_vals[zbin]

            # get the spline module specific to zbin
            spline_pz = self.nz['splines'][zbin]

            # perform masking
            mask_min = z_modified >= min(self.nz['zs'])
            mask_max = z_modified <= max(self.nz['zs'])
            mask = mask_min & mask_max

            # points outside the z-range of the histograms are set to 0.
            pz_dist[mask, zbin] = spline_pz(z_modified[mask])

            # Normalize selection functions
            dz = self.nz['z'][1:] - self.nz['z'][:-1]

            # area due to the small elements
            delta_area = 0.5 * (pz_dist[1:, zbin] + pz_dist[:-1, zbin]) * dz

            # normalisation factor
            pz_norm[zbin] = np.sum(delta_area)

        return pz_dist, pz_norm
