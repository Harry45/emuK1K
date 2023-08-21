# Author: Arrykrishna Mootoovaloo
# Collaborators: Prof. Alan Heavens, Prof. Andrew Jaffe, Dr. Florent Leclercq
# Email : arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk
# Affiliation : Imperial Centre for Inference and Cosmology
# Status : Under Development

"""Calculates the different type of weak lensing power spectra using the 3D matter power spectrum"""

from typing import Tuple
import numpy as np

# our scripts
import setemu as se
import cosmology.classcompute as cc
import cosmology.cosmofuncs as cf
import utils.common as uc


class spectra(cc.pkclass):

    def __init__(self, nz_method='som'):

        cc.pkclass.__init__(self, nz_method)

        # values of ells
        self.ells = np.geomspace(se.ell_min, se.ell_max, se.nells, endpoint=True)

        # number of auto- and cross- correlation
        self.nzcorrs = int(se.nzbins * (se.nzbins + 1) / 2)

    def pk_prediction(self, parameters: dict, a_bary: float = 0.0) -> Tuple[np.ndarray, dict]:
        """Calculate the non-linear matter power spectrum (inteprolated) and other important quantities

        Args:
            parameters (dict): the set of cosmological parameters
            a_bary (float, optional): the baryon feedback parameter. Defaults to 0.0.

        Returns:
            Tuple[np.ndarray, dict]: the non-linear matter power spectrum and some quantities
        """

        # calcualte the power spectrum
        # 50 values of k equally spaced in log and 30 values of redshift qually spaced in linear scale
        # also return some
        spectrum, quant = self.pk_nonlinear(parameters, a_bary)

        chi = quant['chi']

        # emulator is trained with k in units of h Mpc^-1
        # therefore, we should input k = k/h in interpolator
        # example: interp(*[np.log(0.002/d['h']), 2.0])
        inputs = [self.k_grid, self.z_grid, spectrum.flatten()]

        interp = uc.like_interp_2d(inputs)

        # Get power spectrum P(k=l/r,z(r)) from cosmological module or emulator
        pk_matter = np.zeros((se.nells, self.nz['nzmax']), 'float64')

        k_max_in_inv_mpc = se.kmax * parameters['h']

        for il in range(se.nells):
            for iz in range(self.nz['nzmax']):

                k_in_inv_mpc = (self.ells[il] + 0.5) / chi[iz]

                if k_in_inv_mpc > k_max_in_inv_mpc:

                    # assign a very small value of matter power
                    pk_dm = 1E-300

                else:

                    # the interpolator is built on top of log(k[h/Mpc])
                    newpoint = [np.log(k_in_inv_mpc / parameters['h']), self.nz['z'][iz]]

                    # predict the power spectrum
                    pk_dm = interp(*newpoint)

                # record the matter power spectrum
                pk_matter[il, iz] = pk_dm

        return pk_matter, quant

    def lensing_kernel(self, chi: np.ndarray, pchi: np.ndarray) -> np.ndarray:
        """Calculates the lensing kernel using the comoving radial distance and the source distribution

        Args:
            chi (np.ndarray): the comoving radial distance
            pchi (np.ndarray): the source distribution, n(chi), where n(chi) d(chi) = n(z) dz

        Returns:
            np.ndarray: the lensing kernel for each bin alpha
        """

        g = np.zeros((self.nz['nzmax'], se.nzbins), 'float64')

        for alpha in range(se.nzbins):

            for nr in range(1, self.nz['nzmax'] - 1):

                fun = pchi[nr:, alpha] * (chi[nr:] - chi[nr]) / chi[nr:]

                g[nr, alpha] = np.sum(0.5 * (fun[1:] + fun[:-1]) * (chi[nr + 1:] - chi[nr:-1]))
                g[nr, alpha] *= chi[nr] * (1. + self.nz['z'][nr])

        return g

    def total_shear(self, cosmology: dict, shifts: dict, nuisance: dict) -> Tuple[np.ndarray, list]:
        """Calculates the total shear power spectrum

        Args:
            cosmology (dict): A dictionary with the cosmological parameters
            shifts (dict): A dictionary for the shifts in the n(z) distribution
            nuisance (dict): A dictionary for the nuisance parameters, A_IA and A_bary

        Returns:
            Tuple[dict, list]: the total shear power spectrum and the keys
        """

        cl, keys = self.shear_power_spectrum(cosmology, shifts, nuisance['A_bary'])

        # total shear, including intrinsic alignment
        cl_total = cl['cl_gg'] + nuisance['A_IA']**2 * cl['cl_ii'] - nuisance['A_IA'] * cl['cl_gi']

        return cl_total, keys

    def shear_power_spectrum(self, cosmology: dict, shifts: dict, a_bary: float = 0.0) -> Tuple[dict, list]:
        """Calculates the different shear power spectra (GG, GI, II) using the functional form of n(z)

        Args:
            cosmology (dict): A dictionary with the cosmological parameters
            shifts (dict): A dictionary for the shifts in the n(z) distribution
            a_bary (float, optional): [description]. Defaults to 0.0.

        Returns:
            Tuple[dict, list]: shear power spectrum and the keys
        """

        # get the 3D matter power spectrum and the important quantities
        # we need to improve the first line to incorporate interpolation to k_values > k_max
        # polyfit in KiDS-1000 likelihood
        # pk, quant = self.pk_prediction(cosmology, a_bary)
        pk, quant = self.get_matter_power_spectrum(cosmology, a_bary)

        # get the new n(z) and their normalisation when using SOM
        pz, pz_norm = self.shift_redshifts(shifts, sample=se.nz_sample)

        # Jacobian : calculate the source distribution (in terms of the comoving radial distance)
        pchi = pz * (quant['dzdr'][:, np.newaxis] / pz_norm)

        # get the lensing kernel
        g = self.lensing_kernel(quant['chi'], pchi)

        Cl_GG_integrand = np.zeros((self.nz['nzmax'], self.nzcorrs), 'float64')
        Cl_II_integrand = np.zeros((self.nz['nzmax'], self.nzcorrs), 'float64')
        Cl_GI_integrand = np.zeros((self.nz['nzmax'], self.nzcorrs), 'float64')

        Cl_GG = np.zeros((self.nzcorrs, se.nells), 'float64')
        Cl_II = np.zeros((self.nzcorrs, se.nells), 'float64')
        Cl_GI = np.zeros((self.nzcorrs, se.nells), 'float64')

        # empty list to store the different keys
        cl_keys = []

        # calculate delta chi
        dchi = quant['chi'][1:] - quant['chi'][:-1]

        # Start loop over l for computation of shear power spectrum
        for il in range(se.nells):
            for alpha in range(se.nzbins):
                for beta in range(alpha, se.nzbins):

                    # record the keys
                    if il == 0:
                        cl_keys += ['bin_{:}_{:}'.format(beta + 1, alpha + 1)]

                    # one dimensional index for KiDS-1000
                    odi = cf.one_dim_index(alpha, beta, se.nzbins)

                    # get_factor_ia(quant: dict, redshift: np.ndarray, amplitude: float, exponent=0.0)
                    factor_IA = cf.get_factor_ia(quant, self.nz['z'])

                    # get some pre-factors involving the lensing kernel and the source dstribution
                    f_gg = g[1:, alpha] * g[1:, beta] / quant['chi'][1:]**2
                    f_ii = pchi[1:, alpha] * pchi[1:, beta] / quant['chi'][1:]**2
                    f_gi = (g[1:, alpha] * pchi[1:, beta] + g[1:, beta] * pchi[1:, alpha]) / quant['chi'][1:]**2

                    # calculate the integrand
                    Cl_GG_integrand[1:, odi] = f_gg * pk[il, 1:]
                    Cl_II_integrand[1:, odi] = f_ii * factor_IA[1:]**2 * pk[il, 1:]
                    Cl_GI_integrand[1:, odi] = f_gi * factor_IA[1:] * pk[il, 1:]

            for j in range(self.nzcorrs):
                Cl_GG[j, il] = np.sum(0.5 * (Cl_GG_integrand[1:, j] + Cl_GG_integrand[:-1, j]) * dchi)
                Cl_II[j, il] = np.sum(0.5 * (Cl_II_integrand[1:, j] + Cl_II_integrand[:-1, j]) * dchi)
                Cl_GI[j, il] = np.sum(0.5 * (Cl_GI_integrand[1:, j] + Cl_GI_integrand[:-1, j]) * dchi)

                Cl_GG[j, il] *= quant['a_fact']**2
                Cl_GI[j, il] *= quant['a_fact']

        # store each power spectrum type in a dictionary
        Cl = {}
        Cl.update({'cl_gg': Cl_GG})
        Cl.update({'cl_gi': Cl_GI})
        Cl.update({'cl_ii': Cl_II})

        return Cl, cl_keys

    def get_matter_power_spectrum(self, parameters: dict, a_bary: float = 0.0):

        # Calculate the 3D matter power spectrum
        class_module = self.class_compute(parameters)

        # Get the Hubble parameter
        h = class_module.h()

        # Omega_matter
        omega_m = class_module.Omega_m()

        # h parameter
        small_h = class_module.h()

        # critical density
        rc = cf.get_critical_density(small_h)

        # derive the linear growth factor D(z)
        lgr = np.zeros_like(self.nz['z'])

        for iz, red in enumerate(self.nz['z']):

            # compute linear growth rate
            lgr[iz] = class_module.scale_independent_growth_factor(red)

        # normalise linear growth rate at redshift = 0
        lgr /= class_module.scale_independent_growth_factor(0.)

        # get distances from cosmo-module
        chi, dzdr = class_module.z_of_r(self.nz['z'])

        # numerical stability for chi
        chi += 1E-32

        # pre-factor A in weak lensing
        a_fact = (3. / 2.) * omega_m * small_h**2 / 2997.92458**2

        # important quantities
        quant = {}
        quant.update({'omega_m': omega_m})
        quant.update({'chi': chi})
        quant.update({'dzdr': dzdr})
        quant.update({'lgr': lgr})
        quant.update({'rc': rc})
        quant.update({'a_fact': a_fact})
        quant.update({'small_h': small_h})

        # Get power spectrum P(k=l/r,z(r)) from cosmological module
        pk = np.zeros((se.nells, self.nz['nzmax']), 'float64')

        k_max_in_inv_Mpc = se.k_max_h_by_Mpc * h

        for idx_z in range(self.nz['nzmax']):

            all_k_in_inv_Mpc = (self.ells + 0.5) / chi[idx_z]

            # For k values larger than k_max_in_inv_Mpc we use an interpolation of the
            # matter power spectrum to larger values

            idx_larger_k_max_in_inv_Mpc = all_k_in_inv_Mpc > k_max_in_inv_Mpc

            if any(idx_larger_k_max_in_inv_Mpc):

                # index start
                itp_start = np.where(idx_larger_k_max_in_inv_Mpc)[0][0]

                # extended indices
                itp_indices = np.arange(itp_start - 3, itp_start)

                # values of k in log
                xvals = np.log(all_k_in_inv_Mpc[itp_indices])

                # values of pk in log
                yvals = [np.log(class_module.pk(all_k_in_inv_Mpc[i], self.nz['z'][idx_z])) for i in itp_indices]

                p_dm = np.polyfit(xvals, yvals, 1)

            for idx_ell in range(se.nells):

                # standard Limber approximation:
                k_in_inv_Mpc = (self.ells[idx_ell] + 0.5) / chi[idx_z]

                if k_in_inv_Mpc > k_max_in_inv_Mpc:
                    pk_dm = np.exp(np.polyval(p_dm, np.log(k_in_inv_Mpc)))
                else:
                    pk_dm = class_module.pk(k_in_inv_Mpc, self.nz['z'][idx_z])

                pk[idx_ell, idx_z] = pk_dm * cf.bar_fed(k_in_inv_Mpc / h, self.nz['z'][idx_z], a_bary).flatten()

        # clean class_module to prevent memory issue
        cf.delete_module(class_module)

        return pk, quant
