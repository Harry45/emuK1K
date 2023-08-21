# Author: Arrykrishna Mootoovaloo
# Collaborators: Prof. Alan Heavens, Prof. Andrew Jaffe, Dr. Florent Leclercq
# Email : arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk
# Affiliation : Imperial Centre for Inference and Cosmology
# Status : Under Development (adapted from KiDS-1000 likelihood)

"""All the settings for building the emulator"""

# -----------------------------------------------------------------------------

# method for building the emulator
# "components" refers to the fact that we are using the:
# - linear matter power spectrum, P_lin(k,z0)
# - growth factor, A(z)
# - q function, q(k,z)
# to reconstruct the non-linear matter power spectrum as
# p_nonlinear = A(z) * [1 + q(k,z)] * P_lin(k,z0)

# data.cosmo_arguments['N_eff'] = 2.0328
# data.cosmo_arguments['N_ncdm'] = 1
# data.cosmo_arguments['m_ncdm'] = 0.06
# data.cosmo_arguments['T_ncdm'] = 0.71611

# fixed neutrino mass
neutrino = {'N_eff': 2.0328, 'N_ncdm': 1, 'm_ncdm': 0.06, 'T_ncdm': 0.71611}

# time out function (in seconds)
# CLASS does not run for certain cosmologies
timeout = 60

timed = True

# -----------------------------------------------------------------------------
# Important parameter inputs for calculating the matter power spectrum

# settings for halofit

# halofit needs to evaluate integrals (linear power spectrum times some
# kernels). They are sampled using this logarithmic step size
# default in CLASS is 80
halofit_k_per_decade = 80.

# a smaller value will lead to a more precise halofit result at the
# highest redshift at which halofit can make computations,at the expense
# of requiring a larger k_max; but this parameter is not relevant for the
# precision on P_nl(k,z) at other redshifts, so there is normally no need
# to change it
# default in CLASS is 0.05
halofit_sigma_precision = 0.05

# minimum redshift
zmin = 0.0

# maximum redshift
zmax = 6.0

# maximum of k (for quick CLASS run, set to for example, 50)
k_max_h_by_Mpc = 20.

# additional parameter set by KiDS-1000
min_k_max = 20.

# our wanted kmax
kmax = 20.0

# minimum of k
k_min_h_by_Mpc = 5E-4

# number of k
nk = 50

# number of redshift on the grid
nz = 30

k_pivot = 0.05

# bbn = '/home/harry/Desktop/class/bbn/sBBN.dat'
bbn = '/mnt/zfsusers/phys2286/class/external/bbn/sBBN_2017.dat'
# -----------------------------------------------------------------------------
# folders and files

# data directory
data_dir = 'data/'

# KCAP (KiDS Cosmology Analysis Pipeline)
# kcap_dir = '/home/harry/lensing/kcap/'
kcap_dir = '/mnt/zfsusers/phys2286/kcap/'

# band powers data
bp_data = 'kids_1000.fits'

# -----------------------------------------------------------------------------
# band powers parameters

# set the (angular) theta-range for the integration to band powers:
theta_min = 0.5
theta_max = 300.

# supply file with ell-ranges for band power bins:
# if set, this option will overwrite the log-binning below!
# K1K_BandPowers.ell_bin_file = ''

# set the lower and upper limit of the band power bins:
ell_bin_min = 100.
ell_bin_max = 1500.

# set the number of band power bins for the theory calculation:
# code will perform log-binning then between ell_bin_min and ell_bin_max
nbins = 8

# set to True if analytic solution for g should be used:
analytic = True

# set the type of response function:
response_function = 'tophat'

# set to True if apodisation should be used:
apodise = True

# set length scale of apodisation in arcmin:
delta_x = 0.5

# -----------------------------------------------------------------------------
# n(z) distributions

# number of redshift (tomographic) bins:
nzbins = 5

# number of discrete z-values used for all integrations, can be set to arbitrary numbers now
# for fiducial KV450 analysis: 120
nzmax = 120

# whether to sample the redshift distributions
# can be done for grouping method
# for SOM, only the mean is available
nz_sample = True

# you can choose here any of the scipy.interpolate.interp1d types of interpolation
# (i.e. 'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'previous',
# 'next' in SciPy v1.1.0) for the n(z) interpolation ('linear' is recommended)
# for fiducial KV450 analysis: 'linear'
nz_interp = 'cubic'

# -----------------------------------------------------------------------------
# scale cuts

# set up the scale cuts to use in the following lines:
# we keep band powers 1 to 5 in the data vector:
keep_ang_PeeE = '99.5 1500.5'
# if you want to cut tomo pairs specify like this (cuts out z-bin5 and all of
# its cross-correlations):
# cut_pair_PeeE = '1+5 2+5 3+5 4+5 5+5'

# -----------------------------------------------------------------------------
# precision settings

# these settings set the precision for the Cl integration
# minimum l for C_l
ell_min = 1.

# maximum l for C_l
ell_max = 1e4

# number of (log-spaced) ell-nodes between ell_min and ell_max
nells = 50

# -----------------------------------------------------------------------------
# Baryon Feedback settings

# baryon model to be used
baryon_model = 'AGN'

cst = {'AGN': {'A2': -0.11900, 'B2': 0.1300, 'C2': 0.6000, 'D2': 0.002110, 'E2': -2.0600,
               'A1': 0.30800, 'B1': -0.6600, 'C1': -0.7600, 'D1': -0.002950, 'E1': 1.8400,
               'A0': 0.15000, 'B0': 1.2200, 'C0': 1.3800, 'D0': 0.001300, 'E0': 3.5700},
       'REF': {'A2': -0.05880, 'B2': -0.2510, 'C2': -0.9340, 'D2': -0.004540, 'E2': 0.8580,
               'A1': 0.07280, 'B1': 0.0381, 'C1': 1.0600, 'D1': 0.006520, 'E1': -1.7900,
               'A0': 0.00972, 'B0': 1.1200, 'C0': 0.7500, 'D0': -0.000196, 'E0': 4.5400},
       'DBLIM': {'A2': -0.29500, 'B2': -0.9890, 'C2': -0.0143, 'D2': 0.001990, 'E2': -0.8250,
                 'A1': 0.49000, 'B1': 0.6420, 'C1': -0.0594, 'D1': -0.002350, 'E1': -0.0611,
                 'A0': -0.01660, 'B0': 1.0500, 'C0': 1.3000, 'D0': 0.001200, 'E0': 4.4800}}

# -----------------------------------------------------------------------------

# Settings for the GP emulator module

# noise/jitter term
var = 1E-5

# another jitter term for numerical stability
jitter = 1E-5

# order of the polynomial (maximum is 2)
order = 2

# Transform input (pre-whitening)
x_trans = True

# Centre output on 0 if we want
use_mean = False

# Number of times we want to restart the optimiser
n_restart = 5

# minimum lengthscale (in log)
l_min = -5.0

# maximum lengthscale (in log)
l_max = 5.0

# minimum amplitude (in log)
a_min = 0.0

# maximum amplitude (in log)
a_max = 25.0

# choice of optimizer (better to use 'L-BFGS-B')
method = 'L-BFGS-B'

# tolerance to stop the optimizer
ftol = 1E-30

# maximum number of iterations
maxiter = 600

# decide whether we want to delete the kernel or not
del_kernel = True

# growth factor (not very broad distribution in function space)
gf_args = {'y_trans': False, 'lambda_cap': 1}

# if we want to emulate 1 + q(k,z):
emu_one_plus_q = True

if emu_one_plus_q:

    # q function (expected to be zero)
    qf_args = {'y_trans': True, 'lambda_cap': 1}

    # folder where we will store the files
    d_one_plus = '_op'

else:

    # q function (expected to be zero)
    qf_args = {'y_trans': False, 'lambda_cap': 1}

    # folder where we will store the files
    d_one_plus = ''

# linear matter power spectrum
pl_args = {'y_trans': True, 'lambda_cap': 1000}

# non linear matter power spectrum
pknl_args = {'y_trans': True, 'lambda_cap': 1000}
