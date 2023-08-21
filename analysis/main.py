# Author: Arrykrishna Mootoovaloo
# Collaborators: Prof. Alan Heavens, Prof. Andrew Jaffe, Dr. Florent Leclercq
# Email : arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk
# Affiliation : Imperial Centre for Inference and Cosmology
# Status : Under Development

# Main script for running MCMC
import inference.sampling as sm

# MCMC = sm.mcmc('som', 'PeeE', True)
# MCMC.posterior_sampling('Experiment_3_chain_24_10000')


# MCMC = sm.mcmc('grouping', 'PeeE', True)
# MCMC.posterior_sampling('Experiment_4_grouping_samples_chain_24_10000')


import inference.likelihood as lk
testing = lk.sampling_dist('grouping', 'PeeE', True)

cosmo = {'omega_cdm':0.10802979609861982, 'omega_b': 0.02332493567194329, 'S_8': 0.815371910192108, 'n_s': 0.9704551766630282, 'h': 0.6887657808123486}
shifts = {'d1': 0.0, 'd2': 0.002, 'd3': 0.013, 'd4': 0.011, 'd5': -0.006}
nuisa = {'A_bary': 2.5, 'A_IA': 1.0}

print(testing.loglike(cosmo, shifts, nuisa)

      )
# def run_mcmc(fname):
#     MCMC = sm.mcmc(nz_method='som', stats='PeeE', save=True)
#     MCMC.posterior_sampling(fname)


# def worker(args):
#     return run_mcmc(*args)


# def parallel_run(names: list):
#     ncpu = mp.cpu_count()
#     pool = mp.Pool(processes=ncpu)
#     pool.map(worker, names)
#     pool.close()


# Names = [['test_' + str(i + 1)] for i in range(8)]

# parallel_run(Names)
