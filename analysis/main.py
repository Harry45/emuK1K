# Author: Arrykrishna Mootoovaloo
# Collaborators: Prof. Alan Heavens, Prof. Andrew Jaffe, Dr. Florent Leclercq
# Email : arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk
# Affiliation : Imperial Centre for Inference and Cosmology
# Status : Under Development

# Main script for running MCMC
import inference.sampling as sm

MCMC = sm.mcmc('som', 'PeeE', True)
MCMC.posterior_sampling('Experiment_3_chain_24_10000')


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
