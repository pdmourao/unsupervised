import laboratory as lab
import numpy as np
from tqdm import tqdm
import experiments as exp

def mags_onestep_1d(x_arg, x_values, samples, **kwargs):

    mag_mean = np.empty_like(x_values, dtype = np.float64)
    mag_std = np.empty_like(x_values, dtype = np.float64)

    for idx_x, x in enumerate(x_values):
        print(f'\nComputing magnetizations for {x_arg} = {x}...')
        kwargs[x_arg] = x
        experiment = lab.Experiment(directory = 'Data', func = exp.mags_onestep, **kwargs)
        experiment.create()
        experiment.run_to(samples)
        mags = experiment.read()
        mag_mean[idx_x] = np.mean(mags)
        mag_std[idx_x] = np.std(mags)

    return mag_mean, mag_std