import experiments as exp
import numpy as np
from matplotlib import pyplot as plt
import theory
from tqdm import tqdm
import laboratory as lab
from auxiliary import mags_onestep_1d
from laboratory.systems import Dream as dream
from scipy.optimize import root_scalar

kwargs_gen_hr = {'t': 1,
                 'alpha' : 0.1,
                 'm': 50,
                 'r': 0.6
                 }

kwargs_of = {'t': 1,
             'alpha' : 0.02,
             'm': 5,
             'r': 0.2
             }

kwargs_gen_lr = {'t': 1,
                 'alpha' : 0.02,
                 'm': 5,
                 'r': 0.7
                 }

kwargs_sg = {'t': 1,
             'alpha' : 0.1,
             'm': 20,
             'r': 0.2
             }

# kwargs = kwargs_gen_hr
neurons = 1000
alpha = 0.2
m = 50
r = 0.6
t_values = [0, 4, 10, 30]

def plot_spec(ax, t, ylim = None):
# function for theoretical spectrum
    spec_func = theory.spec_dist(alpha = alpha, r = r, m = m, t = t, diagonal = True)

    xmin, xmax = 0, 1
    xs = np.linspace(xmin, xmax, num = 100)
    # compute theoretical spectrum
    ys = [spec_func(x) for x in tqdm(xs)]

    ax.plot(xs, ys)
    if ylim is not None:
        ax.set_ylim(0,ylim)
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel(r'$\rho(\lambda)$')
    ax.label_outer()
    ax.set_title(rf'$t = {t}$')

fig, axs = plt.subplots(len(t_values),1, sharex=True)

for idx, ax in enumerate(axs.flat):
    t = t_values[idx]
    plot_spec(ax, t)

plt.show()

