import experiments as exp
import numpy as np
from matplotlib import pyplot as plt
import theory
from tqdm import tqdm
import laboratory as lab
from auxiliary import mags_onestep_1d

kwargs_1 = {'alpha' : 0.1,
          'm': 50,
          'r': 0.6,
          'diagonal': False
          }
kwargs_2 = {'alpha' : 0.05,
          'm': 5,
          'r': 0.2,
          'diagonal': False
          }
kwargs_3 = {'alpha' : 0.02,
          'm': 5,
          'r': 0.2,
          'diagonal': False
          }
kwargs = kwargs_3
neurons = 1000
samples = 30
max_it_mc = 200

# experimental spectrum
experiment = lab.Experiment(directory = 'Data', func = exp.spectrum, neurons = neurons, **kwargs)
experiment.run_to(samples)
spec = experiment.read()
# function for theoretical spectrum
spec_func = theory.spec_dist(**kwargs)

plt.hist(np.ravel(spec), bins = 50, density = True)

xmin, xmax = plt.xlim()
xs = np.linspace(xmin, xmax, num = 100)
# compute theoretical spectrum
ys = [spec_func(x) for x in tqdm(xs)]
plt.plot(xs, ys)
# plt.ylim(0,1)
plt.show()

p_values = np.linspace(1,0.1, num = 10)
magarc_mean, magarc_std, = mags_onestep_1d(x_arg = 'p', x_values = p_values, samples = samples, neurons = neurons,
                                           initial = 'arc', attractor = 'arc', **kwargs)
parc_std=plt.errorbar(p_values, magarc_mean, magarc_std, linestyle='None', marker='^', color = 'blue')

magex_mean, magex_std = mags_onestep_1d(x_arg = 'p', x_values = p_values, samples = samples, neurons = neurons,
                                        initial = 'ex', attractor = 'ex', **kwargs)
pex_std=plt.errorbar(p_values, magex_mean, magex_std, linestyle='None', marker='^', color = 'orange')

ps = np.linspace(1, 0, num = 100, endpoint = False)
ms_ex = np.array([theory.mags(p = p, attractor = 'ex', **kwargs) for p in tqdm(ps)])
plt.plot(ps, ms_ex, label = 'examples', color = 'orange')

ms_arc = np.array([theory.mags(p = p, attractor = 'arc', **kwargs) for p in tqdm(ps)])
plt.plot(ps, ms_arc, label = 'archetypes', color = 'blue')
plt.legend()
plt.gca().invert_xaxis()
plt.ylim(0,1)
plt.show()

experiment = lab.Experiment(directory = 'Data', func = exp.attraction_mc, neurons = neurons, initial = 'arc',
                            max_it = max_it_mc, **kwargs)
experiment.run_to(samples)
mags_arc_mc, mags_ex_mc, errors_mc = experiment.read()

plt.hist(np.ravel(mags_arc_mc), bins = 'fd', density = True, color = 'blue')
plt.xlim(0,1)
plt.show()

plt.hist(np.ravel(mags_ex_mc), bins = 'fd', density = True, color = 'orange')
plt.xlim(0,1)
plt.show()

print(f'Max final error across all samples was {np.max(errors_mc)}')
