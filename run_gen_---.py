import experiments as exp
import numpy as np
from matplotlib import pyplot as plt
import theory
from tqdm import tqdm

kwargs = {'alpha' : 0.1,
          'm': 20,
          'r': 0.5,
          'diagonal': False
          }
neurons = 500

# experimental spectrum
# spec = exp.spectrum(samples = 10, neurons = neurons, **kwargs)
# function for theoretical spectrum
spec_func = theory.spec_dist(**kwargs)

# plt.hist(np.ravel(spec), bins = 50, density = True)

xmin, xmax = 0, 1
# xmin, xmax = plt.xlim()
xs = np.linspace(xmin, xmax, num = 100)
# compute theoretical spectrum
ys = [spec_func(x) for x in tqdm(xs)]
plt.plot(xs, ys)
plt.ylim(0,1)
plt.show()

ps = np.linspace(1, 0, num = 100, endpoint = False)
ms_ex = np.array([theory.mags(p = p, attractor = 'ex', **kwargs) for p in tqdm(ps)])
plt.plot(ps, ms_ex, label = 'examples')

ms_arc = np.array([theory.mags(p = p, attractor = 'arc', **kwargs) for p in tqdm(ps)])
plt.plot(ps, ms_arc, label = 'archetypes')
plt.legend()
plt.gca().invert_xaxis()
plt.ylim(0,1)
plt.show()

mags_mc, errors_mc = exp.attraction_mc(neurons = neurons, initial = 'arc', entropy = None, **kwargs)

plt.hist(np.ravel(mags_mc), bins = 50, density = True)
plt.xlim(0,1)
plt.show()

plt.scatter(np.arange(len(errors_mc)), errors_mc)
plt.show()
