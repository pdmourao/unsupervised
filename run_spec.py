import experiments as exp
import numpy as np
from matplotlib import pyplot as plt
import theory
from tqdm import tqdm
import sys
import experiments as exp

kwargs = {'t' : 0,
          'm': 50,
          'r': 0.5,
          'alpha': 0.1,
          'diagonal': False
          }
neurons = 500
samples = 50

spec = exp.spec_nosave(neurons = neurons, samples = samples, **kwargs)
plt.hist(np.ravel(spec), bins='fd', density=True)
x_min, x_max = plt.xlim()

# function for theoretical spectrum
spec_func = theory.spec_dist(**kwargs)
xs = np.linspace(x_min, x_max, num = 10000)
# compute theoretical spectrum
ys = [spec_func(x) for x in tqdm(xs)]

plt.plot(xs, ys)
plt.show()


sys.exit()
ps = np.linspace(1, 0, num = 100, endpoint = False)

ms_ex = np.array([theory.mags(p = p, attractor = 'ex', **kwargs) for p in tqdm(ps)])
plt.plot(ps, ms_ex, label = 'examples', color = 'orange')

ms_arc = np.array([theory.mags(p = p, attractor = 'arc', **kwargs) for p in tqdm(ps)])
plt.plot(ps, ms_arc, label = 'archetypes', color = 'blue')

plt.plot(ps, ps, color = 'grey', linestyle = 'dashed')

plt.legend()
plt.gca().invert_xaxis()
plt.ylim(0,1)
plt.show()