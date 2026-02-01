import laboratory as lab
import experiments as exp
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import theory

state = 'arc'

kwargs = {'neurons': 2000,
          'alpha': 0.05,
          'r': 0.5,
          'm': 20,
          'initial': state,
          'attractor': state,
          'p': 0.5,
          'diagonal': False
          }

deltas_nd = exp.delta(**kwargs)
kwargs.pop('initial')
kwargs.pop('neurons')
dist_func = theory.delta_dist(**kwargs)

plt.hist(np.ravel(deltas_nd), bins=50, density=True)

xmin, xmax = plt.xlim()

xs = np.linspace(xmin, xmax, num = 100)
ys = [dist_func(x) for x in tqdm(xs)]

plt.plot(xs, ys)
plt.show()