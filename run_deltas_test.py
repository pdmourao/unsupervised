import laboratory as lab
import experiments as exp
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import scipy

import theory
from theory import double_peak

state = 'ex'

n = 3000
r = 0.5
p = 0.5
m = 20

kwargs = {'alpha': 0,
          'r': r,
          'm': m,
          'attractor': state,
          'p': p,
          'diagonal': False
          }

probs = [(1 + r) / 2, (1 - r) / 2]
avs = [p/m + r ** 3 * p,
       p/m - r ** 3 * p]

std = np.sqrt(r ** 4 * p ** 2 * (m - 1) * (1 + r) * (1-r) / (m ** 2))

stds = np.full(shape = 2, fill_value = std)

def gausss(x):
    return double_peak(x, probs, avs, stds)

deltas_nd = exp.delta_test(samples = 20, initial = state, neurons = n, **kwargs)
print(np.shape(deltas_nd))
print(deltas_nd)
plt.hist(np.ravel(deltas_nd), bins = 30, density=True)

xmin, xmax = plt.xlim()

xs = np.linspace(xmin, xmax, num = 100)
ys = [gausss(x) for x in tqdm(xs)]

plt.plot(xs, ys)
plt.show()