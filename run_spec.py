import experiments as exp
import numpy as np
from matplotlib import pyplot as plt
import theory
from tqdm import tqdm
import sys

kwargs = {'t' : 0,
          'm': 10,
          'r': 0.6,
          'alpha': 0.1,
          'diagonal': False
          }

kwargs['alpha'] = theory.sep_alpha(kwargs['r'], kwargs['m'])
print(kwargs['alpha'])
# function for theoretical spectrum
spec_func = theory.spec_dist(**kwargs)

xs = np.linspace(-kwargs['alpha'], 1, num = 10000)
# compute theoretical spectrum
ys = [spec_func(x) for x in tqdm(xs)]
plt.plot(xs, ys)
plt.ylim(0,1)
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