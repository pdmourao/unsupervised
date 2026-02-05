import experiments as exp
import numpy as np
from matplotlib import pyplot as plt
import theory
from tqdm import tqdm

kwargs = {'alpha' : 0.01,
          'm': 50,
          'r': 0.2,
          'diagonal': False
          }


# function for theoretical spectrum
spec_func = theory.spec_dist(**kwargs)

xs = np.linspace(0, 1, num = 100)
# compute theoretical spectrum
ys = [spec_func(x) for x in tqdm(xs)]
plt.plot(xs, ys)
# plt.ylim(0,1)
plt.show()

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