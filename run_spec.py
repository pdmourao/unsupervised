import experiments as exp
import numpy as np
from matplotlib import pyplot as plt
import theory
from tqdm import tqdm

kwargs = {'alpha' : 0.2,
          'm': 20,
          'r': 0.5,
          'diagonal': False
          }

# experimental spectrum
# spec = exp.spectrum(samples = 10, neurons = 500, alpha = alpha, m = m, r = r, diagonal = diagonal)
# function for theoretical spectrum
spec_func = theory.spec_dist(**kwargs)

# plt.hist(np.ravel(spec), bins = 50, density = True)

xmin, xmax = 0, 1
# xmin, xmax = plt.xlim()
xs = np.linspace(xmin, xmax, num = 100)
# compute theoretical spectrum
ys = [spec_func(x) for x in tqdm(xs)]
plt.plot(xs, ys)
# plt.ylim(0,1)
plt.show()
