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
alpha = 0.02
m = 5
r = 0.6
t = 1
samples = 10
diagonal = True
# function for theoretical spectrum
spec_func = theory.spec_dist(alpha = alpha, r = r, m = m, t = t, diagonal = True)

xmin, xmax = 0, 1
xs = np.linspace(xmin, xmax, num = 100)
# compute theoretical spectrum
ys = [spec_func(x) for x in tqdm(xs)]

print(theory.dist_roots(alpha = alpha, r = r, m = m, t = t))

plt.plot(xs, ys)

plt.ylim(0,1)
plt.xlabel(r'$\lambda$')
plt.ylabel(r'$\rho(\lambda)$')
plt.title('Spectrum')
plt.show()

