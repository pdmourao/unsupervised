import experiments as exp
import numpy as np
from matplotlib import pyplot as plt
import theory
from tqdm import tqdm
import laboratory as lab
from auxiliary import mags_onestep_1d
from laboratory.systems import Dream as dream

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
alpha = 0.1
m = 50
r = 0.6
t = 10
samples = 10
diagonal = True

# function for theoretical spectrum
spec_func = theory.spec_dist(alpha = alpha, r = r, m = m, t = t, diagonal = True)
spec_func0 = theory.spec_dist(alpha = alpha, r = r, m = m, t = 0, diagonal = True)


k=int(alpha * neurons)

entropy = np.random.SeedSequence(None).entropy
system = dream(neurons=neurons, k= k, rng_ss = np.random.SeedSequence(entropy), diagonal = diagonal, t = t, m = m, r = r)
system.set_interaction()
print(f'Diagonal with mean {np.mean(np.diagonal(system.J))} with std {np.std(np.diagonal(system.J))}')
print(f'Normalized std {np.std(np.diagonal(system.J))*np.sqrt(neurons)}')
# system.diagonal = False
evs = np.real_if_close(np.linalg.eigvals(system.J), tol=1e-3)
# print(evs)
plt.hist(np.ravel(evs), bins = 'fd', density = True)

# xmin, xmax = plt.xlim()
xmin, xmax = 0, 1
xs = np.linspace(xmin, xmax, num = 100)
# compute theoretical spectrum
ys = [spec_func(x) for x in tqdm(xs)]
ys0 = [spec_func0(x) for x in tqdm(xs)]
plt.plot(xs, ys)
plt.plot(xs, ys0)
plt.ylim(0,1)
plt.xlabel(r'$\lambda$')
plt.ylabel(r'$\rho(\lambda)$')
plt.title('Spectrum')
plt.show()

