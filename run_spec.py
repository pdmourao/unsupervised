import experiments as exp
import numpy as np
from matplotlib import pyplot as plt
import theory
from tqdm import tqdm
import sys
import experiments as exp



t = 0
num_points = 50
rank = 5

m_values = np.linspace(1, 50, num = num_points, dtype = int)
r_values = np.linspace(theory.sep_r(alpha = rank / m_values[-1], m = m_values[-1]), 0.9, num = num_points, endpoint = False)
idx = 1

idx_m = int(idx / num_points)
idx_r = idx - idx_m * num_points
m = m_values[-idx_m - 1]
r = r_values[idx_r]
print(m)
print(r)
alpha = rank / m


#print(theory.dist_roots(alpha = alpha, r = r, m = m, t = t, tol = 1e-2))
print(theory.sep_r(alpha, m))
#print(theory.dist_roots_full(alpha = alpha, r = r, m = m, t = t, x_max =  1.24))
#[print(theory.peak_cms_diff(alpha = alpha, r = r, m = m, t = t)) for t in range(100)]
print(theory.t_max_dist(alpha = alpha, r = r, m = m))
#theory.peak_left_tendency(alpha, r, m, t = 87)
#[print(theory.peak_left_tendency(alpha, r, m, tt)) for tt in range(100)]

neurons = 500
samples = 1

spec = exp.spec_nosave(neurons = neurons, samples = samples, diagonal = True, alpha = alpha, r = r, m = m, t = t)
plt.hist(np.ravel(spec), bins='fd', density=True)
x_min, x_max = plt.xlim()

print(np.mean(spec))

# function for theoretical spectrum
spec_func = theory.spec_dist(alpha = alpha, r = r, m = m, t = t)
disc_func = theory.spec_disc( alpha = alpha, r = r, m = m, t = t)
print(disc_func(1.23))
xs = np.linspace(x_min, x_max, num = 10000)
# compute theoretical spectrum
ys = [spec_func(x) for x in tqdm(xs)]
ys_d = [disc_func(x) for x in tqdm(xs)]

plt.plot(xs, ys)
#plt.plot(xs, ys_d)
#plt.ylim(-1, 100)
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