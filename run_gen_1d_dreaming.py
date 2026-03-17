import numpy as np
import laboratory as lab
import experiments as exp
from matplotlib import pyplot as plt
import scipy
import theory
from tqdm import tqdm

neurons = 1000
alpha = 0.1
m = 50
r = 0.6
p = 0.9
max_it = 200
reduced = 'full'
diagonal = False
initial = 'ex'

epsilon = 0

t_values = np.linspace(0, 50, num = 101)

#experiment = lab.Experiment(directory = 'Data', func = exp.gen_t, m = m, r = r, t_values = t_values,
#                            neurons = neurons, alpha = alpha, p = p, reduced = reduced, diagonal = diagonal,
#                            initial = initial, max_it = max_it)
#m_arc, m_ex, its, errors = experiment.read()

#print(f'Maximum recorded iterations and errors were {np.max(its)} and {np.max(errors)}, respectively')
roots = theory.dist_roots(alpha, r, m, 0)
cm1, cm2 = theory.peak_cms(alpha, r, m, 0)
print(roots)
print(cm1, cm2)
fig, axs = plt.subplots(1, 1)
#axs[0].errorbar(t_values, np.mean(m_arc, axis = 0), np.std(m_arc, axis = 0))

predict_gap = lab.core.prediction(directory = 'Predictions', func = theory.peak_sep_t, alpha = alpha, r = r, m = m,
                              t_values = t_values, tol = 1e-4)
predict_cm = lab.core.prediction(directory = 'Predictions', func = theory.peak_cms_diff_t, alpha = alpha, r = r, m = m,
                              t_values = t_values, tol = 1e-4)
left_max = lab.core.prediction(directory = 'Predictions', func = theory.peak_left_max_t, alpha = alpha, r = r, m = m,
                              t_values = t_values, tol = 1e-4)[0]
left_cm = lab.core.prediction(directory = 'Predictions', func = theory.peak_left_cm_t, alpha = alpha, r = r, m = m,
                              t_values = t_values, tol = 1e-4)
predict_gaps = [theory.transf(roots[2], t) - theory.transf(roots[1], t) for t in t_values]
predict_cms = [theory.transf(cm2, t) - epsilon * theory.transf(cm1, t) for t in t_values]
#axs[1].plot(t_values, predict_gap)
#axs[1].plot(t_values, predict_gaps)
#axs[2].plot(t_values, predict_cm)
#axs[2].plot(t_values, predict_cm)
#axs[0].plot(t_values, left_max)
#axs[0].plot(t_values, left_cm)
#axs[0].set_ylabel(r'$m$')
#axs[0].set_ylim(0,1)

axs.plot(t_values, left_max)
axs.plot(t_values, left_cm)

#axs[1].set_xlabel(r'$t$')
#axs[1].set_ylabel(r'$\Delta\lambda$')
#axs[1].set_title('Gap between the peaks')
#axs[2].set_xlabel(r'$t$')
#axs[2].set_ylabel(r'$\Delta\lambda$')
#axs[2].set_title('Gap between the centers of mass')
#axs[0].set_title(rf'$\alpha = {alpha}$, $r = {r}$, $M = {m}$, $p = {p}$')

axs.set_title(rf'$\alpha = {alpha}$, $r = {r}$, $M = {m}$')
plt.show()

t_diff = left_max - left_cm
t_cross = None
for idx_t in range(len(t_values) - 1):
    if t_diff[idx_t] * t_diff[idx_t+1] < 0:
        t_cross = (t_values[idx_t] + t_values[idx_t+1])/2
        break
print(rf'Crossing t found to be $t = {t_cross}$.')
xs = np.linspace(0, 1, 1001)

experiment = lab.Experiment(directory = 'Data_remote', func = exp.spectrum, neurons = 1000, alpha = alpha, r = r, m = m,
                            t = t_cross, diagonal = True)
spec_exp = experiment.read()
plt.hist(np.ravel(spec_exp), bins=50, density=True)

spec_f = theory.spec_dist(alpha = alpha, r = r, m = m, t = t_cross)
spec_values = [spec_f(x) for x in tqdm(xs)]
plt.plot(xs, spec_values)
plt.title(rf'$\alpha = {alpha}$, $r = {r}$, $M = {m}$, $t = {t_cross}$')
plt.show()

t = 10
spec_f = theory.spec_dist(alpha = alpha, r = r, m = m, t = t)
spec_values = [spec_f(x) for x in tqdm(xs)]
plt.plot(xs, spec_values)
plt.title(rf'$\alpha = {alpha}$, $r = {r}$, $M = {m}$, $t = {t}$')
plt.show()