import numpy as np
import laboratory as lab
import experiments as exp
from matplotlib import pyplot as plt
import scipy
import theory

neurons = 1000
alpha = 0.2
m = 50
r = 0.5
p = 0.9
max_it = 200
reduced = 'full'
diagonal = False
initial = 'ex'


t_values = np.linspace(0, 50, num = 101)

experiment = lab.Experiment(directory = 'Data_remote', func = exp.gen_t, m = m, r = r, t_values = t_values,
                            neurons = neurons, alpha = alpha, p = p, reduced = reduced, diagonal = diagonal,
                            initial = initial, max_it = max_it)
m_arc, m_ex, its, errors = experiment.read()

print(f'Maximum recorded iterations and errors were {np.max(its)} and {np.max(errors)}, respectively')
roots = theory.dist_roots(alpha, r, m, 0)
cm1, cm2 = theory.peak_cms(alpha, r, m, 0)
print(roots)
print(cm1, cm2)
fig, axs = plt.subplots(3, 1)
axs[0].errorbar(t_values, np.mean(m_arc, axis = 0), np.std(m_arc, axis = 0))

predict_gap = lab.core.prediction(directory = 'Data', func = theory.peak_sep_t, alpha = alpha, r = r, m = m,
                              t_values = t_values, tol = 1e-4)
predict_cm = lab.core.prediction(directory = 'Data', func = theory.peak_cms_diff_t, alpha = alpha, r = r, m = m,
                              t_values = t_values, tol = 1e-4)
predict_gaps = [theory.transf(roots[2], t) - theory.transf(roots[1], t) for t in t_values]
predict_cms = [theory.transf(cm2, t) - theory.transf(cm1, t) for t in t_values]
#axs[1].plot(t_values, predict_gap)
axs[1].plot(t_values, predict_gaps)
#axs[2].plot(t_values, predict_cm)
axs[2].plot(t_values, predict_cms)

axs[0].set_ylabel(r'$m$')
axs[0].set_ylim(0,1)

axs[1].set_xlabel(r'$t$')
axs[1].set_ylabel(r'$\Delta\lambda$')
axs[1].set_title('Gap between the peaks')
axs[2].set_xlabel(r'$t$')
axs[2].set_ylabel(r'$\Delta\lambda$')
axs[2].set_title('Gap between the centers of mass')
plt.tight_layout()
axs[0].set_title(rf'$\alpha = {alpha}$, $r = {r}$, $M = {m}$, $p = {p}$')
plt.show()

rank = alpha*m
mmin = 10
mmax = 500
ms = np.linspace(mmin, mmax, 100)
seps = [theory.sep_r(alpha = rank / m, m = m) for m in ms]
plt.plot(ms, seps)
plt.ylim(0, 1)
plt.scatter([m], [r])
plt.show()