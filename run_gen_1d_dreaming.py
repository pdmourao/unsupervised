import numpy as np
import laboratory as lab
import experiments as exp
from matplotlib import pyplot as plt
from tqdm import tqdm
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
print(m_arc)
print(f'Maximum recorded iterations and errors were {np.max(its)} and {np.max(errors)}, respectively')

fig, axs = plt.subplots(2, 1)
axs[0].errorbar(t_values, np.mean(m_arc, axis = 0), np.std(m_arc, axis = 0))

predict = lab.core.prediction(directory = 'Data', func = theory.peak_sep_t, alpha = alpha, r = r, m = m,
                              t_values = t_values, tol = 1e-4)
axs[1].plot(t_values, predict)

axs[1].set_xlabel(r'$t$')
axs[0].set_ylabel(r'$m$')
axs[0].set_ylim(0,1)
axs[1].set_ylabel(r'$\Delta\lambda$')

axs[0].set_title(rf'$\alpha = {alpha}$, $r = {r}$, $M = {m}$, $p = {p}$')
plt.show()