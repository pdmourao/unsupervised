import experiments as exp
import laboratory as lab
import numpy as np
from matplotlib import pyplot as plt

t_values = np.arange(5, 51, step = 5)
n_values = [250, 500, 1000, 2000, 4000]
markers = ['o', '^', 'd', '+', 'x']
r = 0.5
m = 50
alpha = 0.1

for idx_n, n in enumerate(n_values):
    experiment = lab.Experiment(directory = 'Data', func = exp.diag_std_t, n = n, r = r, m = m, alpha = alpha, t_values = t_values)
    experiment.create()
    experiment.run_to(1)
    means, stds = experiment.read()

    plt.scatter(t_values, stds*np.sqrt(n), label = rf'$N={n}$', marker = markers[idx_n], s = 20)
plt.ylim(0,0.15)
plt.xlabel(r'$t$')
plt.ylabel(r'$\sigma\times\sqrt{N}$')
plt.title(rf'$r = {r}$')
plt.legend()
plt.show()
