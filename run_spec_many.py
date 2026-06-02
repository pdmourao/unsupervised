import experiments as exp
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import theory
from tqdm import tqdm
import sys
import experiments as exp

uni_size = 15

plt.rcParams.update({
    'axes.labelsize': uni_size,
    'axes.titlesize': uni_size,
    'xtick.labelsize': uni_size,
    'ytick.labelsize': uni_size,
})

t_values = [1,10]
alpha = 0.1
m = 50
r_values = [0.3, 0.5]
neurons = 1000
samples = 50

#alpha = theory.sep_alpha(r, m)


#print(theory.dist_roots(alpha = alpha, r = r, m = m, t = t, tol = 1e-2))
#print(theory.sep_r(alpha, m))
#print(theory.dist_roots_full(alpha = alpha, r = r, m = m, t = t, x_max =  1.24))
#[print(theory.peak_cms_diff(alpha = alpha, r = r, m = m, t = t)) for t in range(100)]
#print(theory.t_max_dist(alpha = alpha, r = r, m = m))
#theory.peak_left_tendency(alpha, r, m, t = 87)
#[print(theory.peak_left_tendency(alpha, r, m, tt)) for tt in range(100)]

fig, axs = plt.subplots(len(t_values), len(r_values), sharex = 'col', sharey = 'row')

for idx_t, t in enumerate(t_values):
    for idx_r, r in enumerate(r_values):
        ax = axs[idx_t, idx_r]
        spec = exp.spec_nosave(neurons = neurons, samples = samples, diagonal = False, alpha = alpha, r = r, m = m, t = t)
        ax.hist(np.ravel(spec), bins='fd', density=True)
        x_min, x_max = ax.get_xlim()

        # function for theoretical spectrum
        spec_func = theory.spec_dist(alpha = alpha, r = r, m = m, t = t, diagonal = False)
        disc_func = theory.spec_disc( alpha = alpha, r = r, m = m, t = t)
        print(disc_func(1.23))
        xs = np.linspace(x_min, x_max, num = 10000)
        # compute theoretical spectrum
        ys = [spec_func(x) for x in tqdm(xs)]
        #ys_d = [disc_func(x) for x in tqdm(xs)]

        ax.plot(xs, ys)
        ax.set_xlabel(r'$\lambda$')
        ax.set_ylabel(r'$\tilde{\rho}(\lambda)$')
        #ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=2))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        ax.label_outer()
        #plt.plot(xs, ys_d)
        #plt.ylim(-1, 100)
    axs[idx_t, 0].annotate(rf'$t={t}$', xy=(0, 0.5), xycoords='axes fraction',
                        xytext=(-0.25, 0.5), textcoords='axes fraction',
                        ha='right', va='center', fontsize=uni_size, rotation = 90)
for idx_r, r in enumerate(r_values):
    axs[0, idx_r].set_title(rf'$r={r}$')
plt.tight_layout()
#plt.subplots_adjust(left=0.18)
#plt.subplots_adjust(right=0.2)
plt.show()