import numpy as np
import laboratory as lab
import experiments as exp
from matplotlib import pyplot as plt
from matplotlib import patches
import scipy
import theory
import sys
from tqdm import tqdm 

rank = 5
num_points = 100

plt.rcParams.update({
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
})

m_values = np.linspace(1, 50, num = num_points, endpoint = False)
r_values = np.linspace(theory.sep_r(alpha = rank / m_values[-1], m = m_values[-1])-0.01, 0.9, num = num_points, endpoint = False)

#m = 45.05050505050505
#r = 0.9
#t = 94.16830618882699
#x_max = theory.dist_max(alpha = rank / m, r = r, m = m, t = 0, prints = False)
#print(f'Calculated x_max to be {x_max}')
#weird = theory.dist_roots_full(alpha = rank/m, r = r, m = m, t = t, x_max = x_max)

m_grid, r_grid = np.meshgrid(m_values, r_values, indexing='ij')

pred_max_cross_dist = lab.core.prediction(directory ='Predictions', func = theory.t_max_cross_dist, vec = theory.vec_mr,
                                    r_values = r_values, m_values = m_values, rank = rank)

w = 2

sep_line = [theory.sep_r(alpha = rank / m, m = m) for m in m_values]

fig, ax = plt.subplots()
c = ax.contourf(m_grid, r_grid, pred_max_cross_dist, levels = 50)
ax.set_xlabel(r'$M$')
ax.set_ylabel(r'$r$')
ax.set_title(rf'Optimal $t$ for $\alpha M = {rank}$')
ax.autoscale(False)
ax.plot(m_values, sep_line, linestyle = 'dashed', color = 'black', linewidth=w)
fig.colorbar(c, ax = ax)
alpha_c=0.138
ax.vlines(x=rank / alpha_c, ymin=r_values[0], ymax=r_values[-1], colors='red')
plt.show()

