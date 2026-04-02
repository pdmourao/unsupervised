import numpy as np
import laboratory as lab
import experiments as exp
from matplotlib import pyplot as plt
from matplotlib import patches
import scipy
import theory
import sys

rank = 5
num_points = 100

m_values = np.linspace(1, 50, num = num_points)
r_values = np.linspace(theory.sep_r(alpha = rank / m_values[-1], m = m_values[-1]), 0.9, num = num_points, endpoint = False)

m_grid, r_grid = np.meshgrid(m_values, r_values, indexing='ij')

pred_crossing = lab.core.prediction(directory ='Predictions', func = theory.t_crossing, vec = theory.vec_mr,
                                    r_values = r_values, m_values = m_values, rank = rank)

pred_max_dist = lab.core.prediction(directory ='Predictions', func = theory.t_max_dist, vec = theory.vec_mr,
                                    r_values = r_values, m_values = m_values, rank = rank)

pred_max_cross_dist = lab.core.prediction(directory ='Predictions', func = theory.t_max_cross_dist, vec = theory.vec_mr,
                                    r_values = r_values, m_values = m_values, rank = rank)

w = 2

vec_max = np.maximum(pred_crossing, pred_max_dist)
vec_tale = np.where(pred_max_dist > pred_crossing, pred_max_dist, np.nan)
sep_line = [theory.sep_r(alpha = rank / m, m = m) for m in m_values]

fig, ax = plt.subplots()
c = ax.contourf(m_grid, r_grid, pred_max_cross_dist, levels = 50)
ax.set_xlabel(r'$M$')
ax.set_ylabel(r'$r$')
ax.set_title(rf'Optimal $t$ for $\alpha M = {rank}$')
ax.autoscale(False)
ax.plot(m_values, sep_line, linestyle = 'dashed', color = 'black', linewidth=w)
fig.colorbar(c, ax = ax)
plt.show()

