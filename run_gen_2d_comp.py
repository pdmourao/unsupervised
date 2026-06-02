import sys

import numpy as np
import laboratory as lab
import experiments as exp
from matplotlib import pyplot as plt
from tqdm import tqdm
import theory

neurons = 1000
ranks = [0.1, 2]
p = 0.9
# samples = 10
max_it = 200
reduced = 'full'
diagonal = False
initial = 'ex'
alpha_c = 0.138

r_values = np.linspace(1, 0, num = 50, endpoint = False)[::-1]
m_values = np.linspace(1, 50, num = 50, dtype = int)
t = 0
n_lines = len(ranks)

m_arc_list = []
m_ex_list = []
its_list = []
errors_list = []


x_min, x_max = m_values[0].astype(float).item(), m_values[-1].astype(float).item()
y_min, y_max = r_values[0].item(), r_values[-1].item()

step_x = (m_values[1] - m_values[0])/2
step_y = (r_values[1] - r_values[0])/2

x_min -= step_x
x_max += step_x
y_min -= step_y
y_max += step_y

uni_size = 12
plt.rcParams.update({
            'axes.labelsize': uni_size,
            'axes.titlesize': uni_size,
            'axes.titlesize': uni_size,
            'figure.titlesize': uni_size,
            'xtick.labelsize': uni_size,
            'ytick.labelsize': uni_size,
        })
fig, axs = plt.subplots(n_lines, 2, sharex = True, sharey = True)

for idx_r, rank in enumerate(ranks):
    ms = np.linspace(1, 50, 1000)
    seps = [theory.sep_r(alpha=rank / m, m=m) for m in tqdm(ms[1:])]
    ms_red, seps_red = zip(*[(x, y) for x, y in zip(ms[1:], seps) if y is not None])

    experiment = lab.Experiment(directory = 'Data', func = exp.gen_mr, m_values = m_values, r_values = r_values,
                                neurons = neurons, rank = rank, t = t, p = p, reduced = reduced, diagonal = diagonal,
                                initial = initial, max_it = max_it)
    m_arc, m_ex, its, errors = experiment.read()
    c_arc = axs[idx_r,0].imshow(np.transpose(np.flip(np.mean(np.abs(m_arc), axis = 0), axis=-1)), cmap='Blues', vmin=0, vmax=1, aspect='auto',
                        interpolation='nearest',
                        extent=(x_min, x_max, y_min, y_max))
    c_ex = axs[idx_r,1].imshow(np.transpose(np.flip(np.mean(np.abs(m_ex), axis = 0), axis=-1)), cmap='YlOrBr', vmin=0, vmax=1, aspect='auto',
                        interpolation='nearest',
                        extent=(x_min, x_max, y_min, y_max))
    plt.colorbar(c_arc, ax=axs[idx_r, 0])
    plt.colorbar(c_ex, ax=axs[idx_r, 1])
    axs[idx_r, 0].annotate(rf'$\alpha M={rank}$', xy=(0, 0.5), xycoords='axes fraction',
                        xytext=(-0.4, 0.5), textcoords='axes fraction',
                        ha='right', va='center', fontsize=uni_size, rotation = 90)
    for ax in axs[idx_r]:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        ax.set_xlabel(r'$M$')
        ax.set_ylabel(r'$r$')
        ax.label_outer()

        ax.plot(ms_red, seps_red, color='black', linestyle='dashed', lw = 2)

        if rank > alpha_c:
            ax.vlines(x=rank / alpha_c, ymin=y_min, ymax=y_max, colors='red', linewidth = 2)
axs[0, 0].set_title('Archetype recall')
axs[0, 1].set_title('Example recall')
plt.subplots_adjust(left=0.15)
plt.show()