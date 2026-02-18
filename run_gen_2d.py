import numpy as np
import laboratory as lab
import experiments as exp
from matplotlib import pyplot as plt
from tqdm import tqdm
import theory

neurons = 1000
rank = 2
p = 0.9
# samples = 10
max_it = 200
reduced = 'full'
diagonal = False
initial = 'ex'
alpha_c = 0.138

draw_capacity = rank > alpha_c

r_values = np.linspace(1, 0, num = 50, endpoint = False)[::-1]
m_values = np.linspace(1, 50, num = 50, dtype = int)
t_values = [0, 1, 10, 1000]

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

ms = np.linspace(1, 50, 1000)
seps = [theory.sep_r(alpha=rank / m, m=m, alpha_c=-alpha_c) for m in tqdm(ms[1:])]
ms_red, seps_red = zip(*[(x, y) for x, y in zip(ms[1:], seps) if y is not None])


for t in t_values:
    experiment = lab.Experiment(directory = 'Data_remote', func = exp.gen_mr, m_values = m_values, r_values = r_values,
                                neurons = neurons, rank = rank, t = t, p = p, reduced = reduced, diagonal = diagonal,
                                initial = initial, max_it = max_it)
    m_arc, m_ex, its, errors = experiment.read()
    m_arc_list.append(m_arc)
    m_ex_list.append(m_ex)
    its_list.append(its)
    errors_list.append(errors)

# print(f'Maximum recorded iterations and errors were {np.max(its)} and {np.max(errors)}, respectively')

def draw_plot(array, header, color_scheme, apply_over_samples = np.mean, vmax = 1, draw_cap = draw_capacity):
    fig, axs = plt.subplots(2, 2, sharex = True, sharey = True)

    for idx_t, ax in enumerate(axs.flat):
        c = ax.imshow(np.transpose(np.flip(apply_over_samples(array[idx_t], axis = 0), axis=-1)), cmap=color_scheme, vmin=0, vmax=vmax, aspect='auto',
                      interpolation='nearest',
                      extent=(x_min, x_max, y_min, y_max))

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)


        ax.plot(ms_red, seps_red, color='black', linestyle='dashed')

        if draw_capacity:
            ax.vlines(x=rank / alpha_c, ymin=y_min, ymax=y_max, colors='red')

        ax.set_xlabel(r'$M$')
        ax.set_ylabel(r'$r$')
        ax.label_outer()
        ax.set_title(rf'$t = {t_values[idx_t]}$')

    #fig.supxlabel(r'$M$')
    #fig.supylabel(r'$r$')
    fig.colorbar(c, ax=axs.ravel().tolist())
    fig.suptitle(rf'{header} for $\alpha M = {rank}$')
    plt.show()

draw_plot(m_arc_list, header = 'Archetype recall', color_scheme = 'Blues')
draw_plot(m_ex_list, header = 'Example recall', color_scheme = 'YlOrBr')
draw_plot(its_list, header = 'Maximum iterations', color_scheme = 'Reds', vmax = max_it)
draw_plot(errors_list, header = 'Maximum final errors', color_scheme = 'Greys', apply_over_samples=np.max)
draw_plot(errors_list, header = 'Mean final errors', color_scheme = 'Greys')