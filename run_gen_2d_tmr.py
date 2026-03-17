import numpy as np
import laboratory as lab
import experiments as exp
from matplotlib import pyplot as plt
from tqdm import tqdm
import theory

neurons = 1000
rank = 5
p = 1
# samples = 10
max_it = 200
reduced = 'full'
diagonal = False
initial = 'new_ex'
alpha_c = 0.138

draw_capacity = rank > alpha_c

t_values = np.linspace(0, 49, num = 50)
m_values = np.linspace(11, 60, num = 50, dtype = int)
r_buffer = 0.05

x_min, x_max = t_values[0].astype(float).item(), t_values[-1].astype(float).item()
y_min, y_max = m_values[0].item(), m_values[-1].item()

step_x = (t_values[1] - t_values[0])/2
step_y = (m_values[1] - m_values[0])/2

x_min -= step_x
x_max += step_x
y_min -= step_y
y_max += step_y



experiment = lab.Experiment(directory = 'Data_remote', func = exp.gen_tm_transition, m_values = m_values, r_buffer = r_buffer,
                                neurons = neurons, rank = rank, t_values = t_values, p = p, reduced = reduced, diagonal = diagonal,
                                initial = initial, max_it = max_it)
m_arc, m_ex, its, errors = experiment.read()

# print(f'Maximum recorded iterations and errors were {np.max(its)} and {np.max(errors)}, respectively')

def draw_plot(array, header, color_scheme, apply_over_samples = np.mean, vmax = 1, draw_cap = draw_capacity):
    idx_t = 0
    fig, ax = plt.subplots(1)

    c = ax.imshow(np.transpose(np.flip(apply_over_samples(np.abs(array), axis=0), axis=-1)),
                  cmap=color_scheme, vmin=0, vmax=vmax, aspect='auto',
                  interpolation='nearest',
                  extent=(x_min, x_max, y_min, y_max))

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    if draw_capacity:
        ax.vlines(x=rank / alpha_c, ymin=y_min, ymax=y_max, colors='red')

    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$M$')
    ax.label_outer()
    ax.set_title(rf'Dreaming generalization for $\alpha M = {rank}$')

    fig.colorbar(c, ax=ax)
    fig.suptitle(rf'{header} for $\alpha M = {rank}$')
    plt.show()

draw_plot(m_arc, header = 'Archetype recall', color_scheme = 'Blues')
draw_plot(m_arc, header = 'Maximum archetype recall', color_scheme = 'Blues', apply_over_samples = np.max)
draw_plot(m_ex, header = 'Example recall', color_scheme = 'YlOrBr')
draw_plot(m_ex, header = 'Maximum example recall', color_scheme = 'YlOrBr', apply_over_samples = np.max)
draw_plot(its, header = 'Maximum iterations', color_scheme = 'Reds', vmax = max_it, apply_over_samples=np.max)
draw_plot(errors, header = 'Maximum final errors', color_scheme = 'Greys', apply_over_samples=np.max)
draw_plot(errors, header = 'Mean final errors', color_scheme = 'Greys')

fig, ax = plt.subplots(1)

m_a = np.mean(m_arc, axis = 0)
m_e = np.mean(m_ex, axis = 0)
m_max = np.maximum(m_a, m_e)

# Boolean mask: where A wins
a_wins = m_a >= m_e

# Mask the values
masked_arc = np.ma.masked_where(~a_wins, m_max)
masked_ex = np.ma.masked_where(a_wins, m_max)

# Plot using different colormaps

im_arc = ax.imshow(np.transpose(np.flip(masked_arc, axis=-1)), cmap='Blues', vmin=0, vmax=1, aspect='auto',
                    interpolation='nearest',
                    extent=(x_min, x_max, y_min, y_max))

im_ex = ax.imshow(np.transpose(np.flip(masked_ex, axis=-1)), cmap='YlOrBr', vmin=0, vmax=1,
                  aspect='auto',
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
cbar_ex = fig.colorbar(im_ex, ax=axs.ravel().tolist(), pad = -0.07)
cbar_arc = fig.colorbar(im_arc, ax=axs.ravel().tolist())
cbar_arc.set_ticks([])
fig.suptitle(rf'Arc vs Examples for $\alpha M = {rank}$')
plt.show()

print(m_arc_list[3][:,25,25])
print(m_ex_list[3][:,25,25])