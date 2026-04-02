import numpy as np
import laboratory as lab
import experiments as exp
from matplotlib import pyplot as plt
import sys
import theory

neurons = 1000
ranks = [2, 5]
p = 1
# samples = 10
max_it = 200
reduced = 'full'
diagonal = False
initial = 'new_ex'
tol = 1e-4
num_points = 100

r_values = np.array([0.3, 0.5, 0.7])


draw_capacity = False

t_values = np.linspace(0, 49, num = 50, endpoint = True)
m_values = np.linspace(1, 50, num = 50, dtype = int)


fig1, axs1 = plt.subplots(len(ranks), len(r_values))
fig2, axs2 = plt.subplots(len(ranks), len(r_values))


def draw_plot(array, ax, header, color_scheme, x_min, x_max, y_min, y_max, apply_over_samples=np.mean, vmax=1):
    c = ax.imshow(np.transpose(np.flip(apply_over_samples(np.abs(array), axis=0), axis=-1)),
                  cmap=color_scheme, vmin=0, vmax=vmax, aspect='auto',
                  interpolation='nearest',
                  extent=(x_min, x_max, y_min, y_max))

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$M$')
    ax.set_title(rf'$\alpha M = {rank}$')

    return c

for idx_rank, rank in enumerate(ranks):
    for idx_r, r in enumerate(r_values):
        m_values_alt = np.copy(m_values)
        while theory.sep_r(alpha=rank / m_values_alt[0], m=m_values_alt[0]) > r:
            m_values_alt += 1
        try:
            experiment = lab.Experiment(directory = 'Data_remote', func = exp.gen_tm, m_values = m_values, r = r,
                                        neurons = neurons, rank = rank, t_values = t_values, p = p, reduced = reduced, diagonal = diagonal,
                                        initial = initial, max_it = max_it)
        except AssertionError:
            experiment = lab.Experiment(directory='Data_remote', func=exp.gen_tm, m_values=m_values_alt, r=r,
                                        neurons=neurons, rank=rank, t_values=t_values, p=p, reduced=reduced,
                                        diagonal=diagonal,
                                        initial=initial, max_it=max_it)

        m_arc, m_ex, its, errors = experiment.read()


        x_min, x_max = t_values[0].astype(float).item(), t_values[-1].astype(float).item()
        y_min, y_max = m_values_alt[0].item(), m_values_alt[-1].item()

        step_x = (t_values[1] - t_values[0])/2
        step_y = (m_values_alt[1] - m_values_alt[0])/2

        x_min -= step_x
        x_max += step_x
        y_min -= step_y
        y_max += step_y

        t_v = np.linspace(t_values[0], t_values[-1], num = num_points)
        m_v = np.linspace(m_values_alt[0], m_values_alt[-1], num = num_points)

        t_grid, m_grid = np.meshgrid(t_v, m_v, indexing='ij')

        #pred_right_cm = lab.core.prediction(directory ='Predictions', func = theory.peak_right_cm, vec = theory.vec_tm,
        #                                        t_values = t_v, m_values = m_v, rank = rank, r = r)

        #pred_left_cm = lab.core.prediction(directory ='Predictions', func = theory.peak_left_cm, vec = theory.vec_tm,
        #                                       t_values = t_v, m_values = m_v, rank = rank, r = r)

        #pred_left_max = lab.core.prediction(directory = 'Predictions', func = theory.peak_left_max, vec = theory.vec_tm,
         #                              t_values = t_v, m_values = m_v, rank = rank, r = r)

        #pred_right_max = lab.core.prediction(directory = 'Predictions', func = theory.peak_right_max, vec = theory.vec_tm,
        #                               t_values = t_v, m_values = m_v, rank = rank, r = r)

        #pred_roots = lab.core.prediction(directory = 'Predictions', func = theory.dist_roots_full, vec = theory.vec_tm,
        #                               t_values = t_v, m_values = m_v, rank = rank, r = r)

        #pred_cm = lab.core.prediction(directory = 'Predictions', func = theory.dist_cm, vec = theory.vec_tm, t_values = t_v,
        #                              m_values = m_v, rank = rank, r = r)

        #pred_right_cm -= pred_cm
        #pred_left_cm -= pred_cm
        #pred_right_max -= pred_cm
        #pred_left_max -= pred_cm

        #for idx in range(len(pred_roots)):

        #    pred_roots[idx] -= pred_cm

        # print(f'Maximum recorded iterations and errors were {np.max(its)} and {np.max(errors)}, respectively')

        ax1 = axs1[idx_rank, idx_r]
        ax2 = axs2[idx_rank, idx_r]
        #pred_diff_right = np.where(t_grid > 10, pred_right_cm - pred_left_cm, np.nan)
        c = draw_plot(m_arc, ax = ax1, header = 'Archetype recall', color_scheme = 'Blues', x_min = x_min, x_max = x_max, y_min = y_min, y_max = y_max)
        fig1.colorbar(c, ax=ax1)
        #plt.contour(t_grid, m_grid, pred_left_max, levels = [0], colors ='green', linestyles ='dashed')
        #peak1 = ax1.contour(t_grid, m_grid, pred_left_max-pred_left_cm, levels = [0], colors ='red', linestyles ='dashed')
        #cms = ax1.contour(t_grid, m_grid, pred_right_cm - pred_left_cm, levels = [0.35], colors ='black', linestyles ='dashed')
        #plt.contour(t_grid, m_grid, pred_right_cm-pred_left_cm, levels = [0.2], colors ='black', linestyles ='dashed')
        #plt.contour(t_grid, m_grid, pred_right_cm - pred_left_cm, levels = [0.2], colors ='black', linestyles ='dashed')
        #plt.show()

        #ax2.contourf(t_grid, m_grid, pred_left_max-pred_left_cm)
        #plt.show()
        #c = ax2.contourf(t_grid, m_grid, pred_right_cm - pred_left_cm)
        #ax2.title(rf'$\alpha M = {rank}$')
        #fig2.colorbar(c, ax = ax2)
plt.show()
#draw_plot(m_arc, header = 'Maximum archetype recall', color_scheme = 'Blues', apply_over_samples = np.max)
#draw_plot(m_ex, header = 'Example recall', color_scheme = 'YlOrBr')
#draw_plot(m_ex, header = 'Maximum example recall', color_scheme = 'YlOrBr', apply_over_samples = np.max)
#draw_plot(its, header = 'Maximum iterations', color_scheme = 'Reds', vmax = max_it, apply_over_samples=np.max)
#draw_plot(errors, header = 'Maximum final errors', color_scheme = 'Greys', apply_over_samples=np.max)
#draw_plot(errors, header = 'Mean final errors', color_scheme = 'Greys')
sys.exit()
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

if draw_capacity:
    ax.vlines(x=rank / alpha_c, ymin=y_min, ymax=y_max, colors='red')

ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$M$')

#fig.supxlabel(r'$M$')
#fig.supylabel(r'$r$')
cbar_ex = fig.colorbar(im_ex, ax=ax, pad = -0.07)
cbar_arc = fig.colorbar(im_arc, ax=ax)
cbar_arc.set_ticks([])
fig.suptitle(rf'Arc vs Examples for $\alpha M = {rank}$')
plt.show()