import numpy as np
import laboratory as lab
import experiments as exp
from matplotlib import pyplot as plt
from tqdm import tqdm
import theory

neurons = 1000
rank = 2
t = 0
p = 0.9
samples = 50
max_it = 200
reduced = 'full'
diagonal = False
initial = 'ex'
alpha_c = 0.138

draw_capacity = rank > alpha_c

r_values = np.linspace(1, 0, num = 50, endpoint = False)[::-1]
m_values = np.linspace(1, 50, num = 50, dtype = int)

experiment = lab.Experiment(directory = 'Data', func = exp.gen_mr, m_values = m_values, r_values = r_values,
                            neurons = neurons, rank = rank, t = t, p = p, reduced = reduced, diagonal = diagonal,
                            initial = initial, max_it = max_it)
experiment.create()
experiment.run_to(samples)
m_arc, m_ex, its, errors = experiment.read()

print(f'Maximum recorded iterations and errors were {np.max(its)} and {np.max(errors)}, respectively')

x_min, x_max = m_values[0].astype(float).item(), m_values[-1].astype(float).item()
y_min, y_max = r_values[0].item(), r_values[-1].item()

step_x = (m_values[1] - m_values[0])/2
step_y = (r_values[1] - r_values[0])/2

x_min -= step_x
x_max += step_x
y_min -= step_y
y_max += step_y

fig, ax = plt.subplots(1)
c = ax.imshow(np.transpose(np.flip(np.mean(m_arc, axis = 0), axis = -1)), cmap='Blues', vmin=0, vmax=1, aspect='auto', interpolation='nearest',
              extent = (x_min, x_max, y_min, y_max))
fig.colorbar(c)

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

ms = np.linspace(1, 50, 1000)
seps = [theory.sep_r(alpha = rank / m, m = m) for m in tqdm(ms)]
ax.plot(ms, seps, color = 'black', linestyle = 'dashed')

if draw_capacity:
    ax.vlines(x = rank / alpha_c, ymin = y_min, ymax = y_max, colors = 'red')

ax.set_xlabel(r'$M$')
ax.set_ylabel(r'$r$')

ax.set_title(rf'Archetype recall for $\alpha M = {rank}$')
plt.show()


fig, ax = plt.subplots(1)
c = ax.imshow(np.transpose(np.flip(np.mean(m_ex, axis = 0), axis = -1)), cmap='YlOrBr', vmin=0, vmax=1, aspect='auto', interpolation='nearest',
              extent = (x_min, x_max, y_min, y_max))
fig.colorbar(c)

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

ax.plot(ms, seps, color = 'black', linestyle = 'dashed')

if draw_capacity:
    ax.vlines(x = rank / alpha_c, ymin = y_min, ymax = y_max, colors = 'red')

ax.set_xlabel(r'$M$')
ax.set_ylabel(r'$r$')

ax.set_title(rf'Example recall for $\alpha M = {rank}$')
plt.show()