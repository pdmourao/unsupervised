import experiments as exp
import numpy as np
from matplotlib import pyplot as plt
import theory
from tqdm import tqdm
from matplotlib import animation


alpha = 0.2
m = 50
r = 0.5
t_values = [0, 5, 12, 25]

# calculate the maximum of the spectrum
x_max = theory.dist_max(alpha = alpha, r = r, m = m, t = 0)

# spectral distribution
spec_func = theory.spec_dist(alpha = alpha, r = r, m = m, t = 0, diagonal = True)

# Create figure
fig, ax = plt.subplots()
ax.set_ylim(0,10)
# first image
xs = np.linspace(0, x_max, 1000)
line, = ax.plot(xs, [spec_func(x) for x in xs])
text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

total_frames = 100
max_t = 50

# code the updates
def update(frame):
    t = frame * max_t / total_frames
    spec_func = theory.spec_dist(alpha=alpha, r=r, m=m, t=t, diagonal=True)
    y = [spec_func(x) for x in xs]
    line.set_ydata(y)
    text.set_text(rf"$t={t}$")
    return line, text

ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=50)
ani.save("spectrum_animation.mp4", writer="ffmpeg", fps=30)
plt.show()

def plot_spec(ax, t, ylim = None):
# function for theoretical spectrum
    spec_func = theory.spec_dist(alpha = alpha, r = r, m = m, t = t, diagonal = True)

    xmin, xmax = 0, 1
    xs = np.linspace(xmin, xmax, num = 10000)
    # compute theoretical spectrum
    ys = [spec_func(x) for x in tqdm(xs)]

    ax.plot(xs, ys)
    if ylim is not None:
        ax.set_ylim(0, ylim)

    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel(r'$\rho(\lambda)$')
    ax.label_outer()
    ax.set_xlim(xmin,xmax)
    ax.set_title(rf'$t = {t}$')

other_g = False
if other_g:
    fig, axs = plt.subplots(len(t_values),1, sharex=True)

    for idx, ax in enumerate(axs.flat):
        t = t_values[idx]
        plot_spec(ax, t, ylim = 7)

plt.show()

