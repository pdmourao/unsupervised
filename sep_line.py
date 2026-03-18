import numpy as np
import laboratory as lab
import experiments as exp
from matplotlib import pyplot as plt
from tqdm import tqdm
import theory

rank = 2

mmin = 2
mmin_values = 5
mmax = 54
ms = np.linspace(mmin, mmax, 1000)

r_gap = 0.05
m_values = np.arange(mmin_values, mmax, num = 50, dtype = int)
print(m_values)


seps = [theory.sep_r(alpha = rank/m, m = m) for m in ms]
points = [theory.sep_r(alpha = rank/m, m = m) + r_gap for m in m_values]
print(points[0])
plt.plot(ms, seps)
plt.vlines(x=rank / 0.138, ymin=0, ymax=1, colors='red')
plt.xlabel(r'$M$')
plt.ylabel(r'$r$')
plt.title(rf'$\alpha M = {rank}$')
plt.ylim(0, 1)
plt.scatter(m_values, points)
plt.show()