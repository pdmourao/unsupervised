import numpy as np
import laboratory as lab
import experiments as exp
from matplotlib import pyplot as plt
from tqdm import tqdm
import theory

alpha = 0.1
m = 20
r = 0.45

rank = alpha*m
mmin = 2
mmax = 50
ms = np.linspace(mmin, mmax, 1000)
seps = [theory.sep_r(alpha = rank/m, m = m) for m in ms]
print(seps)
plt.plot(ms, seps)
plt.vlines(x=rank / 0.138, ymin=0, ymax=1, colors='red')
plt.ylim(0, 1)
plt.scatter([m], [r])
plt.show()