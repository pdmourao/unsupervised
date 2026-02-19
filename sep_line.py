import numpy as np
import laboratory as lab
import experiments as exp
from matplotlib import pyplot as plt
from tqdm import tqdm
import theory

neurons = 1000
alpha = 0.03
m = 10
r = 0.5
p = 0.9
max_it = 200
reduced = 'full'
diagonal = False
initial = 'ex'

rank = alpha*m
mmin = 1
mmax = 50
ms = np.linspace(mmin, mmax, 100)
seps = [theory.sep_r(alpha = rank / m, m = m) for m in ms]
plt.plot(ms, seps)
plt.ylim(0, 1)
plt.scatter([m], [r])
plt.show()