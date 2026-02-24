import numpy as np
import laboratory as lab
import experiments as exp
from matplotlib import pyplot as plt
from tqdm import tqdm
import theory

alpha = 0.5
m = 10
r = 0.5
p = 0.9
max_it = 200
reduced = 'full'
diagonal = False
initial = 'ex'

def transf(t, x):
    return (1+t)*x/(1+t*x)

x0 = 0.1
x1 = 0.5
ts = np.linspace(0, 10, 1000)
dels = [transf(t, x1) - transf(t, x0) for t in ts]
#plt.plot(ts, dels)
#plt.show()

rank = alpha*m
mmin = 2
mmax = 50
ms = np.linspace(mmin, mmax, 1000)
seps = [theory.sep_r(alpha = alpha, m = m) for m in ms]
print(seps)
plt.plot(ms, seps)
plt.ylim(0, 1)
plt.scatter([m], [r])
plt.show()