import numpy as np
from matplotlib import pyplot as plt
from laboratory.systems import Dream as dream
from tqdm import tqdm
from time import time

n = 1000
k = 100
m = 50
x = np.random.rand(m, k, n)
y = x.reshape((m*k,n))
for u in range(m):
    for a in range(k):
        print(np.array_equal(y[k*u+a], x[u,a]))

t = time()
J1 = np.einsum('aui, auj -> ij', x, x)
print(time() - t)
t = time()
J2 = np.transpose(x.reshape((m*k,n))) @ x.reshape((m*k,n))
print(time() - t)
print(np.allclose(J1,J2))