import laboratory as lab
import experiments as exp
import numpy as np
from matplotlib import pyplot as plt
import theory
from tqdm import tqdm
from auxiliary import mags_onestep_1d

kwargs = {'neurons': 1000,
          'alpha': 0.05,
          'r': 0.5,
          'initial': 'ex',
          'attractor': 'ex',
          'diagonal': True,
          'p': 1
          }
samples = 5

m_values = np.linspace(50,450, num = 9, dtype = int)
mag_mean, mag_std= mags_onestep_1d(x_arg = 'm', x_values = m_values, samples = samples, **kwargs)
p_std=plt.errorbar(m_values, mag_mean, mag_std, linestyle='None', marker='^', color = 'blue')

ms = np.linspace(1, 500, 100)
kwargs.pop('initial')
kwargs.pop('neurons')
kwargs['attractor'] = 'ex'
mag_values_ex = [theory.mags(**kwargs, m = m) for m in tqdm(ms)]

p1 = plt.plot(ms, mag_values_ex, color = 'blue')

plt.ylim(0,1)
plt.show()