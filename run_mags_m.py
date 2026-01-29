import laboratory as lab
import experiments as exp
import numpy as np
from matplotlib import pyplot as plt
import theory
from tqdm import tqdm

kwargs = {'neurons': 1000,
          'alpha': 0.05,
          'r': 0.5,
          'initial': 'ex',
          'diagonal': True,
          'p': 1
          }
samples = 5

m_values = np.linspace(50,450, num = 9, dtype = int)
mag_list_arc_mean = []
mag_list_arc_std = []
mag_list_ex_std = []
mag_list_ex_mean = []

for m in m_values:
    print(f'Starting m = {m}.')
    experiment = lab.Experiment(directory = 'Data', func = exp.mags_onestep, **kwargs, m = m)
    experiment.run_to(samples)
    mags_arc, mags_ex = experiment.read()
    mag_list_arc_mean.append(np.mean(mags_arc))
    mag_list_arc_std.append(np.std(mags_arc))
    mag_list_ex_std.append(np.std(mags_ex))
    mag_list_ex_mean.append(np.mean(mags_ex))

ms = np.linspace(1, 500, 500)
kwargs.pop('initial')
kwargs.pop('neurons')
kwargs['attractor'] = 'ex'
mag_values_ex = [theory.mags(**kwargs, m = m) for m in tqdm(ms)]

p_std=plt.errorbar(m_values, mag_list_ex_mean, mag_list_ex_std, linestyle='None', marker='^', color = 'blue')
p1 = plt.plot(ms, mag_values_ex, color = 'blue')

plt.ylim(0,1)
plt.show()