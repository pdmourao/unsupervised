import experiments as exp
import numpy as np
from matplotlib import pyplot as plt
import theory
from tqdm import tqdm
from auxiliary import mags_onestep_1d

kwargs_5 = {'alpha' : 0.02,
            'm': 5,
            'r': 0.4,
            'reduced': 'partial',
            'diagonal': False
            }

kwargs = kwargs_5
neurons = 1000
samples = 10

p_values = np.linspace(1,0.1, num = 10)
magarc_mean, magarc_std, = mags_onestep_1d(x_arg = 'p', x_values = p_values, samples = samples, neurons = neurons,
                                           initial = 'arc', attractor = 'arc', **kwargs)
parc_std=plt.errorbar(p_values, magarc_mean, magarc_std, linestyle='None', marker='^', color = 'blue')

magex_mean, magex_std = mags_onestep_1d(x_arg = 'p', x_values = p_values, samples = samples, neurons = neurons,
                                        initial = 'ex', attractor = 'ex', **kwargs)
pex_std=plt.errorbar(p_values, magex_mean, magex_std, linestyle='None', marker='^', color = 'orange')

ps = np.linspace(1, 0, num = 100, endpoint = False)
ms_ex = np.array([theory.mags(p = p, attractor = 'ex', **kwargs) for p in tqdm(ps)])
plt.plot(ps, ms_ex, label = 'examples', color = 'orange')

ms_arc = np.array([theory.mags(p = p, attractor = 'arc', **kwargs) for p in tqdm(ps)])
plt.plot(ps, ms_arc, label = 'archetypes', color = 'blue')
plt.legend()

plt.plot(ps, ps, color = 'grey', linestyle = 'dashed')

plt.gca().invert_xaxis()
plt.ylim(0,1)

plt.xlabel(r'$p$')
plt.ylabel(r'$m_1$')
plt.title('One-step magnetizations')

plt.show()

