import laboratory as lab
import experiments as exp
import numpy as np
from matplotlib import pyplot as plt

kwargs = {'neurons': 1000,
          'alpha': 0.05,
          'r': 0.5,
          'm': 50,
          'initial': 'ex',
          'attractor': 'ex',
          'p': 1,
          'diagonal': False
          }

experiment = lab.Experiment(directory = 'Data', func = exp.delta, **kwargs)
experiment.run_to(5)

deltas_nd = experiment.read()
plt.hist(np.ravel(deltas_nd[:,0,0]), bins=50, density=True)
plt.show()