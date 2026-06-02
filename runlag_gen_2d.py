import numpy as np
import laboratory as lab
import experiments as exp

samples = 10

r_values = np.linspace(1, 0, num = 50, endpoint = False)[::-1]
m_values = np.linspace(1, 50, num = 50, dtype = int)

kwargs = {'neurons': 1000,
          'rank': 0.1,
          'max_it': 200,
          'reduced': 'full',
          'diagonal': False,
          'initial': 'ex',
          'm_values': m_values,
          'r_values': r_values,
          'p': 0.9}

t_values = [1, 5, 10, 50, 100]

for t in t_values:
    experiment = lab.Experiment(directory = 'Data_remote', func = exp.gen_mr, t = t, **kwargs)

    #experiment.create()
    experiment.run_to(samples)



