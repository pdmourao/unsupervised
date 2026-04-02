import numpy as np
import laboratory as lab
import experiments as exp

samples = 10
rank = 1.1

if rank == 5:
    M_in = 11
else:
    M_in = 5

t_values = np.linspace(0, 49, num = 50, endpoint = True)
m_values = np.linspace(M_in, M_in + 49, num = 50, dtype = int)

kwargs = {'neurons': 1000,
          'rank': rank,
          'max_it': 200,
          'reduced': 'full',
          'diagonal': False,
          'initial': 'new_ex',
          't_values': t_values,
          'm_values': m_values,
          'r_buffer': 0.05,
          'p': 1}


experiment = lab.Experiment(directory = 'Data_remote', func = exp.gen_tm_transition, **kwargs)

experiment.create()
experiment.run_to(samples)



