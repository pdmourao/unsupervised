import numpy as np
import laboratory as lab
import experiments as exp
import theory

samples = 50
rank = 5
num_points = 50

m_values = np.linspace(1, 50, num = num_points, endpoint = False)
r_values = np.linspace(theory.sep_r(alpha = rank / m_values[-1], m = m_values[-1])-0.01, 0.9, num = num_points, endpoint = False)

kwargs = {'neurons': 1000,
          'rank': rank,
          'max_it': 200,
          'reduced': 'full',
          'diagonal': False,
          'initial': 'new_ex',
          'm_values': m_values,
          'r_values': r_values,
          'p': 1}

experiment = lab.Experiment(directory = 'Data_remote', func = exp.gen_optimal, **kwargs)
experiment.create()
experiment.run_to(samples, predict_folder = 'Predictions_remote')



