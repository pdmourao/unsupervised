import numpy as np
import laboratory as lab
import experiments as exp
import theory

neurons = 1000
alpha = 0.1
m = 50
r = 0.6
diagonal = True

t_values = np.linspace(0, 50, num = 101)

left_max = lab.core.prediction(directory = 'Data_remote', func = theory.peak_left_max_t, alpha = alpha, r = r, m = m,
                              t_values = t_values, tol = 1e-4)[0]
left_cm = lab.core.prediction(directory = 'Data_remote', func = theory.peak_left_cm_t, alpha = alpha, r = r, m = m,
                              t_values = t_values, tol = 1e-4)

t_diff = left_max - left_cm
t_cross = None
for idx_t in range(len(t_values) - 1):
    if t_diff[idx_t] * t_diff[idx_t+1] < 0:
        t_cross = (t_values[idx_t] + t_values[idx_t+1])/2
        break
xs = np.linspace(0, 1, 1001)

experiment = lab.Experiment(directory = 'Data_remote', func = exp.spectrum, neurons = neurons, alpha = alpha, r = r, m = m,
                            t = t_cross, diagonal = True)
experiment.create()
experiment.run_to(10)