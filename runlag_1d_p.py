import numpy as np
import laboratory as lab
import experiments as exp

neurons = 1000
alpha = 0.1
m = 20
r = 0.45
p_values = np.linspace(0, 1, num = 101)
samples = 10
max_it = 200
reduced = 'full'
diagonal = False
t = 0

t_values = np.linspace(0, 50, num = 101)

experiment = lab.Experiment(directory = 'Data_remote', func = exp.gen_p, m = m, r = r, t = t, neurons = neurons,
                            alpha = alpha, p_values = p_values, reduced = reduced, diagonal = diagonal, max_it = max_it)
experiment.create()
experiment.run_to(samples)