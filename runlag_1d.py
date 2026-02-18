import numpy as np
import laboratory as lab
import experiments as exp

neurons = 1000
alpha = 0.2
m = 50
r = 0.5
p = 0.9
samples = 10
max_it = 200
reduced = 'full'
diagonal = False
initial = 'ex'

t_values = np.linspace(0, 50, num = 101)

experiment = lab.Experiment(directory = 'Data_remote', func = exp.gen_t, m = m, r = r, t_values = t_values,
                            neurons = neurons, alpha = alpha, p = p, reduced = reduced, diagonal = diagonal,
                            initial = initial, max_it = max_it)
experiment.create()
experiment.run_to(samples)