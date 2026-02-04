import numpy as np
from matplotlib import pyplot as plt
from laboratory.systems import Dream as dream
from tqdm import tqdm

neurons = 1000
alpha = 0.01
m = 5
r = 0.2
diagonal = False
samples = 10
max_it = 200
reduced = True
initial = 'arc'
p = 1

k = int(alpha * neurons)
mags_arc = np.empty((samples, k), dtype = np.float64)
mags_ex = np.empty((samples, k), dtype = np.float64)
errors = np.empty(samples, dtype = np.float64)

rng_list = np.random.SeedSequence().spawn(samples)
for sample in tqdm(range(samples)):
    system = dream(neurons = neurons, k = k, r = r, m = m, rng_ss = rng_list[sample],
                   diagonal = diagonal)
    system.set_interaction()

    system.initial_state = system.gen_samples(system.state(initial, reduced = True), p=p)
    final_state, these_errors = system.simulate_zero_T(max_it = max_it)

    o_state_arc = system.state('arc')
    o_state_ex = system.state('ex', reduced = True)
    mags_arc[sample] = np.mean(o_state_arc * final_state, axis = -1)
    mags_ex[sample] = np.mean(o_state_ex * final_state, axis=-1)
    errors[sample] = these_errors[-1]

plt.hist(np.ravel(mags_arc), bins = 'fd', density = True, color = 'blue')
plt.xlim(0,1)
plt.show()

plt.hist(np.ravel(mags_ex), bins = 'fd', density = True, color = 'orange')
plt.xlim(0,1)
plt.show()

print(f'Max final error across all second samples was {np.max(errors)}')