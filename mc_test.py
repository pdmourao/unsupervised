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
errors = []
mags = []
rng_list = np.random.SeedSequence().spawn(samples)

system = dream(neurons = neurons, k = k, r = r, m = m, rng_ss = rng_list[0],
                   diagonal = diagonal)
system.set_interaction()


p = 0.2
noise = np.random.default_rng(rng_list[1]).choice([-1, 1], p = [(1-p)/2, (1+p)/2], size = neurons)
state = system.patterns[0]
idxs = tuple([np.empty(0) for _ in np.shape(state)])
while len(errors) < max_it:  # do the simulation

        old_state = state
        state = np.sign(state @ system.J)
        mags.append(1/neurons*np.dot(system.examples[0,0], state))
        this_error = np.max((1 - np.mean(state * old_state, axis=-1)) / 2)
        diff = state - old_state
        prev_idxs = idxs
        idxs = np.nonzero(diff)
        errors.append(this_error)
        if this_error <= 0 or all([np.array_equal(x, y) for x, y in zip(idxs, prev_idxs)]):
            break

print(mags)