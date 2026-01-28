import numpy as np
from time import time as time
from tqdm import tqdm
from laboratory.systems import Dream as dream
from laboratory.storage import sanity_check
from matplotlib import pyplot as plt

inputs = {'neurons': 1000,
          'alpha': 0.05
          }

def delta(entropy, neurons, alpha, r, m, initial, attractor, p = 1, diagonal = False, checker = None):

    t = time()

    system = dream(neurons = neurons, k = int(alpha * neurons), r = r, m = m, rng_ss = np.random.SeedSequence(entropy),
                   diagonal = diagonal)
    system.set_interaction()

    print(f'Interaction matrix computed in {time() - t} seconds.')

    t = time()
    i_state = system.gen_samples(system.state(initial), p = p)
    o_state = system.state(attractor)
    deltas = o_state * (i_state @ system.J)
    print(f'Deltas computed in {time() - t} seconds.')

    sanity_check(deltas, checker=checker)

    return deltas

# remove auto creation (for when mistake is made in giving inputs)
# create single_checker and remove sanity_checker from these experiments
# finish mags_onestep, returning both attractors
# take it to graphs and data treatment
# create function to read off every datapoint that matches a given set of inputs
def mags_onestep(entropy, neurons, alpha, r, m, initial, attractor, p, diagonal = False, checker = None):

    t = time()

    system = dream(neurons = neurons, k = int(alpha * neurons), r = r, m = m, rng_ss = np.random.SeedSequence(entropy),
                   diagonal = diagonal)
    system.set_interaction()

    print(f'Interaction matrix computed in {time() - t} seconds.')

    t = time()
    i_state = system.gen_samples(system.state(initial), p = p)
    o_state = system.state(attractor)
    deltas = o_state * (i_state @ system.J)
    print(f'Deltas computed in {time() - t} seconds.')

    sanity_check(deltas, checker=checker)

    return deltas