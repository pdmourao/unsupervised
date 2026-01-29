import numpy as np
from time import time as time
from laboratory.systems import Dream as dream


def delta(entropy, neurons, alpha, r, m, initial, attractor, p = 1, diagonal = False):

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


    return deltas

# remove auto creation (for when mistake is made in giving inputs)
# create single_checker and remove sanity_checker from these experiments
def mags_onestep(entropy, neurons, alpha, r, m, initial, p, diagonal = False, disable = False):

    t = time()

    system = dream(neurons = neurons, k = int(alpha * neurons), r = r, m = m, rng_ss = np.random.SeedSequence(entropy),
                   diagonal = diagonal)
    system.set_interaction()

    if not disable:
        print(f'Interaction matrix computed in {time() - t} seconds.')

    t = time()
    i_state = system.gen_samples(system.state(initial), p = p)
    o_state_arc = system.state('arc')
    o_state_ex = system.state('ex')
    mags_arc = np.mean(np.sign(o_state_arc * (i_state @ system.J)), axis = -1)
    mags_ex = np.mean(np.sign(o_state_ex * (i_state @ system.J)), axis=-1)

    if not disable:
        print(f'Magnetizations computed in {time() - t} seconds.')

    return mags_arc, mags_ex


