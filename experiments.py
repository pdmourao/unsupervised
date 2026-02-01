import numpy as np
from time import time as time
from laboratory.systems import Dream as dream
from tqdm import tqdm


def delta(neurons, alpha, r, m, initial, attractor, p = 1, diagonal = False, entropy = None):

    t = time()
    if alpha > 0:
        k = int(alpha * neurons)
    else:
        k = 1
    system = dream(neurons = neurons, k = k, r = r, m = m, rng_ss = np.random.SeedSequence(entropy),
                   diagonal = diagonal)
    system.set_interaction()

    print(f'Interaction matrix computed in {time() - t} seconds.')

    t = time()
    i_state = system.gen_samples(system.state(initial), p = p)
    o_state = system.state(attractor)
    deltas = o_state * (i_state @ system.J)
    print(f'Deltas computed in {time() - t} seconds.')

    return deltas


def delta_test(samples, neurons, alpha, r, m, initial, attractor, p = 1, diagonal = False, entropy = None):
    deltas = np.zeros((samples, m, 1, neurons))
    for sample in tqdm(range(samples)):
        t = time()
        rng_ss_list = np.random.SeedSequence(entropy).spawn(2)
        system = dream(neurons = neurons, k = 1, r = r, m = m, rng_ss = rng_ss_list[0],
                   diagonal = diagonal)
        system.set_interaction()
        examples = system.examples
        J = 1 / (neurons * m) * np.einsum('aui, auj -> ij', examples[1:], examples[1:], optimize = True)
        phi = np.random.default_rng(rng_ss_list[1]).choice([-1, 1], p=[(1 - p) / 2, (1 + p) / 2], size = (m, 1, neurons))
        input = examples[0,0]*phi[0,0]
        output = examples[0,0]
        # deltas[sample] = ( 1 / m) * phi + output * ( J @ input )
        deltas[sample] = examples[0,0] * (system.J @ ((examples * phi)[0,0]))

        # print(f'Delta {sample+1}/{samples} computed in {time() - t} seconds.')

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


