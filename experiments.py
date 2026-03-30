import numpy as np
from time import time as time
from laboratory.systems import Dream as dream
from tqdm import tqdm
import theory


def delta(neurons, alpha, r, m, initial, attractor, t, p, diagonal, entropy = None):

    t0 = time()
    if alpha > 0:
        k = int(alpha * neurons)
    else:
        k = 1
    system = dream(neurons = neurons, k = k, r = r, m = m, rng_ss = np.random.SeedSequence(entropy),
                   diagonal = diagonal)
    system.set_interaction()

    print(f'Interaction matrix computed in {time() - t0} seconds.')

    t0 = time()
    i_state = system.gen_samples(system.state(initial), p = p)
    o_state = system.state(attractor)
    deltas = o_state * (i_state @ system.J)
    print(f'Deltas computed in {time() - t0} seconds.')

    return deltas


def delta_test(samples, neurons, alpha, r, m, initial, attractor, p = 1, diagonal = False, entropy = None):
    deltas = []
    for sample in tqdm(range(samples)):

        rng_ss_list = np.random.SeedSequence(entropy).spawn(2)
        examples = np.random.default_rng(rng_ss_list[0]).choice([-1, 1], p=[(1 - r) / 2, (1 + r) / 2], size = (m, neurons))
        J = 1 / (neurons * m) * np.einsum('ai, aj -> ij', examples[1:], examples[1:], optimize = True)
        np.fill_diagonal(J,0)
        Jt = 1 / (neurons * m) * np.einsum('i, j -> ij', examples[0], examples[0], optimize=True)
        np.fill_diagonal(Jt,0)
        phi = np.random.default_rng(rng_ss_list[1]).choice([-1, 1], p=[(1 - p) / 2, (1 + p) / 2], size = (m, neurons))
        input = examples[0]*phi[0]
        output = examples[0,0]
        # deltas[sample] = ( 1 / m) * phi + output * ( J @ input )
        deltas.append(examples[0] * (J @ ((examples * phi)[0]))+examples[0] * (Jt @ ((examples * phi)[0])))
        # deltas.append(np.mean(input, axis = -1))

        # print(f'Delta {sample+1}/{samples} computed in {time() - t} seconds.')

    return np.array(deltas)


# create single_checker and remove sanity_checker from these experiments
def mags_onestep(entropy, neurons, alpha, r, m, t, initial, attractor, p, diagonal, reduced, disable = False):

    t0 = time()

    system = dream(neurons = neurons, k = int(alpha * neurons), r = r, m = m, rng_ss = np.random.SeedSequence(entropy),
                   diagonal = diagonal)
    system.set_interaction()

    if not disable:
        print(f'Interaction matrix computed in {time() - t0} seconds.')

    t0 = time()
    i_state = system.gen_samples(system.state(initial, reduced = reduced), p = p)
    o_state = system.state(attractor, reduced = reduced)
    mags = np.mean(np.sign(o_state * (i_state @ system.J)), axis=-1)

    if not disable:
        print(f'Magnetizations computed in {time() - t0} seconds.')

    return mags


def spec_nosave(samples, neurons, alpha, r, m, t, entropy = None, diagonal = False):

    t0 = time()
    spec = np.empty((samples, neurons), dtype = float)

    rng_ss_list = np.random.SeedSequence(entropy).spawn(samples)

    for sample in tqdm(range(samples)):
        system = dream(neurons=neurons, k=int(alpha * neurons), r=r, m=m, rng_ss = rng_ss_list[sample], t = t,
                       diagonal=diagonal)
        system.set_interaction()

        spec[sample] = np.real_if_close(np.linalg.eigvals(system.J), tol=1e-3)

    print(f'Eigenvalues computed in {time() - t0} seconds.')
    return spec


def spectrum(entropy, neurons, alpha, r, m, t, diagonal):

    system = dream(neurons=neurons, k=int(alpha * neurons), r=r, m=m, t=t, rng_ss = np.random.SeedSequence(entropy),
                       diagonal=diagonal)
    system.set_interaction()

    return np.real_if_close(np.linalg.eigvals(system.J), tol=1e-3)


def attraction_mc_old(entropy, neurons, alpha, r, m, t, initial, max_it, diagonal, disable = False):

    t0 = time()

    system = dream(neurons = neurons, k = int(alpha * neurons), r = r, m = m, rng_ss = np.random.SeedSequence(entropy),
                   diagonal = diagonal)
    system.set_interaction()

    if not disable:
        print(f'Interaction matrix computed in {round(time() - t0,3)} seconds.')
    t0 = time()
    system.initial_state = system.gen_samples(system.state(initial), p=r)
    final_state, errors = system.simulate_zero_T(max_it = max_it)
    if not disable:
        print(f'System ran in {round(time() - t0,3)} seconds to {len(errors)} iteration(s).')
        print(f'Final max error was {errors[-1]}')
    t0 = time()
    o_state_arc = system.state('arc')
    o_state_ex = system.state('ex')
    mags_arc = np.mean(o_state_arc * final_state, axis = -1)
    mags_ex = np.mean(o_state_ex * final_state, axis=-1)

    if not disable:
        print(f'Magnetizations computed in {round(time() - t0,3)} seconds.')

    return mags_arc, mags_ex, errors[-1]


def attraction_mc(entropy, neurons, alpha, r, m, t, p, initial, max_it, diagonal, reduced, disable = False):

    t0 = time()

    system = dream(neurons = neurons, k = int(alpha * neurons), r = r, m = m, t = t,
                   rng_ss = np.random.SeedSequence(entropy), diagonal = diagonal)
    system.set_interaction()

    if not disable:
        print(f'Interaction matrix computed in {round(time() - t0,3)} seconds.')
    t0 = time()
    system.initial_state = system.gen_samples(system.state(initial, reduced = reduced), p=p)
    final_state, errors = system.simulate_zero_T(max_it = max_it)
    if not disable:
        print(f'System ran in {round(time() - t0,3)} seconds to {len(errors)} iteration(s).')
        print(f'Final max error was {errors[-1]}')
    t0 = time()
    o_state_arc = system.state('arc', reduced = reduced)
    o_state_ex = system.state('ex', reduced = reduced)
    mags_arc = np.mean(o_state_arc * final_state, axis = -1)
    mags_ex = np.mean(o_state_ex * final_state, axis=-1)

    if not disable:
        print(f'Magnetizations computed in {round(time() - t0,3)} seconds.')

    return mags_arc, mags_ex, len(errors), errors[-1]

def gen_mr(entropy, neurons, rank, t, m_values, r_values, p, initial, max_it, diagonal, reduced):

    len_x = len(m_values)
    len_y = len(r_values)

    rng_list = np.random.SeedSequence(entropy).spawn(len_x * len_y)
    mags_arc = np.empty((len_x, len_y), dtype = float)
    mags_ex = np.empty((len_x, len_y), dtype=float)
    its = np.empty((len_x, len_y), dtype=int)
    errors = np.empty((len_x, len_y), dtype=float)

    with tqdm(total = len_x * len_y) as pbar:
        for idx_m, m in enumerate(m_values):
            for idx_r, r in enumerate(r_values):
                # we take the rng for this iteration and provide it to the system with the rest of the inputs
                this_ss = rng_list[idx_m * len(r_values) + idx_r]
                system = dream(neurons=neurons, k=int(rank * neurons / m), r=r, m=m, t=t,
                               rng_ss=this_ss, diagonal=diagonal)
                # generate the iteraction matrix
                system.set_interaction()
                # give the option of starting from a new_example
                if initial == 'new_ex':
                    system.initial_state = system.gen_samples(system.gen_samples(system.state('arc', reduced=reduced), p=r), p=p)
                # otherwise it is either archetypes or stored examples
                else:
                    system.initial_state = system.gen_samples(system.state(initial, reduced=reduced), p=p)
                # here we simulate
                final_state, error_list = system.simulate_zero_T(max_it=max_it)
                # measuring magnetizations
                mags_arc[idx_m, idx_r] = np.mean(system.state('arc', reduced=reduced) * final_state, axis=-1)
                mags_ex[idx_m, idx_r] = np.mean(system.state('ex', reduced=reduced) * final_state, axis=-1)

                its[idx_m, idx_r] = len(error_list)
                errors[idx_m, idx_r] = error_list[-1]

                pbar.update(1)

    return mags_arc, mags_ex, its, errors

def gen_tm_transition(entropy, neurons, rank, t_values, m_values, r_buffer, p, initial, max_it, diagonal, reduced):

    len_x = len(t_values)
    len_y = len(m_values)

    # get the points from just above the transition line
    r_values = [theory.sep_r(alpha=rank / m, m=m) + r_buffer for m in m_values]

    rng_list = np.random.SeedSequence(entropy).spawn(len_x * len_y)
    mags_arc = np.empty((len_x, len_y), dtype = float)
    mags_ex = np.empty((len_x, len_y), dtype=float)
    its = np.empty((len_x, len_y), dtype=int)
    errors = np.empty((len_x, len_y), dtype=float)

    with tqdm(total = len_x * len_y) as pbar:
        for idx_t, t in enumerate(t_values):
            for idx_m, m in enumerate(m_values):
                r = r_values[idx_m]

                # we take the rng for this iteration and provide it to the system with the rest of the inputs
                this_ss = rng_list[idx_t * len(m_values) + idx_m]
                system = dream(neurons=neurons, k=int(rank * neurons / m), r=r, m=m, t=t,
                               rng_ss=this_ss, diagonal=diagonal)
                # generate the iteraction matrix
                system.set_interaction()
                # give the option of starting from a new_example
                if initial == 'new_ex':
                    system.initial_state = system.gen_samples(system.gen_samples(system.state('arc', reduced=reduced), p=r), p=p)
                # otherwise it is either archetypes or stored examples
                else:
                    system.initial_state = system.gen_samples(system.state(initial, reduced=reduced), p=p)
                # here we simulate
                final_state, error_list = system.simulate_zero_T(max_it=max_it)
                # measuring magnetizations
                mags_arc[idx_t, idx_m] = np.mean(system.state('arc', reduced=reduced) * final_state, axis=-1)
                mags_ex[idx_t, idx_m] = np.mean(system.state('ex', reduced=reduced) * final_state, axis=-1)

                its[idx_t, idx_m] = len(error_list)
                errors[idx_t, idx_m] = error_list[-1]

                pbar.update(1)

    return mags_arc, mags_ex, its, errors


def gen_t(entropy, neurons, alpha, t_values, m, r, p, initial, max_it, diagonal, reduced):

    len_x = len(t_values)

    rng_list = np.random.SeedSequence(entropy).spawn(len_x)
    mags_arc = np.empty(len_x, dtype = float)
    mags_ex = np.empty(len_x, dtype=float)
    its = np.empty(len_x, dtype=int)
    errors = np.empty(len_x, dtype=float)

    for idx_t, t in enumerate(tqdm(t_values)):
        this_ss = rng_list[idx_t]
        system = dream(neurons=neurons, k=int(alpha * neurons), r=r, m=m, t=t,
                       rng_ss=this_ss, diagonal=diagonal)
        system.set_interaction()

        # give the option of starting from a new_example
        if initial == 'new_ex':
            system.initial_state = system.gen_samples(system.gen_samples(system.state('arc', reduced=reduced), p=r),
                                                      p=p)
        # otherwise it is either archetypes or stored examples
        else:
            system.initial_state = system.gen_samples(system.state(initial, reduced=reduced), p=p)
        # here we simulate
        final_state, error_list = system.simulate_zero_T(max_it=max_it)

        mags_arc[idx_t] = np.mean(system.state('arc', reduced=reduced) * final_state, axis=-1)
        mags_ex[idx_t] = np.mean(system.state('ex', reduced=reduced) * final_state, axis=-1)

        its[idx_t] = len(error_list)
        errors[idx_t] = error_list[-1]


    return mags_arc, mags_ex, its, errors


def gen_p(entropy, neurons, alpha, t, m, r, p_values, max_it, diagonal, reduced):

    len_x = len(p_values)

    rng_list = np.random.SeedSequence(entropy).spawn(len_x)
    mags_arc = np.empty(len_x, dtype = float)
    mags_ex = np.empty(len_x, dtype=float)
    its = np.empty(len_x, dtype=int)
    errors = np.empty(len_x, dtype=float)

    for idx_p, p in enumerate(tqdm(p_values)):
        this_ss = rng_list[idx_p]
        system = dream(neurons=neurons, k=int(alpha * neurons), r=r, m=m, t=t,
                       rng_ss=this_ss, diagonal=diagonal)
        system.set_interaction()

        # initial state
        system.initial_state = system.gen_samples(system.state('arc', reduced=reduced), p=p)
        # here we simulate
        final_state, error_list = system.simulate_zero_T(max_it=max_it)

        mags_arc[idx_p] = np.mean(system.state('arc', reduced=reduced) * final_state, axis=-1)
        mags_ex[idx_p] = np.mean(system.state('ex', reduced=reduced) * final_state, axis=-1)

        its[idx_p] = len(error_list)
        errors[idx_p] = error_list[-1]


    return mags_arc, mags_ex, its, errors


def gen_tm(entropy, neurons, rank, t_values, m_values, r, p, initial, max_it, diagonal, reduced):

    len_x = len(t_values)
    len_y = len(m_values)

    while theory.sep_r(alpha=rank / m_values[0], m=m_values[0]) > r:
        m_values += 1

    print(f'Determined that M should start at {m_values[0]}.')

    rng_list = np.random.SeedSequence(entropy).spawn(len_x * len_y)
    mags_arc = np.empty((len_x, len_y), dtype = float)
    mags_ex = np.empty((len_x, len_y), dtype=float)
    its = np.empty((len_x, len_y), dtype=int)
    errors = np.empty((len_x, len_y), dtype=float)

    with tqdm(total = len_x * len_y) as pbar:
        for idx_t, t in enumerate(t_values):
            for idx_m, m in enumerate(m_values):

                # we take the rng for this iteration and provide it to the system with the rest of the inputs
                this_ss = rng_list[idx_t * len(m_values) + idx_m]
                system = dream(neurons=neurons, k=int(rank * neurons / m), r=r, m=m, t=t,
                               rng_ss=this_ss, diagonal=diagonal)
                # generate the iteraction matrix
                system.set_interaction()
                # give the option of starting from a new_example
                if initial == 'new_ex':
                    system.initial_state = system.gen_samples(system.gen_samples(system.state('arc', reduced=reduced), p=r), p=p)
                # otherwise it is either archetypes or stored examples
                else:
                    system.initial_state = system.gen_samples(system.state(initial, reduced=reduced), p=p)
                # here we simulate
                final_state, error_list = system.simulate_zero_T(max_it=max_it)
                # measuring magnetizations
                mags_arc[idx_t, idx_m] = np.mean(system.state('arc', reduced=reduced) * final_state, axis=-1)
                mags_ex[idx_t, idx_m] = np.mean(system.state('ex', reduced=reduced) * final_state, axis=-1)

                its[idx_t, idx_m] = len(error_list)
                errors[idx_t, idx_m] = error_list[-1]

                pbar.update(1)

    return mags_arc, mags_ex, its, errors