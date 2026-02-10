import experiments as exp
import numpy as np
from matplotlib import pyplot as plt
import theory
from tqdm import tqdm
import laboratory as lab
from auxiliary import mags_onestep_1d

kwargs_gen_hr = {'t': 0,
                 'alpha' : 0.1,
                 'm': 50,
                 'r': 0.6,
                 'p': 0.9,
                 'diagonal': False
                 }

kwargs_of = {'t': 0,
             'alpha' : 0.02,
             'm': 5,
             'r': 0.2,
             'p': 0.5,
             'diagonal': False
             }

kwargs_gen_lr = {'t': 0,
                 'alpha' : 0.02,
                 'm': 5,
                 'r': 0.7,
                 'p': 0.9,
                 'diagonal': False
                 }

kwargs_sg = {'t': 0,
             'alpha' : 0.1,
             'm': 20,
             'r': 0.2,
             'p': 0.9,
             'diagonal': False
             }

kwargs = kwargs_of
neurons = 1000
samples = 100
max_it_mc = 200
reduced = 'partial'
r = kwargs['r']
p = kwargs['p']
kwargs.pop('p')

# experimental spectrum
experiment = lab.Experiment(directory = 'Data', func = exp.spectrum, neurons = neurons, **kwargs)
experiment.create()
experiment.run_to(samples)
spec = experiment.read()
# function for theoretical spectrum
spec_func = theory.spec_dist(**kwargs)

plt.hist(np.ravel(spec), bins = 50, density = True)

xmin, xmax = plt.xlim()
xs = np.linspace(xmin, xmax, num = 100)
# compute theoretical spectrum
ys = [spec_func(x) for x in tqdm(xs)]
plt.plot(xs, ys)
plt.ylim(0,1)
plt.xlabel(r'$\lambda$')
plt.ylabel(r'$\rho(\lambda)$')
plt.title('Spectrum')
plt.show()

p_values = np.linspace(1,0.1, num = 10)
magarc_mean, magarc_std, = mags_onestep_1d(x_arg = 'p', x_values = p_values, samples = samples, neurons = neurons,
                                           initial = 'arc', attractor = 'arc', reduced = reduced, **kwargs)
parc_std=plt.errorbar(p_values, magarc_mean, magarc_std, linestyle='None', marker='^', color = 'blue')

magex_mean, magex_std = mags_onestep_1d(x_arg = 'p', x_values = p_values, samples = samples, neurons = neurons,
                                        initial = 'ex', attractor = 'ex', reduced = reduced, **kwargs)
pex_std=plt.errorbar(p_values, magex_mean, magex_std, linestyle='None', marker='^', color = 'orange')

ps = np.linspace(1, 0, num = 100, endpoint = False)
ms_ex = np.array([theory.mags(p = p, attractor = 'ex', **kwargs) for p in tqdm(ps)])
plt.plot(ps, ms_ex, label = 'examples', color = 'orange')

ms_arc = np.array([theory.mags(p = p, attractor = 'arc', **kwargs) for p in tqdm(ps)])
plt.plot(ps, ms_arc, label = 'archetypes', color = 'blue')
plt.legend()

plt.plot(ps, ps, color = 'grey', linestyle = 'dashed')

plt.gca().invert_xaxis()
plt.ylim(0,1)

plt.xlabel(r'$p$')
plt.ylabel(r'$m_1$')
plt.title('One-step magnetizations')

plt.show()

experiment = lab.Experiment(directory = 'Data', func = exp.attraction_mc, neurons = neurons, initial = 'arc',
                            max_it = max_it_mc, p = r, reduced = reduced, **kwargs)
experiment.create()
experiment.run_to(samples)
mags_arc_mc, mags_ex_mc, its_mc, errors_mc = experiment.read()

plt.hist(np.ravel(mags_arc_mc), bins = 'fd', density = True, color = 'blue')
plt.xlim(0,1)

plt.xlabel(r'$m_\infty$')
plt.ylabel(r'$p(m_\infty)$')
plt.title('Attractiveness of archetypes, starting from a new example')

plt.show()

plt.hist(np.ravel(mags_ex_mc), bins = 'fd', density = True, color = 'orange')
plt.xlim(0,1)

plt.xlabel(r'$m_\infty$')
plt.ylabel(r'$p(m_\infty)$')
plt.title('Attractiveness of examples, starting from a new example')

plt.show()

print(f'Max final error {np.max(errors_mc)} and iteration {np.max(its_mc)} across all first samples.')

experiment = lab.Experiment(directory = 'Data', func = exp.attraction_mc, neurons = neurons, initial = 'arc', p =1,
                            reduced = reduced, max_it = max_it_mc, **kwargs)
experiment.create()
experiment.run_to(samples)
mags_arc_mc, mags_ex_mc, its_mc, errors_mc = experiment.read()

plt.hist(np.ravel(mags_arc_mc), bins = 'fd', density = True, color = 'blue')
plt.xlim(0,1)

plt.xlabel(r'$m_\infty$')
plt.ylabel(r'$p(m_\infty)$')
plt.title('Attractiveness of archetypes, starting from a stored archetype')

plt.show()

plt.hist(np.ravel(mags_ex_mc), bins = 'fd', density = True, color = 'orange')
plt.xlim(0,1)

plt.xlabel(r'$m_\infty$')
plt.ylabel(r'$p(m_\infty)$')
plt.title('Attractiveness of examples, starting from a stored archetype')

plt.show()

print(f'Max final error {np.max(errors_mc)} and iteration {np.max(its_mc)} across all second samples.')

run_last = True
if run_last:
    experiment = lab.Experiment(directory = 'Data', func = exp.attraction_mc, neurons = neurons, initial = 'ex', p =p,
                                reduced = reduced, max_it = max_it_mc, **kwargs)
    experiment.create()
    experiment.run_to(samples)
    mags_arc_mc, mags_ex_mc, its_mc, errors_mc = experiment.read()

    plt.hist(np.ravel(mags_arc_mc), bins = 'fd', density = True, color = 'blue')
    plt.xlim(0,1)

    plt.xlabel(r'$m_\infty$')
    plt.ylabel(r'$p(m_\infty)$')
    plt.title('Attractiveness of archetypes, starting close to a stored example')

    plt.show()

    plt.hist(np.ravel(mags_ex_mc), bins = 'fd', density = True, color = 'orange')
    plt.xlim(0,1.1)

    plt.xlabel(r'$m_\infty$')
    plt.ylabel(r'$p(m_\infty)$')
    plt.title('Attractiveness of examples, starting close to the same example')

    plt.show()

    print(f'Max final error {np.max(errors_mc)} and iteration {np.max(its_mc)} across all third samples.')
