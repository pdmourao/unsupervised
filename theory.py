import numpy as np
import math
import scipy
from tqdm import tqdm
from time import time

# Gives the discriminant instead
def spec_disc(alpha, r, m, t):
    mu1 = (1 - r ** 2) / m
    mu2 = r ** 2 + (1 - r ** 2) / m

    def disc(x):
        x = x / (1 + t - t * x)
        a = x * mu1 * mu2
        b = (m * alpha - 1) * mu1 * mu2 - x * (mu1 + mu2)
        c = (1 - alpha * (m - 1)) * mu1 + (1 - alpha) * mu2 + x
        d = - 1

        p = (3 * a * c - b ** 2) / (3 * a ** 2)
        q = (2 * b ** 3 - 9 * a * b * c + 27 * a ** 2 * d) / (27 * a ** 3)

        return (q / 2) ** 2 + (p / 3) ** 3

    return disc

# gives the spectral distribution, predicted with Edwards-Jones
def spec_dist(alpha, r, m, t, diagonal = True):

    if np.isinf(m):
        lp = r ** 2 * (1 + np.sqrt(alpha)) ** 2 + alpha * (1 - r ** 2)
        lm = r ** 2 * (1 - np.sqrt(alpha)) ** 2 + alpha * (1 - r ** 2)
        def dist(x):
            x = x / (1 + t - t * x)
            if lm < x < lp:
                return (1 + t * x) ** 2/ (1 + t) * np.sqrt((lp - x) * (x - lm)) / (2 * math.pi * r ** 2 * (x - alpha * (1 - r ** 2)))
            else:
                return 0

    elif r == 1:
        lp = (1 + np.sqrt(alpha)) ** 2
        lm = (1 - np.sqrt(alpha)) ** 2
        def dist(x):
            x = x / (1 + t - t * x)
            if lm < x < lp:
                return (1 + t * x) ** 2/ (1 + t) * np.sqrt((lp - x) * (x - lm)) / (2 * math.pi * x)
            else:
                return 0

    else:

        mu1 = (1 - r ** 2) / m
        mu2 = r ** 2 + (1 - r ** 2) / m

        def dist(x):
            x = x / (1 + t - t * x)
            a = x * mu1 * mu2
            b = (m * alpha - 1) * mu1 * mu2 - x * (mu1 + mu2)
            c = (1 - alpha * (m - 1)) * mu1 + (1 - alpha) * mu2 + x
            d = - 1

            p = (3 * a * c - b ** 2) / (3 * a ** 2)
            q = (2 * b ** 3 - 9 * a * b * c + 27 * a ** 2 * d) / (27 * a ** 3)

            disc = (q / 2) ** 2 + (p / 3) ** 3

            if disc < 0:
                return 0
            else:
                return (1 + t * x) ** 2/ (1 + t) * np.sqrt(3) / math.pi / 2 * (np.cbrt(-q / 2 + np.sqrt(disc)) + np.cbrt(q / 2 + np.sqrt(disc)))

    if diagonal:
        return dist
    elif t == 0:
        return lambda x: dist(x + alpha)
    else:
        shift = scipy.integrate.quad(lambda x: x * dist(x), 0, np.inf)[0]
        return lambda x: dist(x + shift)

# gives the delta peak
def peak(alpha, m, r, diagonal = True):
    if diagonal:
        shift=0
    else:
        shift = -alpha
    if np.isinf(m):
        pos = alpha * (1 - r ** 2) -shift
        height = max(1 - alpha,0)
    elif r == 1:
        pos = -shift
        height = max(1 - alpha,0)
    else:
        pos = -shift
        height = max(1 - alpha * m, 0)
    return pos, height


# the spec_integrator is used to integrate a function (integrand) over a spectrum (measure) plus a delta_peak
# the delta peak is given as a tuple (position, height)
def spec_integrator(integrand, measure, min_x=-np.inf, max_x=np.inf, peak_coordinates=(0,0)):
    # new_min, new_max = domain(lambda x: measure(x, **other_kwargs), min_x, max_x)
    pos, height = peak_coordinates
    return integrand(pos) * height + scipy.integrate.quad(lambda x: integrand(x) * measure(x), min_x, max_x)[0]

# gives the moments of the J distribution
def J_moments(n, measure, peak_coordinates = (0,0)):
    def integrand(x):
        return x ** n
    return spec_integrator(integrand, measure=measure, peak_coordinates=peak_coordinates)

# this function is called double_peak but it can be used for any number of gaussian peaks
# probs are the probabilities of each peak, avs their averages and stds their standard deviations
def double_peak(x, probs, avs, stds):
    assert len(probs) == len(avs) and len(probs) == len(stds), 'Input arrays for double_peak do not have the same lengths'
    peaks=np.array([p*scipy.stats.norm.pdf(x, av, std) for p, av, std in zip(probs, avs, stds)])
    return np.sum(peaks)

# magnetizations given their peak distributions as a sum of error functions
def double_peak_mags(probs, avs, stds):
    assert len(probs) == len(avs) and len(probs) == len(
        stds), 'Input arrays for double_peak_mags do not have the same lengths'
    peaks=np.array([p*scipy.special.erf(av/(np.sqrt(2)*std)) for p, av, std in zip(probs, avs, stds)])
    return np.sum(peaks)

# peak_args turns associative memories inputs into the corresponding set of peaks
def peak_args(alpha, r, m, t = 0, p=1, attractor = 'arc', diagonal = False):
    assert t == 0, '1-step magnetizations not determined for t > 0.'
    if diagonal:
        shift=alpha
    else:
        shift=0

    if alpha > 0:
        mu2 = J_moments(2, measure = spec_dist(alpha, r, m, t), peak_coordinates = peak(alpha, r, m))
        # print(mu2-alpha**2)
    else:
        mu2 = 0

    if attractor == 'arc':
        std = np.sqrt(mu2 - alpha ** 2 + r ** 2 * p ** 2 * (1 + r) * (1 - r) * (m - 1) / m ** 2)
        av1 = p *  r ** 2 + shift
        av2 = p * r ** 2 - shift
        return [(1+p)/2, (1-p)/2], [av1,av2], [std,std]
    elif attractor == 'ex':
        # print(r ** 4 * p ** 2 * (m - 1) / m ** 2)
        std = np.sqrt(mu2 - alpha ** 2 + r ** 4 * p ** 2 * (1 + r) * (1 - r) * (m - 1) / m ** 2)
        av1 = p / m + shift + r ** 3 * p
        av2 = p / m - shift + r ** 3 * p
        av3 = p / m + shift - r ** 3 * p
        av4 = p / m - shift - r ** 3 * p
        return [(1+r)*(1+p)/4, (1+r)*(1-p)/4, (1-r)*(1+p)/4, (1-r)*(1-p)/4], [av1,av2,av3,av4], [std,std,std,std]
    else:
        raise TypeError("Attractor invalid")

def delta_dist(*args, **kwargs):
    return lambda x: double_peak(x, *peak_args(*args, **kwargs))

def mags(*args, **kwargs):
    return double_peak_mags(*peak_args(*args, **kwargs))

def mags_int(*args, **kwargs):
    dist = delta_dist(*args, **kwargs)
    cumulative = scipy.integrate.quad(dist, -np.inf, 0)[0]
    return 1 - 2 * cumulative

# bissection method
def findroot(f, x1, x2, tol = 1e-3, limit = None):
    if f(x1) * f(x2) > 0:
        return limit
    x = x1 + (x2 - x1)/2
    while abs(x2-x1) > tol:
        if f(x) * f(x1) > 0:
            x1 = x
        else:
            x2 = x
        x = x1 + (x2 - x1) / 2
    return x

def gen(arg, x1, x2, tol = 1e-3, **kwargs):
    def f(x):
        return mags(attractor = 'arc', red = True, **{arg: x}, **kwargs) - mags(attractor = 'ex', red = True, **{arg: x}, **kwargs)
    return findroot(f, x1, x2, tol = tol)

def sep_alpha(r, m):
    mu1 = (1 - r ** 2)/m
    mu2 = r ** 2 + (1 - r ** 2) / m
    p1 = (m-1)/m
    p2 = 1/m
    return (mu2-mu1)**2/(m*(np.cbrt(p1*mu1**2)+np.cbrt(p2*mu2**2))**3)

def sep_r(alpha, m, tol = 1e-4, alpha_c = 0):
    return findroot(lambda r: sep_alpha(r, m) - alpha + alpha_c, 0, 1, tol = tol, limit = 1)

def dist_max(alpha, r, m, t, tol = 1e-2, x_max = 100, diagonal = True, prints = True):
    f = spec_dist(alpha = alpha, r = r, m = m, t = t, diagonal = diagonal)
    for x in np.arange(1.+tol, x_max, tol)[::-1]:
        if f(x) == 0:
            x_max = x
        else:
            break
    if prints:
        print(f'x_max determined to be {x_max}')
    return x_max

def dist_roots(alpha, r, m, t, tol, x_max = None):
    if x_max is None:
        x_max = dist_max(alpha, r, m, 0)
    f = spec_dist(alpha = alpha, r = r, m = m, t = t, diagonal = True)
    for p in np.arange(10) + 3:
        x_in = 10.**(-p)
        if f(x_in) == 0:
            break

    xs = np.linspace(x_in, x_max, num = int(1/tol) + 1)

    roots_down = 0
    roots_up = 0
    roots = []
    for idx_x, x0 in enumerate(xs[:-1]):
        x1 = xs[idx_x + 1]
        if f(x0) == 0 and f(x1) > 0:
            roots.append((x1+x0)/2)
            roots_up += 1
        elif f(x0) > 0 and f(x1) == 0:
            roots.append((x1 + x0) / 2)
            roots_down += 1
    if len(roots) < 1 or len(roots) > 4 or roots_up > roots_down:
        raise Exception(rf'{roots_up} ascending and {roots_down} descending roots ({roots}) found for alpha = {alpha}, r = {r}, M = {m}, t = {t}.')
    return tuple(roots)

def dist_roots_full(alpha, r, m, t, x_max = None):
    if x_max is None:
        x_max = dist_max(alpha, r, m, 0)
    f = spec_disc(alpha = alpha, r = r, m = m, t = t)
    for p in np.arange(10) + 3:
        x_in = 10.**(-p)
        if f(x_in) < 0:
            break

    xs = np.linspace(x_in, x_max, num = 100)

    intervals = []

    for idx_x, x0 in enumerate(xs[:-1]):
        if f(x0) * f(xs[idx_x + 1]) < 0:
            intervals.append((x0, xs[idx_x + 1]))
    if len(intervals) == 2:
        neg_x = neg_search_golden(f, intervals[0][1], intervals[1][0])
        intervals = [intervals[0], (intervals[0][1], neg_x), (neg_x, intervals[1][0]), intervals[1]]

    roots = [scipy.optimize.brentq(f, *args) for args in intervals]

    return roots

def neg_search_golden(f, a, b):
    invphi = (math.sqrt(5) - 1) / 2

    xs = np.linspace(a, b, 1000)
    idx_high = -1
    idx_low = 0
    # in case the extrema of the interval are local minima, we have to restrict the interval
    while f(xs[idx_high - 1]) > f(xs[idx_high]):
        idx_high -= 1
    while f(xs[idx_low]) < f(xs[idx_low + 1]):
        idx_low += 1
    a = xs[idx_low]
    b = xs[idx_high]


    guess = (a + b) / 2

    while f(guess) >= 0:
        c = b - (b - a) * invphi
        d = a + (b - a) * invphi
        if f(c) < f(d):
            b = d
        else:  # f(c) > f(d) to find the maximum
            a = c
        guess = (a + b) / 2
        #print(guess)

    return (b + a) / 2

def peak2_var(alpha, r, m, t, tol = 1e-4):
    measure = spec_dist(alpha=alpha, r=r, m=m, t=t, diagonal=True)
    roots = dist_roots(alpha = alpha, r = r, m = m, t = t, tol = tol)
    if len(roots) == 4:
        cm1 = scipy.integrate.quad(lambda x: x * measure(x), roots[0], roots[1])[0]
        norm1 = scipy.integrate.quad(lambda x: measure(x), roots[0], roots[1])[0]
        cm2 = scipy.integrate.quad(lambda x: x * measure(x), roots[2], roots[3])[0]
        norm2 = scipy.integrate.quad(lambda x: measure(x), roots[2], roots[3])[0]
        return cm1/norm1, cm2/norm2
    elif len(roots) == 3:
        cm1 = scipy.integrate.quad(lambda x: x * measure(x), 0., roots[0])[0]
        norm1 = scipy.integrate.quad(lambda x: measure(x), 0., roots[0])[0]
        cm2 = scipy.integrate.quad(lambda x: x * measure(x), roots[1], roots[2])[0]
        norm2 = scipy.integrate.quad(lambda x: measure(x), roots[1], roots[2])[0]
        return cm1/norm1, cm2/norm2
    else:
        return None


def peak_sep_t(t_values, alpha, r, m, tol):
    roots_v = np.empty(len(t_values), dtype = float)
    x_max = dist_max(alpha, r, m, 0)
    for idx_t, t in enumerate(tqdm(t_values)):
        roots_v[idx_t] = peak_sep(alpha = alpha, r = r, m = m, t = t, tol = tol, x_max = x_max)
    return roots_v

def peak2_var_t(t_values, alpha, r, m, tol):
    roots_v = np.empty(len(t_values), dtype = float)
    for idx_t, t in enumerate(tqdm(t_values)):
        roots_v[idx_t] = peak_sep(alpha = alpha, r = r, m = m, t = t, tol = tol)
    return roots_v

def peak_cms_diff_t(t_values, alpha, r, m, tol):


    roots_v = np.empty(len(t_values), dtype = float)
    for idx_t, t in enumerate(tqdm(t_values)):
        roots_v[idx_t] = peak_right_cm(alpha, r, m, t, tol) - peak_left_cm(alpha, r, m, t, tol)
    return roots_v

def transf(x, t, alpha = None):
    if alpha is None:
        shift = 0
    else:
        shift = alpha*(1+t)/(1+t*alpha)
    return (1+t)*x/(1+t*x) - shift

def peak_left_max_t(t_values, alpha, r, m, tol):
    args = np.empty(len(t_values), dtype=float)
    vals = np.empty(len(t_values), dtype=float)
    x_max = dist_max(alpha, r, m, 0)
    for idx_t, t in enumerate(tqdm(t_values)):
        roots = dist_roots(alpha=alpha, r=r, m=m, t=t, tol=tol, x_max = x_max)
        assert len(roots) == 4, 'Not enough roots?'
        f = spec_dist(alpha = alpha, r = r, m = m, t = t)
        maxim = scipy.optimize.minimize_scalar(lambda x: -f(x), bounds = (roots[0], roots[1]))
        args[idx_t] = maxim.x
        vals[idx_t] = - maxim.fun

    return args, vals

def vec_tmr(func, t_values, m_values, r_buffer, rank, *args, **kwargs):

    len_x = len(t_values)
    len_y = len(m_values)
    # get the points from just above the transition line
    r_values = [sep_r(alpha=rank / m, m=m) + r_buffer for m in m_values]

    output_test = func(t = t_values[0], m = m_values[0], r = r_values[0], alpha = rank / m_values[0], *args, **kwargs)
    is_tuple = isinstance(output_test, tuple)

    if is_tuple:
        output = np.empty((len(output_test), len_x, len_y), dtype=float)
    else:
        output = np.empty((len_x, len_y), dtype=float)

    with tqdm(total = len_x * len_y) as pbar:
        for idx_m, m in enumerate(m_values):
            r = r_values[idx_m]
            alpha = rank / m
            x_max = dist_max(alpha, r, m, 0, prints = False)
            for idx_t, t in enumerate(t_values):
                if is_tuple:
                    for idx, value in enumerate(func(t = t, m = m, r = r, alpha = alpha, x_max = x_max, *args, **kwargs)):
                        output[idx, idx_t, idx_m] = value
                else:
                    output[idx_t, idx_m] = func(t = t, m = m, r = r, alpha = alpha, x_max = x_max, *args, **kwargs)

                pbar.update(1)

    return output

def vec_mr(func, rank, m_values, r_values, *args, **kwargs):

    len_x = len(m_values)
    len_y = len(r_values)
    output = np.empty((len_x, len_y), dtype=float)

    with tqdm(total = len_x * len_y, disable = False) as pbar:
        for idx_m, m in enumerate(m_values[::-1]):
            alpha = rank / m
            #print(f'Computing for m = {m}. ({idx_m}/50)')
            for idx_r, r in enumerate(r_values):
                t = time()
                #print(f'Computing for r = {r}. ({idx_r}/50)')
                output[idx_m, idx_r] = func(m = m, r = r, alpha = alpha, *args, **kwargs)
                #print(f'Computed output in {time() - t} seconds.')
                pbar.update(1)

    return np.flip(output, axis = 0)

def vec_r(func, r_values, *args, **kwargs):

    len_r = len(r_values)
    output = np.empty(len_r, dtype=float)

    with tqdm(total = len_r, disable = True) as pbar:
        for idx_r, r in enumerate(r_values):
            t = time()
            print(f'Computing for r = {r}. ({idx_r}/50)')
            output[idx_r] = func(r = r, *args, **kwargs)
            print(f'Computed output in {time() - t} seconds.')
            pbar.update(1)

    return output


def peak_sep(alpha, r, m, t, tol = 1e-4, x_max = None):
    roots = dist_roots(alpha = alpha, r = r, m = m, t = t, tol = tol, x_max = x_max)
    if len(roots) == 4:
        return roots[2] - roots[1]
    elif len(roots) == 3:
        return roots[1] - roots[0]
    else:
        return 0

def peak_cms_diff(alpha, r, m, t, x_max = None):
    roots = dist_roots_full(alpha, r, m, t, x_max)
    f = spec_dist(alpha=alpha, r=r, m=m, t=t)
    peak_cm_left = scipy.integrate.quad(lambda x: x * f(x), roots[0], roots[1])[0] / scipy.integrate.quad(lambda x: f(x), roots[0], roots[1])[0]
    peak_cm_right = scipy.integrate.quad(lambda x: x * f(x), roots[2], roots[3])[0] / scipy.integrate.quad(lambda x: f(x), roots[2], roots[3])[0]
    return peak_cm_right - peak_cm_left

def peak_left_cm(alpha, r, m, t, tol = 1e-4, x_max = None):
    measure = spec_dist(alpha=alpha, r=r, m=m, t=t, diagonal=True)
    roots = dist_roots(alpha = alpha, r = r, m = m, t = t, tol = tol, x_max = x_max)
    if len(roots) == 4:
        return scipy.integrate.quad(lambda x: x * measure(x), roots[0], roots[1])[0]/scipy.integrate.quad(lambda x: measure(x), roots[0], roots[1])[0]
    elif len(roots) == 3:
        return scipy.integrate.quad(lambda x: x * measure(x), 0., roots[0])[0]/scipy.integrate.quad(lambda x: measure(x), 0., roots[0])[0]
    else:
        return None

def dist_cm(alpha, r, m, t, x_max = None):
    if x_max is None:
        x_max = dist_max(alpha, r, m, 0)
    measure = spec_dist(alpha=alpha, r=r, m=m, t=t, diagonal=True)
    return scipy.integrate.quad(lambda x: x * measure(x), 0., x_max)[0]

def peak_right_cm(alpha, r, m, t, tol, x_max = None):
    measure = spec_dist(alpha=alpha, r=r, m=m, t=t, diagonal=True)
    roots = dist_roots(alpha = alpha, r = r, m = m, t = t, tol = tol, x_max = x_max)
    if len(roots) == 4:
        return scipy.integrate.quad(lambda x: x * measure(x), roots[2], roots[3])[0]/scipy.integrate.quad(lambda x: measure(x), roots[2], roots[3])[0]
    elif len(roots) == 3:
        return scipy.integrate.quad(lambda x: x * measure(x), roots[1], roots[2])[0]/scipy.integrate.quad(lambda x: measure(x), roots[1], roots[2])[0]
    else:
        return None,

def peak_left_max(alpha, r, m, t, tol, x_max = None):
    roots = dist_roots(alpha=alpha, r=r, m=m, t=t, tol=tol, x_max = x_max)
    if len(roots) == 4:
        x_min = roots[0]
        x_max = roots[1]
    elif len(roots) == 3:
        x_min = 0.
        x_max = roots[0]
    else:
        return None

    f = spec_dist(alpha=alpha, r=r, m=m, t=t)

    return scipy.optimize.minimize_scalar(lambda x: -f(x), bounds=(x_min, x_max)).x

def peak_right_max(alpha, r, m, t, tol, x_max = None):
    roots = dist_roots(alpha=alpha, r=r, m=m, t=t, tol=tol, x_max = x_max)
    if len(roots) == 4:
        x_min = roots[2]
        x_max = roots[3]
    elif len(roots) == 3:
        x_min = roots[1]
        x_max = roots[2]
    else:
        return None

    f = spec_dist(alpha=alpha, r=r, m=m, t=t)

    return scipy.optimize.minimize_scalar(lambda x: -f(x), bounds=(x_min, x_max)).x


def peak_left_tendency(alpha, r, m, t, x_max = None):
    roots = dist_roots_full(alpha, r, m, t, x_max)
    f = spec_dist(alpha=alpha, r=r, m=m, t=t)
    a = roots[0]
    b = roots[1]
    peak_max = scipy.optimize.minimize_scalar(lambda x: -f(x), bounds=(a, b)).x
    peak_cm = scipy.integrate.quad(lambda x: x * f(x), a, b)[0] / scipy.integrate.quad(lambda x: f(x), a, b)[0]
    return peak_max - peak_cm

def t_max_dist(alpha, r, m, x_max = None):
    if x_max is None:
        x_max = dist_max(alpha, r, m, 0, prints = False)
    if sep_alpha(r, m) < alpha:
        return np.nan
    else:
        t0 = 0
        t1 = 1
        while peak_cms_diff(alpha, r, m, t1, x_max=x_max) > peak_cms_diff(alpha, r, m, t0, x_max=x_max):
            t1 += 1
            t0 += 1
        t_max = t1
        t_min = max(0, t0 - 1)
        return scipy.optimize.minimize_scalar(lambda t: - peak_cms_diff(alpha, r, m, t, x_max = x_max), bounds = (t_min, t_max)).x

def t_crossing(alpha, r, m):
    x_max = dist_max(alpha, r, m, 0, prints = False)
    if sep_alpha(r, m) < alpha:
        return np.nan
    else:
        t1 = 1
        while peak_left_tendency(alpha, r, m, t1, x_max = x_max) < 0:
            t1 += 1
        t0 = t1 - 1
        return scipy.optimize.brentq(lambda t: peak_left_tendency(alpha, r, m, t, x_max = x_max), a = t0, b = t1)