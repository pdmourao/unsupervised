import numpy as np
import math
import scipy

# gives the spectral distribution, predicted with Edwards-Jones
def spec_dist(alpha, r, m):

    if np.isinf(m):
        lp = r ** 2 * (1 + np.sqrt(alpha)) ** 2 + alpha * (1 - r ** 2)
        lm = r ** 2 * (1 - np.sqrt(alpha)) ** 2 + alpha * (1 - r ** 2)
        def dist(x):
            if lm < x < lp:
                return np.sqrt((lp - x) * (x - lm)) / (2 * math.pi * r ** 2 * (x - alpha * (1 - r ** 2)))
            else:
                return 0

    elif r == 1:
        lp = (1 + np.sqrt(alpha)) ** 2
        lm = (1 - np.sqrt(alpha)) ** 2
        def dist(x):
            if lm < x < lp:
                return np.sqrt((lp - x) * (x - lm)) / (2 * math.pi * x)
            else:
                return 0

    else:

        mu1 = m / (1 - r ** 2)
        mu2 = 1 / (r ** 2 + (1 - r ** 2) / m)

        def dist(x):
            a = x
            b = (m * alpha - 1) - x * (mu1 + mu2)
            c = (1 - alpha) * mu1 + (1 - alpha * (m - 1)) * mu2 + x * mu1 * mu2
            d = -mu1 * mu2

            p = (3 * a * c - b ** 2) / (3 * a ** 2)
            q = (2 * b ** 3 - 9 * a * b * c + 27 * a ** 2 * d) / (27 * a ** 3)

            disc = (q / 2) ** 2 + (p / 3) ** 3

            if disc < 0:
                return 0
            else:
                return np.sqrt(3) / math.pi / 2 * (np.cbrt(-q / 2 + np.sqrt(disc)) + np.cbrt(q / 2 + np.sqrt(disc)))

    return dist

# gives the delta peak
def peak(alpha, m, r):

    if np.isinf(m):
        pos = alpha * (1 - r ** 2)
        height = max(1 - alpha,0)
    elif r == 1:
        pos = 0
        height = max(1 - alpha,0)
    else:
        pos = 0
        height = max(1 - alpha * m, 0)
    return pos, height


# the spec_integrator is used to integrate a function (integrand) over a spectrum (measure) plus a delta_peak
# the delta peak is given as a tuple (position, height)
def spec_integrator(integrand, measure, min_x=-np.inf, max_x=np.inf, peak_coordinates=(0,0)):
    # new_min, new_max = domain(lambda x: measure(x, **other_kwargs), min_x, max_x)
    pos, height = peak_coordinates
    return integrand(pos) * height + scipy.integrate.quad(lambda x: integrand(x) * measure(x), min_x, max_x)[0]

# gives the moments of the J distribution
def J_moments(n, measure, peak_coordinates):
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
def peak_args(alpha, r, m, p=1, attractor = 'arc', diagonal = False):

    if diagonal:
        shift=alpha
    else:
        shift=0

    if alpha > 0:
        mu2 = J_moments(2, measure = spec_dist(alpha, r, m), peak_coordinates = peak(alpha, r, m))
        print(mu2-alpha**2)
    else:
        mu2 = 0

    if attractor == 'arc':
        std = np.sqrt(mu2 - alpha ** 2 + r ** 2 * p ** 2 * (m - 1) / m ** 2)
        av1 = p *  r ** 2 + shift
        av2 = p * r ** 2 - shift
        return [(1+p)/2, (1-p)/2], [av1,av2], [std,std]
    elif attractor == 'ex':
        print(r ** 4 * p ** 2 * (m - 1) / m ** 2)
        std = np.sqrt(mu2 - alpha ** 2 + r ** 4 * p ** 2 * (m - 1) / m ** 2)
        av1 = 1 / m + shift + r ** 3 * p
        av2 = - 1 / m - shift + r ** 3 * p
        av3 = 1 / m + shift - r ** 3 * p
        av4 = - 1 / m - shift - r ** 3 * p
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
def findroot(f, x1, x2, tol = 1e-3):
    if f(x1) * f(x2) > 0:
        if f(x1) * (f(x2) - f(x1)) > 0:
            return x1
        else:
            return x2
    x = (x2 - x1)/2
    while abs(x2-x1) > tol:
        x = x1 + (x2 - x1)/2
        if f(x) * f(x1) > 0:
            x1 = x
        else:
            x2 = x
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