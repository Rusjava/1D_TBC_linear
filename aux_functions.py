# Auxiliary function definitions

import numpy as np
import math

#  Dielectric function definitions


def parabolic_p(x, x0, rad, v1, v2):
    """Parabolic potential. x is a ndarray; x0, rad are complex numbers; v1, v2 are real numbers"""
    res = np.zeros(x.size,dtype=complex)
    for i in np.r_[0:x.size]:
        if abs(x[i]-x0)>rad:
            res[i] = v2
        else:
            res[i] = (v1-v2)*(1.-((x[i]-x0)/rad)**2) + v2
    return res


def step_p(x, x0, rad, v1, v2):
    """Step-like potential. x is a ndarray; x0, rad are complex numbers; v1, v2 are real numbers"""
    res = np.zeros(x.size,dtype=complex)
    for i in np.r_[0:x.size]:
        if abs(x[i]-x0) > rad:
            res[i] = v2
        else:
            res[i] = v1
    return res


# Temporal oscillations
def sin_temp(t, fq):
    """Oscillating potential field. t is a ndarray; fq (frequancy) is real number"""
    return np.sin(fq*t)


# The main state energy level
def ms_energy(a, eps):
    y = min([math.pi/2, a])
    y_pr = 0
    kappa = math.sqrt(a)
    while abs(y_pr-y)/y > eps:
        y_pr = y
        y = y + 0.1*(math.cos(y) - y/kappa)
    return y


# the main state wave function
def ms_function(x, x0, R, k, kappa):
    res = np.zeros(x.size, dtype=complex)
    for i in np.r_[0:x.size]:
        if abs(x[i]-x0) <= R:
            res[i] = math.cos(k*(x[i]-x0))
        else:
            res[i] = math.cos(k*R)*math.exp(kappa*(R - abs(x0-x[i])))
    return res