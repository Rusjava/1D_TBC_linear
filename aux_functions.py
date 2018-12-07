# Auxiliary function definitions

import numpy as np
import math, cmath

#  Dielectric function definitions


def parabolic_p(x, x0, rad, v1, v2):
    """Parabolic potential. x is a ndarray; x0, rad are real numbers; v1, v2 are complecx numbers"""
    res = np.zeros(x.size, dtype=complex)
    for i in np.r_[0:x.size]:
        if abs(x[i]-x0) > rad:
            res[i] = v2
        else:
            res[i] = (v1-v2)*(1.-((x[i]-x0)/rad)**2) + v2
    return res


def step_p(x, x0, rad, v1, v2):
    """Step-like potential. x is a ndarray; x0, rad are real numbers; v1, v2 are complex numbers"""
    res = np.zeros(x.size, dtype=complex)
    for i in np.r_[0:x.size]:
        if abs(x[i]-x0) > rad:
            res[i] = v2
        else:
            res[i] = v1
    return res


# Temporal oscillations
def sin_temp(t, fq):
    """Oscillating potential field. t is a ndarray; fq (frequency) is real number"""
    return np.sin(fq*t)


# The main state energy level
def ms_energy(a, eps):
    """Eigenvalue of the ground state in a rectangular potential well"""
    kappa = math.sqrt(a)
    y = min([math.pi/2, kappa])
    y_pr = 0
    incr = min(kappa, 0.1)
    while abs((y_pr-y)/y) > eps:
        y_pr = y
        y += incr*(math.cos(y) - y/kappa)
    return y


# The main state wave function
def ms_function(x, x0, rad, k, kappa):
    """Eigenvector of the ground state in a rectangular potential well for a given eigenvalue"""
    res = np.zeros(x.size, dtype=complex)
    for i in np.r_[0:x.size]:
        if abs(x[i]-x0) <= rad:
            res[i] = math.cos(k*(x[i]-x0))
        else:
            res[i] = math.cos(k * rad) * math.exp(kappa * (rad - abs(x0 - x[i])))
    return res


# Exact solutions and initial functions

def gaussian_f(x, t, x0, rad, K):
    """1D Gaussian solution to Schrodinger ewqustion. x is a ndarray; t, x0, rad, K are real numbers"""
    rad2 = rad**2 + 4*1j*t
    coef = 1./cmath.sqrt(rad2)/math.sqrt(math.pi)
    return coef * np.exp(-(x-x0-2*K*t)**2/rad2 + 1j*K*x - 1j*K**2*t)

def planewave_f(x, t, x0, K):
    """1D plane wave solution to Schrodinger ewqustion. x is a ndarray; t, x0, K are real numbers"""
    return np.exp(1j*K*(x-x0) - 1j*K**2*t)