# Auxiliary function definitions

import numpy as np

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