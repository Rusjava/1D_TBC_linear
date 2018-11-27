# Auxiliary function definitions

import numpy as np

#  Dielectric function definition: x is a ndarray

def d_function(x, x0, rad, v1, v2):
    """x is a ndarray; x0, rad are complex numbers; v1, v2 are real numbers"""
    res = np.zeros(x.size,dtype=complex)
    for i in np.r_[0:x.size]:
        if abs(x[i]-x0)>rad:
            res[i] = v2
        else:
            res[i] = (v1-v2)*(1.-((x[i]-x0)/rad)**2) + v2
    return res