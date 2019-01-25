# Implementation of the unconditionally stable TBC for Schrodinger equation with linear oscillating potential

# Python inports
import math
import cmath
import numpy as np
import io
import aux_functions as aux

RMIN = 30  # ------------------------Gap semi-thickness
RMAX = 100  # ------------Maximum x
ZMAX = 1e3  # ----------------Waveguide length, nm
eps = 0.0001  # ------------------Numerical precision
progress = 0

h = 0.5  # ----------------------------- Transversal step
tau_int = 1  # ----------------------------- Longitudinal step
sprsn = 2  # ----------------------------ARRAY thinning(long range)
sprsm = 1  # ----------------------------ARRAY thinning

U0 = 0.02  # ---------------------------The potential well depth
alp1 = 4*U0
alp0 = 0

kappa = 0.001  # ------------------------------- The external field strength
K = 0  # -----------------------The spatial frequency of the initial condition
fq = 0.01  # ------------------------Longitudinal oscillation frequency
model = 2  # -----------------------------The initial probability model
kk = 0
N = 1  # Number of longitudinal oscialltions

# -----------------------------------------------------Arrays for the coordinates
zplot = None
rplot = None
WAIST = 1;

# The main computational function
def compute_amplitude ():
    """"The function computes the amplitude with a given initial condition and with the unconditionally stable TBC"""
    global WAIST, rplot, zplot, alp0, alp1, progress

    T = N * 2 * math.pi / fq  # ---------------------------------- The external field extent
    T_fq = T * fq  # ------------------------------------- Normalized longitudinal field length
    tau_fq = tau_int * fq  # ------------------------------------- Normalized longitudinal frequency
    K1 = kappa / fq  # ------------------------------------Additional spatial frequency related to the linear potential
    K2 = 2 * K1 / fq
    G0 = aux.sin_G(0, T_fq, K1)
    delta_max = aux.sin_F(T * fq, T_fq, K2)
    N_delta_max = int(math.floor(delta_max / h))

    MMAX = int(round((2. * RMAX + delta_max) / h)) - 1
    muMAX = int(round(2. * RMAX / h / sprsm)) - 1  # The dimension of sparsed matrices in x direction
    MMIN = int(round((RMAX - RMIN) / h)) - 1
    MMIN2 = int(round((RMAX + RMIN) / h)) - 1
    NMAX = int(round(ZMAX / tau_int))
    WAIST = RMIN  # Gaussian beam waist

    # Sparsing parameters
    if sprsn != 1:
        nuMAX = math.floor(NMAX / sprsn) + 1
    else:
        nuMAX = NMAX

    # -----------------------------------------------------Arrays for the coordinates
    zplot = np.zeros(nuMAX)
    rplot = h * sprsm * np.r_[0:muMAX]

    # -------------------------------------------------The array for the results
    uplot = np.zeros((muMAX, nuMAX), dtype=complex)

    # -----------------------------------------------Potential(x)
    r = np.r_[0:MMAX+2] * h
    z = tau_int * np.r_[0:NMAX]
    if model == 0:
        # ------------------------------PLANE WAVE
        u0 = aux.planewave_f(r, 0, RMAX, K + G0)
    elif model == 1:
        # ---------------------------------------GAUSSIAN BEAM
        u0 = aux.gaussian_f(r, 0, RMAX, WAIST, K + G0)
    elif model == 2:
        # ----------------------------------The lowest bound state
        kk = aux.ms_energy(U0*RMIN**2, eps)/RMIN
        kk1 = math.sqrt(U0 - kk**2)
        u0 = aux.ms_function(r, RMAX, RMIN, kk, kk1) * np.exp(1j*G0*(r-RMAX))

    # -------------------------------------
    u = np.copy(u0)

    # ----------------Creating main matrices
    utop = np.zeros(NMAX,dtype=complex)
    ubottom = np.zeros(NMAX,dtype=complex)
    alp = np.zeros(MMAX+2,dtype=complex)
    beta = np.zeros(NMAX,dtype=complex)
    gg = np.zeros(NMAX,dtype=complex)

    # ----------------------------------------------MARCHING -- old TBC
    utop[0] = u[MMAX]
    ubottom[0] = u[1]
    P = np.ones(MMAX + 2,dtype=complex)
    Q = np.ones(MMAX + 2,dtype=complex)
    c1 = h**2/4
    alp0 *= c1
    alp1 *= c1

    # Initializing sparse field amplitude array
    nuu = 1
    zplot[0] = z[0]
    uplot[0:muMAX, 0] = u[sprsm * np.r_[0:muMAX]]

    c0 = 2 * 1j * h**2 / tau_int
    ci = 2. - c0
    cci = 2. + c0

    # ----------------------------------------------MARCHING - new TBC
    beta0 = -1j * 2. * cmath.sqrt(c0 - c0**2 / 4.)
    phi = -1. / 2. - (-1.)**np.r_[0:NMAX+1] + ((-1.)**np.r_[0:NMAX+1]) / 2. * ((1. + c0 / 4.) / (1. - c0 / 4.))**np.r_[1:NMAX+2]
    beta[0] = phi[0]
    gg[0] = 1
    qq = -cmath.sin((K + G0)* h) / cmath.sqrt(c0 - c0**2 / 4.)
    yy = cmath.cos((K + G0) * h)

    for cntn in np.r_[1:NMAX]:

        #  Dielectric constants with the bent term
        if cntn*tau_int <= T:
            alp[0:MMAX + 2] = aux.step_p(r - aux.sin_F(cntn*tau_fq, T_fq, K2), RMAX, RMIN, alp1, alp0)
        else:
            alp[0:MMAX + 2] = aux.step_p(r - delta_max, RMAX, RMIN, alp1, alp0)

        # Top and bottom boundary conditions
        gg[cntn] = ((c0 + 2. - 2. * yy) / (c0 - 2. + 2. * yy))**cntn
        betaflipped = beta[cntn - 1::-1]  # Flipping the order of coefficients

        SS = -np.dot(ubottom[0:cntn], betaflipped) - ((qq - 1) * gg[cntn] - np.dot(gg[0:cntn], betaflipped)) * ubottom[0]
        SS1 = -np.dot(utop[0:cntn], betaflipped) + ((qq + 1) * gg[cntn] + np.dot(gg[0:cntn], betaflipped)) * utop[0]
        beta[cntn] = (np.dot(phi[0:cntn], betaflipped) + phi[cntn]) / (cntn + 1)

        # Initial condition at the bottom
        c = ci - alp[1]
        cconj = cci - alp[1]
        d = u[2] - cconj * u[1] + u[0]

        P[0] = -(c - beta0) / 2.
        Q[0] = -(d - beta0 * SS) / 2.

        # Preparation for marching
        for cntm in np.r_[1:MMAX+1]:
            c = ci - alp[cntm]
            cconj = cci - alp[cntm]
            d = u[cntm + 1] - cconj * u[cntm] + u[cntm-1]

            P[cntm] = -1. / (c + P[cntm-1])
            Q[cntm] = -(Q[cntm-1] + d) * P[cntm]

        # Initial condition at the top
        u[MMAX + 1] = (beta0 * SS1 + Q[MMAX-1] - (P[MMAX-1] + beta0) * Q[MMAX]) / (1. - (beta0 + P[MMAX-1]) * P[MMAX])

        # Solving the system
        for cntm in np.r_[MMAX+1:0: -1]:
            u[cntm-1] = Q[cntm-1] - P[cntm-1] * u[cntm]

        # Preserving boundary values
        utop[cntn] = u[MMAX]
        ubottom[cntn] = u[1]

        # Sparsing
        if (cntn-1) / sprsn - math.floor((cntn-1) / sprsn) == 0:
            zplot[nuu] = z[cntn-1]
            #  Multiplying by the phase factor
            if cntn * tau_int <= T:
                coef = cmath.exp(-1j * aux.sin_phi(cntn * tau_fq, T_fq, K1 ** 2)) \
                    * np.exp(-1j * (rplot - RMAX) * aux.sin_G(cntn * tau_fq, T_fq, K1))
                uplot[0:muMAX, nuu] = coef * u[sprsm * np.r_[0:muMAX] \
                                               + int(math.floor(aux.sin_F(cntn * tau_fq, T_fq, K2 / h)))]
            else:
                coef = cmath.exp(-1j * aux.sin_phi(T_fq, T_fq, K1 ** 2)) \
                       * np.exp(-1j * (rplot - RMAX) * aux.sin_G(T_fq, T_fq, K1))
                uplot[0:muMAX, nuu] = coef * u[sprsm * np.r_[0:muMAX] + N_delta_max]
            nuu = nuu + 1
        # Printing the execution progress
        progress = int(round(1.*(cntn-1) / NMAX * 100))

    rplot = rplot - RMAX

    # ---------------------------Returning the results of the computation and the initial condition
    return uplot, u0[sprsm * np.r_[0:muMAX]]



