# Implementation of the unconditionally stable TBC for Schrodinger equation with linear oscillating potential -- test 1

import math
import cmath
import numpy as np
import io
import matplotlib.pyplot as plt
import aux_functions as aux
import os, sys

if __name__ == '__main__':
    #  Result file path formation
    fpath = os.path.dirname(sys.argv[0])
    drv = os.path.splitdrive(fpath)
    imagefilename = drv[0] + "\\Python\\Results\\1D_stable_linear_colorplot.png"  # ------------ The name of the image file to save results to

    RMIN = 30  # ------------------------Gap semi-thickness
    RMAX = 100  # ------------Maximum x
    ZMAX = 1e3  # ----------------Waveguide length, nm
    eps = 0.0001  # Numerical precision

    h = 0.5  # ----------------------------- Transversal step
    tau_int = 1  # ----------------------------- Longitudinal step
    sprsn = 2  # ----------------------------ARRAY thinning(long range)
    sprsm = 1  # ----------------------------ARRAY thinning

    kappa = 0.001  # ------------------------------- The external field strength
    K = 0  # The spatial frequency of the initial condition
    fq = 0.01  # Oscillation frequency
    model = 1  # The initial probability model
    N = 2  # Number of longitudinal oscialltions
    T = N * 2 * math.pi / fq  # ---------------------------------- The external field extent

    MMAX = int(round(2. * RMAX / h)) - 1
    muMAX = math.floor((MMAX + 2) / sprsm)
    MMIN = int(round((RMAX - RMIN) / h)) - 1
    MMIN2 = int(round((RMAX + RMIN) / h)) - 1
    NMAX = int(round(ZMAX / tau_int))
    WAIST = RMIN / 2  # Gaussian beam waist

    # Sparsing parameters
    if sprsn != 1:
        nuMAX = math.floor(NMAX / sprsn) + 1
    else:
        nuMAX = NMAX

    r = np.r_[0:MMAX+2] * h
    z = tau_int * np.r_[0:NMAX]
    K1 = kappa / fq  # ------------------------------------Additional spatial frequency related to the linear potential

    # -----------------------------------------------Potential(r)
    if model == 0:
        # ------------------------------PLANE WAVE
        u0 = aux.planewave_f(r, 0, RMAX, K - K1)
    elif model == 1:
        # ---------------------------------------GAUSSIAN BEAM
        u0 = aux.gaussian_f(r, 0, RMAX, WAIST, K - K1)

    # -------------------------------------
    u = np.copy(u0)

    # ----------------Creating main matrices
    utop = np.zeros(NMAX,dtype=complex)
    ubottom = np.zeros(NMAX,dtype=complex)
    beta = np.zeros(NMAX,dtype=complex)
    gg = np.zeros(NMAX,dtype=complex)
    uplot = np.zeros((muMAX,nuMAX),dtype=complex)
    uplot_exact = np.zeros((muMAX, nuMAX), dtype=complex)

    # ----------------------------------------------MARCHING -- old TBC
    utop[0] = u[MMAX]
    ubottom[0] = u[1]
    zplot = np.zeros(nuMAX)
    rplot = r[sprsm * np.r_[0:muMAX]]
    P = np.ones(MMAX + 2,dtype=complex)
    Q = np.ones(MMAX + 2,dtype=complex)

    # Initializing sparse field amplitude array
    nuu = 1
    zplot[0] = z[0]
    uplot[0:muMAX, 0] = u[sprsm * np.r_[0:muMAX]]

    c0 = 2 * 1j * h**2 / tau_int
    ci = 2. - c0
    cci = 2. + c0
    delta_x = 2*kappa/fq**2  # ---------------------------------The amplitude of x oscillations

    # ----------------------------------------------MARCHING - new TBC
    beta0 = -1j * 2. * cmath.sqrt(c0 - c0**2 / 4.)
    phi = -1. / 2. - (-1.)**np.r_[0:NMAX+1] + ((-1.)**np.r_[0:NMAX+1]) / 2. * ((1. + c0 / 4.) / (1. - c0 / 4.))**np.r_[1:NMAX+2]
    beta[0] = phi[0]
    gg[0] = 1
    qq = 2*1j*math.sin((K - K1) * h) / beta0
    yy = math.cos((K - K1) * h)

    for cntn in np.r_[1:NMAX]:

        # Top and bottom boundary conditions
        gg[cntn] = ((c0 + 2. - 2. * yy) / (c0 - 2. + 2. * yy))**cntn
        betaflipped = beta[cntn-1::-1]  # Flipping the order of coefficients

        SS = -np.dot(ubottom[0:cntn], betaflipped) - ((qq-1)*gg[cntn] - np.dot(gg[0:cntn], betaflipped)) * ubottom[0]
        SS1 = -np.dot(utop[0:cntn], betaflipped) + ((qq+1)*gg[cntn] + np.dot(gg[0:cntn], betaflipped)) * utop[0]
        beta[cntn] = (np.dot(phi[0:cntn], betaflipped) + phi[cntn])/(cntn+1)

        # Initial condition at the bottom
        c = ci
        cconj = cci
        d = u[2] - cconj * u[1] + u[0]
        P[0] = -(c - beta0) / 2.
        Q[0] = -(d - beta0 * SS) / 2.

        # Preparation for marching
        for cntm in np.r_[1:MMAX+1]:
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
            coef = cmath.exp(-1j * kappa ** 2 / fq ** 2 / 2 * tau_int * cntn * fq
                             + 3*1j / 4 * kappa ** 2 / fq ** 3 * math.sin(2 * tau_int * cntn * fq))\
                   * np.exp(1j * K1 * rplot * math.cos(tau_int * cntn * fq))
            uplot[0:muMAX, nuu] = coef * u[sprsm * np.r_[0:muMAX]]

            # Exact solution
            if model == 0:
                # ------------------------------ distorted PLANE WAVE
                uplot_exact[0:muMAX, nuu] = coef * aux.planewave_f(rplot, cntn*tau_int, RMAX, K - K1)
            elif model == 1:
                # --------------------------------------- distorted GAUSSIAN BEAM
                uplot_exact[0:muMAX, nuu] = coef * aux.gaussian_f(rplot, cntn*tau_int, RMAX, WAIST, K - K1)

            nuu = nuu + 1
        # Printing the execution progress
        progress = int(round(1.*(cntn-1) / NMAX * 100))
        print(str(progress) + " %")

    rplot = rplot - RMAX - delta_x*math.sin(T*fq)

    # Preparing the title string
    buf = io.StringIO()
    buf.write("|u|: K = %1.5f,  $XMAX =$ %4.2f $\mu$m,  $XMIN =$ %4.2f $\mu$m,  $ZMAX =$ %5.0f $\mu$m" \
              % (K, RMIN * 1e-3, RMAX * 1e-3, ZMAX * 1e-3))

    # Plotting the field amplitude in a color chart
    fig, gplot = plt.subplots()
    gplot.set_title(buf.getvalue(), y = 1.04)
    X, Y = np.meshgrid(zplot * 1e-3, rplot * 1e-3)
    cset = gplot.pcolormesh(X, Y, np.log10(np.abs(uplot)**2), cmap='jet')
    fig.colorbar(cset)
    gplot.set_xlabel('z, mm')
    gplot.set_ylabel('x, $\mu$m')

    # Saving color plot as a raster image
    fig.savefig(imagefilename, dpi = 600)

    # Plotting the exact field amplitude in a color chart
    fig1, gplot1 = plt.subplots()
    gplot1.set_title(buf.getvalue(), y=1.04)
    cset1 = gplot1.pcolormesh(X, Y, np.log10(np.abs(uplot_exact - uplot) ** 2/np.abs(uplot) ** 2), cmap='jet')
    fig1.colorbar(cset1)
    gplot1.set_xlabel('z, mm')
    gplot1.set_ylabel('x, $\mu$m')

    # Showing the figure
    plt.show()