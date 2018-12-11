# Implementation of the unconditionally stable TBC for Schrodinger equation with linear oscillating potential

import math
import cmath
import numpy as np
import io
import matplotlib.pyplot as plt
import aux_functions as aux
import os, sys
import tkinter as tk
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)

if __name__ == '__main__':
    #  Result file path formation
    fpath = os.path.dirname(sys.argv[0])
    drv = os.path.splitdrive(fpath)
    imagefilename = drv[0] + "\\Python\\Results\\1D_stable_linear_colorplot.png"  # ------------ The name of the image file to save results to

    RMIN = 30  # ------------------------Gap semi-thickness
    RMAX = 100  # ------------Maximum x
    ZMAX = 3e3  # ----------------Waveguide length, nm
    eps = 0.0001  # Numerical precision

    h = 1  # ----------------------------- Transversal step
    tau_int = 1  # ----------------------------- Longitudinal step
    sprsn = 2  # ----------------------------ARRAY thinning(long range)
    sprsm = 1  # ----------------------------ARRAY thinning

    U0 = 0.1  # The potential well depth
    alp1 = 4*U0
    alp0 = 0

    kappa = 0.0001  # ------------------------------- The external field strength
    K = 0  # The spatial frequency of the initial condition
    fq = 0.01  # Oscillation frequency
    model = 2  # The initial probability model
    kk = 0
    N = 1  # Number of longitudinal oscialltions
    T = N * 2 * math.pi / fq  # ---------------------------------- The external field extent

    MMAX = int(round(2. * RMAX / h)) - 1
    muMAX = math.floor((MMAX + 2) / sprsm)
    MMIN = int(round((RMAX - RMIN) / h)) - 1
    MMIN2 = int(round((RMAX + RMIN) / h)) - 1
    NMAX = int(round(ZMAX / tau_int))
    WAIST = RMIN  # Gaussian beam waist

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
    elif model == 2:
        # ----------------------------------The lowest bound state
        kk = aux.ms_energy(U0*RMIN**2, eps)/RMIN
        kk1 = math.sqrt(U0 - kk**2)
        u0 = aux.ms_function(r, RMAX, RMIN, kk, kk1) * np.exp(-1j*K1*(r-RMAX))

    # -------------------------------------
    u = np.copy(u0)

    # ----------------Creating main matrices
    utop = np.zeros(NMAX,dtype=complex)
    ubottom = np.zeros(NMAX,dtype=complex)
    alp = np.zeros(MMAX+2,dtype=complex)
    beta = np.zeros(NMAX,dtype=complex)
    gg = np.zeros(NMAX,dtype=complex)
    uplot = np.zeros((muMAX,nuMAX),dtype=complex)

    # ----------------------------------------------MARCHING -- old TBC
    utop[0] = u[MMAX]
    ubottom[0] = u[1]
    zplot = np.zeros(nuMAX)
    rplot = r[sprsm * np.r_[0:muMAX]]
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
    delta_x = 2 * kappa / fq ** 2  # ---------------------------------The amplitude of x oscillations

    # ----------------------------------------------MARCHING - new TBC
    beta0 = -1j * 2. * cmath.sqrt(c0 - c0**2 / 4.)
    phi = -1. / 2. - (-1.)**np.r_[0:NMAX+1] + ((-1.)**np.r_[0:NMAX+1]) / 2. * ((1. + c0 / 4.) / (1. - c0 / 4.))**np.r_[1:NMAX+2]
    beta[0] = phi[0]
    gg[0] = 1
    qq = -cmath.sin((K - K1)* h) / cmath.sqrt(c0 - c0**2 / 4.)
    yy = cmath.cos((K - K1) * h)

    for cntn in np.r_[1:NMAX]:

        #  Dielectric constants with the bent term
        if cntn*tau_int <= T:
            alp[0:MMAX + 2] = aux.step_p(r - delta_x*math.sin(fq*tau_int*cntn), RMAX, RMIN, alp1, alp0)
        else:
            alp[0:MMAX + 2] = aux.step_p(r - delta_x, RMAX, RMIN, alp1, alp0)

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
            coef = cmath.exp(-1j*kappa**2/fq**2/2*tau_int*cntn*fq + 3*1j/4*kappa**2/fq**3*math.sin(2*tau_int*cntn*fq))
            uplot[0:muMAX, nuu] = coef * np.exp(1j * K1 * rplot * math.cos(tau_int*cntn*fq))*u[sprsm * np.r_[0:muMAX]]
            nuu = nuu + 1
        # Printing the execution progress
        progress = int(round(1.*(cntn-1) / NMAX * 100))
        print(str(progress) + " %")


    rplot = rplot - RMAX - delta_x

    # Preparing the title string
    buf = io.StringIO()
    buf.write("|u|: eigenvalue = %1.5f,  $WAIST =$ %2.2f $\mu$m,  $AMP =$ %2.2f $\mu$m" \
              % (kk, WAIST * 1e-3, 2*kappa/fq/fq * 1e-3))

    # Plotting the initial field amplitude
    fig, gplot = plt.subplots()
    gplot.set_title(buf.getvalue(), y=1.04)
    gplot.plot(rplot*1e-3, np.log10(np.abs(u0) ** 2))
    gplot.set_xlabel('$|u|^2$')
    gplot.set_ylabel('x, $\mu$m')

    # Preparing the title string
    buf = io.StringIO()
    buf.write("$|u|^2$: K = %1.5f,  $XMAX =$ %4.2f $\mu$m,  $XMIN =$ %4.2f $\mu$m,  $ZMAX =$ %3.0f $\mu$m" \
              % (K, RMIN * 1e-3, RMAX * 1e-3, ZMAX * 1e-3))

    # Plotting the field amplitude in a color chart
    fig, gplot = plt.subplots()
    gplot.set_title(buf.getvalue(), y=1.04, x=0.6)
    X, Y = np.meshgrid(zplot * 1e-6, rplot * 1e-3)
    cset = gplot.pcolormesh(X, Y, np.log10(np.abs(uplot)**2), cmap='jet')
    fig.colorbar(cset)
    gplot.set_xlabel('z, mm')
    gplot.set_ylabel('x, $\mu$m')

    # Saving color plot as a raster image
    fig.savefig(imagefilename, dpi=600)

    # ----------------------------------------Using tk library to display results
    master = tk.Tk()

    # ------------------------------The program closing function and event handling
    def quit_program():
        sys.exit(0)

    master.protocol("WM_DELETE_WINDOW", quit_program)

    # Various containers
    topframe = tk.Frame(master)
    frame = tk.Frame(master)
    topframe.pack(fill=tk.BOTH, expand=1)
    frame.pack(fill=tk.BOTH)

    # Canvas for the figure
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=1)

    # Top label of the window
    window_title = "The results of the PWE solution with a discrete TBC"
    msg = tk.Label(topframe, text=window_title)
    msg.config(bg='lightgreen', font=('times', 14, 'italic'))
    msg.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

    # The main loop
    tk.mainloop()
