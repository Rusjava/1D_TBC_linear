# Implementation of the unconditionally stable TBC for linear potential

import math
import cmath
import numpy as np
import io
import matplotlib.pyplot as plt
import aux_functions as aux

if __name__ == '__main__':

    imagefilename = "E:\\Python\\Results\\1D_stable_colorplot.png"  # ------------ The name of the image file to save results to

    lam = 0.1  # -----------------Wavelength, nm
    rho = 0.00054

    f1Si = 14.142
    f2Si = 0.141

    den1 = 2.3  # --------------------------------------SiO2
    M1 = 32.0

    delta1 = den1 * rho / M1 * lam**2 * f1Si
    beta1 = den1 * rho / M1 * lam**2 * f2Si

    RMIN = 30  # ------------------------Gap semi-thickness
    RMAX = 100  # ------------Maximum x
    angle = 0  # ------------------------Incidence angle, mrad
    ZMAX = 1e6  # ----------------Waveguide length, nm
    R = 1e6;  # ---------------------------Curvature radius

    h = 0.5  # ----------------------------- Transversal step
    tau_int = 1e3  # ----------------------------- Longitudinal step
    sprsn = 2  # ----------------------------ARRAY thinning(long range)
    sprsm = 1  # ----------------------------ARRAY thinning


    alp1 = -delta1 + 1j * beta1  # Complex dielectric constant definition
    alp0 = 0

    waveguide = 1; #  -------------------------key: 0 -- rod, 1 -- waveguide

    # Vacuum outside
    if waveguide == 0:
        k = 2 * math.pi / lam
    # Vacuum inside
    else:
        k = cmath.sqrt(1 + alp1) * 2 * math.pi / lam
        alp1 = -alp1 / (1 + alp1)

    aa = 2*k**2/R;  # ------------------------------- a parameter

    MMAX = int(round(2. * RMAX / h))
    muMAX = math.floor((MMAX + 2) / sprsm)
    MMIN = int(round((RMAX - RMIN) / h))
    MMIN2 = int(round((RMAX + RMIN) / h))
    NMAX = int(round(ZMAX / tau_int))
    if sprsn != 1:
        nuMAX = math.floor(NMAX / sprsn) + 1
    else:
        nuMAX = NMAX

    r = np.r_[0:MMAX+2] * h
    z = tau_int * np.r_[0:NMAX]

    # -----------------------------------------------Epsilon(r)

    # ------------------------------PLANE WAVE
    u0 = np.exp(1j * k * math.sin(angle) * r)

    # ---------------------------------------GAUSSIAN
    # WAIST = RMIN
    # u0 = np.exp(-(r - RMAX)**2 / WAIST**2 + 1j * k * math.sin(angle) * r)

    # -------------------------------------
    u = np.copy(u0)

    # ----------------Creating main matrices
    utop = np.zeros(NMAX,dtype=complex)
    ubottom = np.zeros(NMAX,dtype=complex)
    alp = np.zeros(MMAX+2,dtype=complex)
    beta = np.zeros(NMAX,dtype=complex)
    gg = np.zeros(NMAX,dtype=complex)
    uplot = np.zeros((muMAX,nuMAX),dtype=complex)

    # ----------------------------------------------MARCHING - - old TBC
    utop[0] = u[MMAX]
    ubottom[0] = u[1]
    zplot = np.zeros(nuMAX)
    rplot = r[sprsm * np.r_[0:muMAX]]
    P = np.ones(MMAX + 2,dtype=complex)
    Q = np.ones(MMAX + 2,dtype=complex)
    c1 = (k*h)**2
    alp0 *= c1
    alp1 *= c1

    # Initializing sparse field amplitude array
    nuu = 1
    zplot[0] = z[0]
    uplot[0:muMAX, 0] = np.exp(1j * k * z[0]) * u[sprsm * np.r_[0:muMAX]]

    c0 = 4. * 1j * k * h**2 / tau_int
    ci = 2. - c0
    cci = 2. + c0

    # ----------------------------------------------MARCHING - new TBC
    beta0 = -1j * 2. * cmath.sqrt(c0 - c0**2 / 4.)
    phi = -1. / 2. - (-1.)**np.r_[0:NMAX+1] + ((-1.)**np.r_[0:NMAX+1]) / 2. * ((1. + c0 / 4.) / (1. - c0 / 4.))**np.r_[1:NMAX+2]
    beta[0] = phi[0]
    gg[0] = 1
    qq = 1j * 2. * cmath.sin(k * math.sin(angle) * h) / beta0
    yy = cmath.cos(k * math.sin(angle) * h)
    mm = cmath.exp(1j * k * math.sin(angle) * h)

    for cntn in np.r_[1:NMAX]:

    #  Dielectric constants with the bent term
        alp[0:MMAX + 2] = aux.d_function(r, RMAX, RMIN, alp1, alp0)

    # Top and bottom boundary conditions
        gg[cntn] = ((c0 + 2. - 2. * yy) / (c0 - 2. + 2. * yy))**cntn

        SS = -np.dot(ubottom[0:cntn], beta[0:cntn]) - ((qq-1)*gg[cntn] - np.dot(gg[0:cntn], beta[0:cntn])) * ubottom[0]
        SS1 = -np.dot(utop[0:cntn], beta[0:cntn]) + ((qq+1)*gg[cntn] + np.dot(gg[0:cntn], beta[0:cntn])) * utop[0]

        beta[cntn] = (np.dot(phi[0:cntn], beta[0:cntn]) + phi[cntn])/(cntn+1)

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
            uplot[0:muMAX, nuu] = np.exp(1j * k * z[cntn-1]) * u[sprsm * np.r_[0:muMAX]]
            nuu = nuu + 1
        # Printing the execution progress
        progress = int(round(1.*(cntn-1) / NMAX * 100))
        print(str(progress) + " %")


    rplot = rplot - RMAX

    # Preparing the title string
    buf = io.StringIO()
    buf.write("|u|: $\lambda =$ %1.2f nm   $XMAX =$ %4.2f $\mu$m  $XMIN =$ %4.2f $\mu$m  $ZMAX =$ %5.0f $\mu$m " \
              % (lam, RMIN * 1e-3, RMAX * 1e-3, ZMAX * 1e-3))

    # Plotting the field amplitude in a color chart
    fig, gplot = plt.subplots()
    gplot.set_title(buf.getvalue())
    X, Y = np.meshgrid(zplot * 1e-6, rplot * 1e-3)
    cset = gplot.pcolormesh(X, Y, np.log10(np.abs(uplot)**2), cmap='jet')
    fig.colorbar(cset)
    gplot.set_xlabel('z, mm')
    gplot.set_ylabel('x, $\mu$m')

    # Saving color plot as a raster image
    fig.savefig(imagefilename, dpi=600)

    # Showing the figure
    plt.show()