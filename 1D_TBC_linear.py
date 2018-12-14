# Implementation of the unconditionally stable TBC for Schrodinger equation with linear oscillating potential

# Python inports
import math
import cmath
import numpy as np
import io
import matplotlib.pyplot as plt
import aux_functions as aux
import os, sys

# Tkinter imports
import tkinter as tk
import tkinter.filedialog as fd
import tkinter.messagebox as tkm
import tkinter.simpledialog as sdial

# External imports
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)

if __name__ == '__main__':

    RMIN = 30  # ------------------------Gap semi-thickness
    RMAX = 100  # ------------Maximum x
    ZMAX = 1e3  # ----------------Waveguide length, nm
    eps = 0.0001  # Numerical precision

    h = 0.5  # ----------------------------- Transversal step
    tau_int = 1  # ----------------------------- Longitudinal step
    sprsn = 2  # ----------------------------ARRAY thinning(long range)
    sprsm = 1  # ----------------------------ARRAY thinning

    U0 = 0.02  # The potential well depth
    alp1 = 4*U0
    alp0 = 0

    kappa = 0.001  # ------------------------------- The external field strength
    K = 0  # The spatial frequency of the initial condition
    fq = 0.01  # Longitudinal oscillation frequency
    model = 2  # The initial probability model
    kk = 0
    N = 1  # Number of longitudinal oscialltions
    T = N * 2 * math.pi / fq  # ---------------------------------- The external field extent
    T_fq = T * fq  # ------------------------------------- Normalized longitudinal field length
    tau_fq = tau_int * fq  # ------------------------------------- Normalized longitudinal frequency
    K1 = kappa / fq  # ------------------------------------Additional spatial frequency related to the linear potential
    K2 = 2 * K1 / fq
    G0 = aux.sin_G(0, T_fq, K1)
    delta_max = aux.sin_F(T * fq, T_fq, K2)
    N_delta_max = int(math.floor(delta_max/h))

    MMAX = int(round((2. * RMAX + delta_max) / h)) - 1
    muMAX = int(round(2. * RMAX / h/ sprsm)) - 1  # The dimension of sparsed matrices in x direction
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

    # -----------------------------------------------Potential(r)
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
        print(str(progress) + " %")


    rplot = rplot - RMAX

    # Preparing the title string
    buf = io.StringIO()
    buf.write("|u|: eigenvalue = %1.5f,  $WAIST =$ %2.2f $\mu$m,  $AMP =$ %2.2f $\mu$m" \
              % (kk, WAIST * 1e-3, 2*kappa/fq/fq * 1e-3))

    # Plotting the initial field amplitude
    fig1, gplot1 = plt.subplots(figsize=(6, 6), dpi=80)
    gplot1.set_title(buf.getvalue(), y=1.04)
    gplot1.plot(rplot*1e-3, np.log10(np.abs(u0[sprsm * np.r_[0:muMAX]]) ** 2))
    gplot1.set_xlabel('$|u|^2$')
    gplot1.set_ylabel('x, $\mu$m')

    # Preparing the title string
    buf = io.StringIO()
    buf.write("$|u|^2$: K = %1.5f,  $XMAX =$ %4.2f $\mu$m,  $XMIN =$ %4.2f $\mu$m,  $ZMAX =$ %3.0f $\mu$m" \
              % (K, RMIN * 1e-3, RMAX * 1e-3, ZMAX * 1e-3))

    # Plotting the field amplitude in a color chart
    fig2, gplot2 = plt.subplots(figsize=(6, 6), dpi=80)
    gplot2.set_title(buf.getvalue(), y=1.04, x=0.6)
    X, Y = np.meshgrid(zplot * 1e-6, rplot * 1e-3)
    cset = gplot2.pcolormesh(X, Y, np.log10(np.abs(uplot)**2), cmap='jet')
    fig2.colorbar(cset)
    gplot2.set_xlabel('z, mm')
    gplot2.set_ylabel('x, $\mu$m')

    # ----------------------------------------Using tk library to display results
    master = tk.Tk()

    # ------------------------------The program closing function and event handling
    def quit_program():
        sys.exit(0)

    # ------------------------------The color plot function saving the main color plot
    def save_color_plot():
        """Saves the main color plot as a raster image"""
        #  Result file path formation
        fpath = os.path.dirname(sys.argv[0])
        drv = os.path.splitdrive(fpath)
        dirname = drv[0] + "\\Python\\Results"  # ------------ The directory where to save results to

        # ------------ Choosing the name of the image file to save results to
        imagefilename = fd.asksaveasfilename(initialdir = dirname, title = "Choose the file to save the color plot to",\
                                           filetypes = (("png files","*.png"),("all files","*.*")))
        if imagefilename == None:
            fig2.savefig(imagefilename, dpi=600)
            return 1
        else:
            return 0

    # Showing window to adjust color plot properties
    def set_color_plot_properties():
        answer = sdial.askstring("Color scheme", "Enter a name of color scheme", parent=master)
        if answer != None:
            fig2.axes[1].remove()
            gplot2.clear()
            cset = gplot2.pcolormesh(X, Y, np.log10(np.abs(uplot) ** 2), cmap=answer)
            fig2.colorbar(cset)
            canvas2.draw()
            return 1
        else:
            return 0


    # ------------------------------Showing about popup message
    def show_about_message():
        """Saves the main color plot as a raster image"""
        tkm.showinfo("1D finite-difference TBC code", message="Implements 1D finite-difference code with unconditionally stable linear potential TBC.")

    # Reacting on the main window closing event
    master.protocol("WM_DELETE_WINDOW", quit_program)

    # Creating the main menu bar
    mainmenu = tk.Menu(master)

    filemenu = tk.Menu(mainmenu, tearoff=0)
    filemenu.add_command(label="Save color plot", command=save_color_plot)
    filemenu.add_separator()
    filemenu.add_command(label="Exit", command=quit_program)

    plotmenu = tk.Menu(mainmenu, tearoff=0)
    plotmenu.add_command(label="Color plot properties", command=set_color_plot_properties)

    helpmenu = tk.Menu(mainmenu, tearoff=0)
    helpmenu.add_command(label="About", command=show_about_message)

    mainmenu.add_cascade(label="File",menu=filemenu)
    mainmenu.add_cascade(label="Plot", menu=plotmenu)
    mainmenu.add_cascade(label="Help", menu=helpmenu)
    master.config(menu=mainmenu)

    # Popup menus
    colorpopupmenu = tk.Menu(master, tearoff=0)
    colorpopupmenu.add_command(label="Save color plot", command=save_color_plot)
    colorpopupmenu.add_command(label="Color plot properties", command=set_color_plot_properties)

    # Various frame containers
    topframe = tk.Frame(master)
    frame = tk.Frame(master)
    topframe.pack(fill=tk.BOTH)
    frame.pack(fill=tk.BOTH)

    # Canvases for the figures
    canvas1 = FigureCanvasTkAgg(fig1, master=frame)
    canvas1.draw()
    canvas1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
    canvas2 = FigureCanvasTkAgg(fig2, master=frame)
    canvas2.draw()
    canvas2.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)

    # Top label of the window
    window_title = "The results of the PWE solution with a discrete TBC"
    msg = tk.Label(topframe, text=window_title)
    msg.config(bg='lightgreen', font=('times', 14, 'italic'))
    msg.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

    # Popup menu callbacks
    def do_colorpopup(event):
        # display the color properties popup menu
        try:
            colorpopupmenu.tk_popup(event.x_root, event.y_root, 0)
        finally:
            # make sure to release the grab (Tk 8.0a1 only)
            colorpopupmenu.grab_release()

    # Binding colorpopuo callback
    canvas2.get_tk_widget().bind("<Button-3>", do_colorpopup)

    # The main loop
    tk.mainloop()
