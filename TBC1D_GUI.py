# GUI for the 1D TBC code

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

#The main GUI class

class TBC1D_GUI:
    def __init__(self):
        self.master = tk.Tk()
        # Reacting on the main window closing event
        self.master.protocol("WM_DELETE_WINDOW", self.quit_program)
        # Variable declarations
        self.fig1 = None
        self.fig2 = None
        self.gplot1 = None
        self.gplot2 = None
        self.uplot = None
        self.cb = None
        self.X = None
        self.Y = None

        # Creating the main menu bar
        self.mainmenu = tk.Menu(self.master)

        self.filemenu = tk.Menu(self.mainmenu, tearoff=0)
        self.filemenu.add_command(label="Save color plot", command=self.save_color_plot)
        self.filemenu.add_separator()
        self.filemenu.add_command(label="Exit", command=self.quit_program)

        self.plotmenu = tk.Menu(self.mainmenu, tearoff=0)
        self.plotmenu.add_command(label="Color plot properties", command=self.set_color_plot_properties)

        self.helpmenu = tk.Menu(self.mainmenu, tearoff=0)
        self.helpmenu.add_command(label="About", command=self.show_about_message)

        self.mainmenu.add_cascade(label="File", menu=self.filemenu)
        self.mainmenu.add_cascade(label="Plot", menu=self.plotmenu)
        self.mainmenu.add_cascade(label="Help", menu=self.helpmenu)
        self.master.config(menu=self.mainmenu)

        # Popup menus
        self.colorpopupmenu = tk.Menu(self.master, tearoff=0)
        self.colorpopupmenu.add_command(label="Save color plot", command=self.save_color_plot)
        self.colorpopupmenu.add_command(label="Color plot properties", command=self.set_color_plot_properties)

        # Various frame containers
        self.topframe = tk.Frame(self.master)
        self.frame = tk.Frame(self.master)
        self.topframe.pack(fill=tk.BOTH)
        self.frame.pack(fill=tk.BOTH)

        # Top label of the window
        self.window_title = "The results of the PWE solution with a discrete TBC"
        self.msg = tk.Label(self.topframe, text=self.window_title)
        self.msg.config(bg='lightgreen', font=('times', 14, 'italic'))
        self.msg.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

        # The main loop
        tk.mainloop()

    # ------------------------------The program closing method and event handling
    def quit_program(self):
            sys.exit(0)

    # --------------------------------Plotting the graphics
    def plot_graphics(self, uplot, buf1, buf2, rplot, zplot, u0sp):
        """Plotting the calculated graphics"""
        self.uplot = uplot
        self.zplot = zplot
        self.rplot = rplot

        # Plotting the initial field amplitude
        self.fig1, self.gplot1 = plt.subplots(figsize=(6, 6), dpi=80)
        self.gplot1.set_title(buf1.getvalue(), y=1.04)
        self.gplot1.plot(rplot * 1e-3, np.log10(np.abs(u0sp) ** 2))
        self.gplot1.set_xlabel('$|u|^2$')
        self.gplot1.set_ylabel('x, $\mu$m')

        # Plotting the field amplitude in a color chart
        self.fig2, self.gplot2 = plt.subplots(figsize=(6, 6), dpi=80)
        self.gplot2.set_title(buf2.getvalue(), y=1.04, x=0.6)
        self.X, self.Y = np.meshgrid(zplot * 1e-6, rplot * 1e-3)
        cset = self.gplot2.pcolormesh(self.X, self.Y, np.log10(np.abs(uplot) ** 2), cmap='jet')
        self.cb = self.fig2.colorbar(cset)
        self.gplot2.set_xlabel('z, mm')
        self.gplot2.set_ylabel('x, $\mu$m')

        # Canvases for the figures
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=self.frame)
        self.canvas1.draw()
        self.canvas1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=self.frame)
        self.canvas2.draw()
        self.canvas2.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)
        self.frame = tk.Frame(self.master)
        self.frame.pack(fill=tk.BOTH)

        # Binding colorpopuo callback
        self.canvas2.get_tk_widget().bind("<Button-3>", self.do_colorpopup)

    # ------------------------------The color plot function saving the main color plot
    def save_color_plot(self):
        """Saves the main color plot as a raster image"""
        #  Result file path formation
        fpath = os.path.dirname(sys.argv[0])
        drv = os.path.splitdrive(fpath)
        dirname = drv[0] + "\\Python\\Results"  # ------------ The directory where to save results to

        # ------------ Choosing the name of the image file to save results to
        imagefilename = fd.asksaveasfilename(initialdir=dirname, title="Choose the file to save the color plot to", \
                                             filetypes=(("png files", "*.png"), ("all files", "*.*")))
        if imagefilename != '':
            self.fig2.savefig(imagefilename, dpi=600)
            return 0
        else:
            return 1

    # -----------------------------Showing window to adjust color plot properties
    def set_color_plot_properties(self):
        """"Shows a dialog windows for choosing the graphics properties"""
        answer = sdial.askstring("Color scheme", "Enter a name of color scheme", parent=self.master)
        if answer is not None and answer in plt.colormaps():
            self.gplot2.clear()
            self.cb.remove()
            cset = self.gplot2.pcolormesh(self.X, self.Y, np.log10(np.abs(self.uplot) ** 2), cmap=answer)
            self.cb = self.fig2.colorbar(cset)
            self.canvas2.draw()
            return 0
        else:
            return 1

    # ------------------------------Showing about popup message
    def show_about_message(self):
        """Saves the main color plot as a raster image"""
        tkm.showinfo("1D finite-difference TBC code",
                        message="Implements 1D finite-difference code with unconditionally stable linear potential TBC.")

    # ----------------------------------Popup menu callbacks
    def do_colorpopup(self, event):
        """Displays the left graphics popup menu"""
        try:
            self.colorpopupmenu.tk_popup(event.x_root, event.y_root, 0)
        finally:
            # make sure to release the grab (Tk 8.0a1 only)
            self.colorpopupmenu.grab_release()
