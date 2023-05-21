# Driver for Contaminant Dispersion Problem

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np
import matplotlib.pyplot as plt

from mesh import *
from util import *
from master import *
from dgker import rk4, rldgexpl

from app import mkapp_convection_diffusion

if __name__ == "__main__":
    m = 21
    n = 6
    porder = 3
    kappa = 0.01

    time_total  = 2.0
    nstep = 5
    dt    = time_total/(nstep*250)
    ncycl = int(np.ceil(time_total/(nstep*dt)))

    c11    = 10.0          # C11 for bounadry edges
    c11int = 10.0          # C11 for interior edges

    mesh = mkmesh_square(m, n, porder)
    mesh.p[:,0] = 10.0*mesh.p[:,0]
    mesh.p[:,1] = 2.5*mesh.p[:,1] - 1.25
    mesh.dgnodes[:,0] = 10.0*mesh.dgnodes[:,0]
    mesh.dgnodes[:,1] = 2.5*mesh.dgnodes[:,1] - 1.25
    master = mkmaster(mesh, 2*porder)

    Re = 100
    g  = 0.5*Re - np.sqrt(0.25*Re**2 + 4.0*np.pi**2)

    xf = lambda p: 1.0 - np.exp(g*p[:,0]) * np.cos(2.0*np.pi*p[:,1])
    yf = lambda p: 0.5*g*np.exp(g*p[:,0]) * np.sin(2.0*np.pi*p[:,1])/np.pi
    vf = lambda p: np.array([xf(p), yf(p)]).T   # Kovasznay flow

    app = mkapp_convection_diffusion()
    app.pg = True

    app.bcm = [1, 1, 1, 0]        # Manually set boundary conditions
    app.bcs = np.array([[0.0], [0.0]])

    app.arg['vf'] = vf
    app.arg['kappa'] = kappa
    app.arg['c11'] = c11
    app.arg['c11int'] = c11int

    gaus = lambda dg, s: np.exp(-((dg[:,0,:] - 1.0)**2 + (dg[:,1,:] - s)**2) / 0.25)
    init = lambda dg: gaus(dg, 0.0) + gaus(dg, 0.5) + gaus(dg, -0.5)

    u = initu(mesh,app,[0.0])

    pause = lambda : input('(press enter to continue)')

    tm = 0.0
    for iper in range(5):
        u = u + initu(mesh, app, [init])

        for i in range(ncycl):
            u = rk4(rldgexpl, master, mesh, app, u, tm, dt, nstep)
            tm = tm + nstep*dt
            print(i)
            if i % 50 == 0:
                plt.ion()
                scaplot(mesh, u, show_mesh=True, pplot=porder+2, interactive=True)
                pause()
                print("Time:", tm)

        


  