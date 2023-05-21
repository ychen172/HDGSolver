# Driver for Linear Wave Equation

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
from dgker import rk4, rinvexpl

from app import mkapp_wave

if __name__ == "__main__":
    m = 11
    n = 20
    porder = 4

    time_total  = 100.0
    dt    = 0.6e-02
    nstep = 20
    ncycl = int(np.ceil(time_total/(nstep*dt)))
    c = 1.0
    k = [3.0, 0.0]
    kmod = np.sqrt(k[0]**2 + k[1]**2)
   
    mesh = mkmesh_trefftz(m, n, porder, 1, [0.0, 0.0, 1.0])
    master = mkmaster(mesh, 2*porder)

    app = mkapp_wave()
    app.pg = True

    app.bcm = [1, 2]        # Manually set boundary conditions
    app.bcs = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    ub = lambda c, k, p, time: np.sin(p[:,0]*k[0] + p[:,1]*k[1] - c * np.sqrt(k[0]**2 + k[1]**2) * time)   
    app.arg['c'] = c
    app.arg['k'] = k
    app.arg['f'] = ub

    u = initu(mesh, app, [0.0, 0.0, 0.0])
    ue = np.zeros_like(u)
    
    # meshplot_curved(mesh)

    pause = lambda : input('(press enter to continue)')
    # meshplot_curved(mesh, True)

    fig, axs = plt.subplots(2)

    tm = 0.0
    for i in range(ncycl):
        ue[:,2,:] = np.sin(mesh.dgnodes[:,0,:]*k[0] + mesh.dgnodes[:,1,:]*k[1] - c * kmod * tm)
        ue[:,0,:] = -k[0]*ue[:,2,:]/kmod
        ue[:,1,:] = -k[1]*ue[:,2,:]/kmod

        if i == 0:          # Set initial condition
            u = ue
    
        plt.ion()
        scaplot_raw(axs[0], mesh, u[:,2,:]-ue[:,2,:], limits=[-1.2,  1.2], show_mesh=True, pplot=7)
        scaplot_raw(axs[1], mesh, u[:,2,:], limits=[-1.2,  1.2], show_mesh=True, pplot=7)

        u = rk4(rinvexpl, master, mesh, app, u, tm, dt, nstep)
        tm = tm + nstep*dt
        print("Time:", tm)
        pause()

    scaplot(mesh, u[:,2,:], limits=[-1.2,  1.2], show_mesh=True, pplot=4, interactive=True)

  