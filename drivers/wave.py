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

    time_total  = 5.0
    dt    = 5.0e-03
    nstep = 20
    ncycl = int(np.ceil(time_total/(nstep*dt)))
    c = 1.0
   
    mesh = mkmesh_square(m, n, porder)
    master = mkmaster(mesh, 2*porder)

    app = mkapp_wave()
    app.bcm = [1, 0, 0, 1]
    app.bcs = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    app.arg['c'] = c

    init = lambda p: np.exp(-80.0*((p[:,0]-0.5)**2 + (p[:,1]-0.5)**2))  # Gaussian hill
    u = initu(mesh, app, [0.0, 0.0, init])

    pause = lambda : input('(press enter to continue)')
    # meshplot_curved(mesh, True)

    uexact = lambda p: 0.0

    fig, axs = plt.subplots(3)

    tm = 0.0
    for i in range(ncycl):
        plt.ion()
        scaplot_raw(axs[0], mesh, u[:,0,:], limits=[-0.5,0.5], show_mesh=True, pplot=0)
        scaplot_raw(axs[1], mesh, u[:,1,:], limits=[-0.5,0.5], show_mesh=True, pplot=0)
        scaplot_raw(axs[2], mesh, u[:,2,:], limits=[-0.5,1.0], show_mesh=True, pplot=0)

        u = rk4(rinvexpl, master, mesh, app, u, tm, dt, nstep)
        tm = tm + nstep*dt
        print("Time:", tm)
        pause()

    scaplot(mesh, u[:,2,:], limits=[-0.5,1.0], show_mesh=True, pplot=4, interactive=True)
    plt.ioff()
  