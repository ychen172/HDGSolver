# Driver for Linear Convection Problem

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

from app import mkapp_convection


if __name__ == "__main__":
    m = 10
    n = 10
    porder = 3

    time_total  = 2.0*np.pi
    dt    = 2.0*np.pi/300.0
    nstep = 25   

    mesh = mkmesh_square(m, n, porder)
    # mesh = mkmesh_distort(mesh)       # uncomment to distrot mesh
    master = mkmaster(mesh, 2*porder)

    app = mkapp_convection()
    app.bcm = [0, 0, 0, 0]
    app.bcs = np.array([[0.0]])

    vf = lambda p: np.column_stack((-p[:,1]+0.5, p[:,0]-0.5))    # velocity field
    app.arg['vf'] = vf

    init = lambda p: np.exp(-120.0*((p[:,0]-0.6)**2 + (p[:,1]-0.5)**2))  # Gaussian hill
    u = initu(mesh, app, [init])

    pause = lambda : input('(press enter to continue)')
    # meshplot_curved(mesh, True)

    tm = 0.0
    while tm < time_total-1.0e-6:
        plt.ion()
        scaplot(mesh, u, [-0.01, 1.2], show_mesh=True, pplot=0, interactive=True)
        u = rk4(rinvexpl, master, mesh, app, u, tm, dt, nstep)
        tm = tm + nstep*dt
        pause()
        print("Time:", tm)

    scaplot(mesh, u, show_mesh=True, pplot=4)
    plt.ioff()
