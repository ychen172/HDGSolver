# Driver for the Euler equations on a duct

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np
import matplotlib.pyplot as plt
import time

from mesh import *
from util import *
from master import *
from dgker import rk4, rinvexpl

from app import mkapp_euler, eulereval

if __name__ == "__main__":
    m = 10
    n = 8
    porder = 4

    time_total  = 10.0
    dt    = 2.0e-03
    nstep = 20
    ncycl = int(np.ceil(time_total/(nstep*dt)))
    gam = 1.4
    Minf = 0.3
    ui = np.array([1.0, 1.0, 0.0, 0.5 + 1/(gam*(gam-1.0)*Minf**2)])
   
    db = 0.2
    mesh = mkmesh_square(m, n, porder, 1)
    # mesh = mkmesh_distort(mesh)       # Uncomment for mesh distortion
    mesh = mkmesh_duct(mesh, db, 0.0, 1.0)
    master = mkmaster(mesh, 2*porder)
    app = mkapp_euler()

    app.bcm = [1, 0, 1, 0]
    app.bcs = np.row_stack((ui, 0.0*ui))
    app.arg['gamma'] = gam

    vv = lambda dg: np.ones((dg.shape[0], dg.shape[2]))*ui[2]

    rho  = lambda dg: 1.0 + 0.01*np.exp(-80*((dg[:,0,:]-1.5)**2 + (dg[:,1,:]-0.5)**2))     # uniform field + perturbation
    ru   = lambda dg: rho(dg)*ui[1]/ui[0]
    rv   = lambda dg: rho(dg)*ui[2]/ui[0]
    pinf = 1.0 / (gam*Minf**2)
    rE   = lambda dg: 0.5*(ru(dg)**2+rv(dg)**2)/rho(dg) + pinf/(gam-1)

    u = initu(mesh, app, [rho, ru, rv, rE])

    pause = lambda : input('(press enter to continue)')
    # meshplot_curved(mesh, True)

    tm = 0.0
    for i in range(ncycl):
        plt.ion()
        scaplot(mesh, eulereval(u, 'r', gam), limits=None, show_mesh=True, pplot=5)

        u = rk4(rinvexpl, master, mesh, app, u, tm, dt, nstep)
        tm = tm + nstep*dt
        print("Time:", tm)
        #time.sleep(1)
        pause()

    plt.ioff()
  