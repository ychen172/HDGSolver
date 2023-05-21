import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np

from mesh import *
from master import *
from util import *
from cgker import cg_solve

if __name__ == "__main__":
    porder = 6
    ngrid = 16

    mesh = mkmesh_square(ngrid, ngrid, porder)
    master = mkmaster(mesh, 2*porder)

    kappa = 0.1
    c = [1.0, -2.0]
    s = 1.0

    # Forcing term to obtain u = sin(n*pi*x)*sin(m*pi*y);
    n = 10
    m = 8
    exact = lambda p: np.sin(n*np.pi*p[:,0])*np.sin(m*np.pi*p[:,1])
    source = lambda p: kappa*(n**2+m**2)*np.pi**2*(np.sin(n*np.pi*p[:,0])*np.sin(m*np.pi*p[:,1])) + \
                        c[0]*n*np.pi*(np.cos(n*np.pi*p[:,0])*np.sin(m*np.pi*p[:,1])) + \
                        c[1]*m*np.pi*(np.sin(n*np.pi*p[:,0])*np.cos(m*np.pi*p[:,1])) + \
                        s*(np.sin(n*np.pi*p[:,0])*np.sin(m*np.pi*p[:,1])) 

    param = {'kappa': kappa, 'c': c, 's': s}
    uh, energy = cg_solve(mesh, master, source, param)

    # scaplot(mesh, uh, show_mesh=True)

    uexact = exact(mesh.dgnodes)
    scaplot(mesh, uexact, show_mesh=True)

    # meshplot_curved(mesh, True)

    # L2 norm of the error
    error = np.sqrt(l2_error(mesh, uh, exact))
    print("L2 Error: ", error)

    # meshplot_curved(mesh, True, '')

 

