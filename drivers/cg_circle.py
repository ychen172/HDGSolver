import numpy as np

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)


from mesh import *
from master import *
from util import *
from cgker import *


if __name__ == "__main__":
    porder = 6
    siz = 0.2
    #ngrid = 10

    #mesh = mkmesh_square(ngrid, ngrid, porder)
    #mesh = mkmesh_distort(mesh, 0.05)
    mesh = mkmesh_circle(siz, porder)
    meshplot_curved(mesh,True)
    master = mkmaster(mesh, 2*porder)
    source = lambda p: 10.0*np.ones(p.shape[0], dtype=float)

    param = {'kappa': 1.0, 'c': [20, 10], 's': 1.0}
    uh, energy = cg_solve(mesh, master, source, param)

    scaplot(mesh, uh, show_mesh=True, pplot=6)
    # meshplot_curved(mesh, True, '')

 

