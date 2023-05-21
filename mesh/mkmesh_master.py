import numpy as np

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from mesh import Mesh, mkt2f, createnodes, cgmesh
from master import *
from util import *

__all__ = ['mkmesh_master']

def mkmesh_master(porder=1, type=0):
    p = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5*np.sqrt(3.0)]])
    t = np.array([[0, 1, 2]])
    f, t2f =  mkt2f(t)

    fcurved = np.zeros((f.shape[0],))
    tcurved = np.zeros((t.shape[0],))
    plocal, tlocal = localpnts(porder, type)

    mesh = Mesh(p, t, f, t2f, fcurved, tcurved, porder, plocal, tlocal)
    mesh = createnodes(mesh)
    mesh = cgmesh(mesh)

    return mesh

if __name__ == "__main__":
    porder = 2
    mesh = mkmesh_master(porder, 1)
    master = mkmaster(mesh)
    print(master.mass)

    meshplot(mesh, nodes=True)


