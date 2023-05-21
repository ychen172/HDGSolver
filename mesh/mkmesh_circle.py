import numpy as np

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import distmesh as dm
from mesh import Mesh, mkt2f, setbndnbrs, createnodes, fixmesh, cgmesh
from util import *
from master import *

__all__ = ['mkmesh_circle']

def mkmesh_circle(siz=0.4, porder=3):
    fd = lambda p: np.sqrt((p**2).sum(1))-1.0
    p, t = dm.distmesh2d(fd, dm.huniform, siz, (-1,-1,1,1))
    p, t = fixmesh(p, t)
    f, t2f =  mkt2f(t)
    bndexpr = [lambda p: np.sqrt((p**2).sum(1))>1.0-1e-3]
    f = setbndnbrs(p, f, bndexpr)

    fcurved = np.full((f.shape[0],), False)
    fcurved[np.where(f < 0)[0]] = True
    tcurved = np.full((t.shape[0],), False)
    tcurved[f[fcurved,2]] = True
    plocal, tlocal = localpnts(porder, 1)

    mesh = Mesh(p, t, f, t2f, fcurved, tcurved, porder, plocal, tlocal)
    mesh = createnodes(mesh, fd)
    mesh = cgmesh(mesh)

    return mesh

if __name__ == "__main__":
    mesh = mkmesh_circle(0.6, 4)
    # meshplot(mesh, True, '')
    meshplot_curved(mesh, True, '', pplot=5)
    # scaplot(mesh, mesh.dgnodes[:,0,:])
    # print(">> mesh.p \n", mesh.p)
    print(">> mesh.f \n", mesh.f)
    print(">> mesh.tcurved \n", mesh.tcurved)
    print(">> mesh.fcurved \n", mesh.fcurved)

