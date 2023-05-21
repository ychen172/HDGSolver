import numpy as np

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from mesh import Mesh, mkt2f, setbndnbrs, createnodes, squaremesh, cgmesh
from master import *
from util import *


__all__ = ['mkmesh_square']

def mkmesh_square(m=2, n=2, porder=1, parity=0):
    """ 
    mkmesh_square creates 2d mesh data structure for unit square.
    mesh=mkmesh_square(m,n,porder,parity)
 
       mesh:      mesh structure
       m:         number of points in the horizaontal direction 
       n:         number of points in the vertical direction
       porder:    polynomial order of approximation (default=1)
       parity:    flag determining the the triangular pattern
                  flag = 0 (diagonals sw - ne) (default)
                  flag = 1 (diagonals nw - se)
 
    see also: squaremesh, mkt2f, setbndnbrs, uniformlocalpnts, createnodes
    """
    assert n >= 2 and m >= 2
    p, t = squaremesh(m, n, parity)
    f, t2f =  mkt2f(t)

    bndexpr = [lambda p: p[:,1] < 1.0e-3,       lambda p: p[:,0] > 1.0 - 1.0e-3,
               lambda p: p[:,1] > 1.0 - 1.0e-3, lambda p: p[:,0] < 1.0e-3]
    f = setbndnbrs(p, f, bndexpr)

    fcurved = np.full((f.shape[0],), False)
    tcurved = np.full((t.shape[0],), False)
    plocal, tlocal = localpnts(porder, 1)

    mesh = Mesh(p, t, f, t2f, fcurved, tcurved, porder, plocal, tlocal)
    mesh = createnodes(mesh)
    mesh = cgmesh(mesh)

    return mesh

if __name__ == "__main__":
    mesh = mkmesh_square(3,3,3,0)
    # meshplot(mesh, True, 'pt')
    print("p", mesh.p)
    print("t", mesh.t)
    print("t2f", mesh.t2f)
    print("f", mesh.f)
    print("pcg", mesh.pcg)
    print("tcg", mesh.tcg)

    dgnodes = np.empty(mesh.dgnodes.shape, dtype=float)
    for i in range(mesh.t.shape[0]):
        dgnodes[:,:,i] = mesh.pcg[mesh.tcg[i,:],:]

    mesh.dgnodes = dgnodes
    meshplot(mesh, True, annotate='pt')