import numpy as np

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from mesh import Mesh, mkt2f, setbndnbrs, createnodes, squaremesh, fixmesh, cgmesh
from master import *
from util import *

__all__ = ['mkmesh_trefftz']

def mkmesh_trefftz(m=15, n=30, porder=3, type=0, tparam=[0.1, 0.05, 1.98]):
    """
      mkmesh_trefftz creates 2d mesh data structure for trefftz airfoil.
      mesh=mkmesh_trefftz(m,n,porder,tparam)
   
         mesh:      mesh structure
         m:         number of points in the radial direction (default=15)
         n:         number of points in circumferential direction
                    (default=30)
         type:      node spacing type (see localpnts)
         porder:    polynomial order of approximation (default=3)
         tparam:    trefftz foil parameters
                    tparam[0] = left x-shift of circle center 
                                (trailing edge at (1,0)). (default=0.1)
                    tparam[1] = y-shift of circle center. (default=0.05)
                    tparam[2] = k-t exponent (=< 2) (2:jukowski). (default=1.98)       
    """

    n = 2*int(np.ceil(n/2))
    p0, t0 = squaremesh(m, n/2, 0)
    p1, t1 = squaremesh(m, n/2, 1)
    nump = p0.shape[0]
    t1 = t1 + nump
    p1[:,1] = p1[:,1] + 1.0

    p = np.vstack([p0, p1])
    t = np.vstack([t0, t1])

    mesh = Mesh(p, t, None, None, None, None, porder, None, None)

    mesh.plocal, mesh.tlocal = localpnts(mesh.porder, type)
    mesh = createnodes(mesh)

    # First map to a cricle
    mesh.p[:,0] = 2*mesh.p[:,0]
    mesh.p[:,1] = np.pi*mesh.p[:,1]
    z = mesh.p[:,0] + 1j*mesh.p[:,1]
    w = np.exp(z)

    mesh.p[:,0] = w.real
    mesh.p[:,1] = w.imag
    mesh.p, mesh.t = fixmesh(mesh.p, mesh.t)

    mesh.f, mesh.t2f =  mkt2f(mesh.t)
    mesh.fcurved = np.full((mesh.f.shape[0],), True)
    mesh.tcurved = np.full((mesh.t.shape[0],), True)

    bndexpr = [lambda p: np.sqrt((p**2).sum(1))<2.0, lambda p: np.sqrt((p**2).sum(1))>2.0]
    mesh.f = setbndnbrs(mesh.p, mesh.f, bndexpr)

    # Now let's try a K-T transformation
    x0 = tparam[0]
    y0 = tparam[1]
    n  = tparam[2]

    # First Rotate to ensure that a point stays at the trailing edge
    rot = np.arctan2(y0, 1+x0)
    r = np.sqrt((1+x0)**2 + y0**2)

    w = mesh.p[:,0] + 1j*mesh.p[:,1]
    w = r*np.exp(-1j*rot)*w -x0 + 1j*y0

    # Now K-T
    z = ((w-1)/(w+1))**n
    w = ((1+z)/(1-z))*n
    mesh.p[:,0] = w.real
    mesh.p[:,1] = w.imag

    # Now the same for the dgnodes
    z = 2*mesh.dgnodes[:,0,:] + 1j*np.pi*mesh.dgnodes[:,1,:]
    w = np.exp(z)
    w = r*np.exp(-1j*rot)*w -x0 + 1j*y0

    z = ((w-1)/(w+1))**n
    w = ((1+z)/(1-z))*n

    mesh.dgnodes[:,0,:] = w.real
    mesh.dgnodes[:,1,:] = w.imag

    mesh = cgmesh(mesh)

    return mesh


if __name__ == "__main__":
    mesh = mkmesh_trefftz()
    meshplot_curved(mesh, pplot=6)

