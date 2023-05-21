import numpy as np

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from util import *

class Mesh:
    def __init__(self, p, t, f, t2f, fcurved, tcurved, porder, plocal, tlocal):
        self.p = p
        self.t = t
        self.f = f
        self.t2f = t2f
        self.fcurved = fcurved
        self.tcurved = tcurved
        self.porder = porder
        self.plocal = plocal
        self.tlocal = tlocal
        self.dgnodes = None
        self.pcg = None
        self.tcg = None


def mkt2f(t):
    """Compute element connectivities from element indices.

    t2t, t2n = mkt2t(t)
    """
    nt = t.shape[0]
    dim = t.shape[1]-1

    edges = np.vstack(( t[:,[1,2]],
                        t[:,[2,0]],
                        t[:,[0,1]]))

    # Each row of ts is the index of the entry of t when reading the array.
    ts = np.indices(t.shape).reshape(t.ndim,-1,order='F').T

    edges.sort(1)
    _, jx = np.unique(edges, return_inverse=True, axis=0)
    ix = jx.argsort()

    jx = jx[ix]
    ts = ts[ix]

    ix, = (np.diff(jx) == 0).nonzero()

    ts1 = ts[ix]
    ts2 = ts[ix+1]

    t2t = np.empty((nt, dim+1), dtype=int); t2t.fill(-1)

    t2t[ts1[:,0], ts1[:,1]] = ts2[:,0]
    t2t[ts2[:,0], ts2[:,1]] = ts1[:,0]

    nb = sum([np.sum(t2t[:,i] == -1) for i in [0,1,2]])
    nf = (nb + 3*nt) // 2

    f = np.empty((nf, dim+2), dtype=int); f.fill(-1)
    t2f = np.empty((nt, dim+1), dtype=int)

    aux = [0, 1, 2, 0, 1]
    jf = 0
    for i in range(nt):
        for j in [0, 1, 2]:

            n0 = t[i,aux[j+1]]
            n1 = t[i,aux[j+2]]
            ie = t2t[i,j]

            if ie > i:
                if n1 > n0:
                    f[jf,:] = [n0, n1, i, ie]
                    t2f[i,j] = jf + 1
                    t2f[ie,np.where(t[ie,:] == np.sum(t[ie,:])-n0-n1)[0][0]] = -(jf + 1)  
                else:
                    f[jf,:] = [n1, n0, ie, i]
                    t2f[i,j] = -(jf + 1)
                    t2f[ie,np.where(t[ie,:] == np.sum(t[ie,:])-n0-n1)[0][0]] = jf + 1

                jf = jf + 1

    for i in range(nt):
        for j in [0, 1, 2]:

            n0 = t[i,aux[j+1]]
            n1 = t[i,aux[j+2]]
            ie = t2t[i,j]

            if ie == -1:
                f[jf,:] = [n0, n1, i, ie]
                t2f[i,j] = jf + 1
                jf = jf + 1             # remmeber +1 shift to allow negative values

    return f, t2f


def setbndnbrs(p, f, bndexpr):
    """ 
      p:         Node positions (:,2)
      f:         Face Array (:,4)
      bndexpr:   Cell Array of boundary expressions. The 
                 number of elements in BNDEXPR determines 
                 the number of different boundaries

   Example: (Setting boundary types for a unit square mesh - 4 types)
      bndexpr = [lambda p: np.all(p[:,0]<1e-3, lambda p: np.all(p[:,0]>1-1e-3),
                lambda p: np.all(p[:,1]<1e-3, lambda p: np.all(p[:,1]>1-1e-3)]
      f = setbndnbrs(p,f,bndexpr);

   Example: (Setting boundary types for the unit circle - 1 type)
      bndexpr = [lambda p: np.all(np.sqrt((p**2).sum(1))>1.0-1e-3)] 
      f = setbndnbrs(p,f,bndexpr);
    """
    fb = np.where(f == -1)
    for i in fb[0]:
        pb = p[f[i,0:2],:]
        
        found = False
        for ii in range(len(bndexpr)):
            if bndexpr[ii](pb).all():
                found = True
                bnd = ii
                break
        
        if not found:
            raise SystemExit("Can't identify boundary")
        
        f[i,3] = -bnd-1     # shift bnd by 1 to avoid zero based arrays

    return f

def createnodes(mesh, fd=None):
    """
    createdgnodes computes the coordinates of the dg nodes.
    dgnodes=createnodes(mesh,fd)
 
       mesh:      mesh data structure
       fd:        distance function d(x,y)
       dgnodes:   triangle indices (nplx2xnt). the nodes on 
                  the curved boundaries are projected to the
                  true boundary using the distance function fd
    """

    p = mesh.p
    t = mesh.t
    plocal = mesh.plocal

    # Allocte nodes
    dgnodes = np.zeros((plocal.shape[0], 2, t.shape[0]), dtype=float)
    for dim in [0,1]:
        for node in [0,1,2]:
            dp = np.outer(plocal[:,node], p[t[:,node],dim])
            dgnodes[:,dim,:] = dgnodes[:,dim,:] + dp

    # Project nodes on curved boundary
    if fd is not None:
        eps = 1.0e-16
        tc = np.where(mesh.tcurved)[0]
        for it in tc:
            p = dgnodes[:,:,it]
            deps = np.sqrt(eps)*(np.max(p) - np.min(p))
            ed = np.where(mesh.f[abs(mesh.t2f[it,:])-1, 3] < 0)[0]
            for id in ed:
                e = np.where(mesh.plocal[:,id] < 1.0e-6)[0]
                d = fd(p[e,:])
                dgradx = (fd(np.vstack([p[e,0]+deps, p[e,1]]).T) - d)/deps
                dgrady = (fd(np.vstack([p[e,0], p[e,1]+deps]).T) - d)/deps
                dgrad2 = dgradx**2 + dgrady**2
                dgrad2[np.where(dgrad2 == 0)[0]] = 1.0
                p[e,0] = p[e,0] - d*dgradx/dgrad2
                p[e,1] = p[e,1] - d*dgrady/dgrad2

            dgnodes[:,:,it] = p
        
    mesh.dgnodes = dgnodes
    return mesh


def cgmesh(mesh, ptol=2e-13):
    """
    cgmesh: create the continuos high hordewr mesh from a standrard dg mesh
            by eliminating repeated nodes. new fileds created are:
    mesh.pcg(np,2): point coordinates
    mesh.tcg(nt,2): unique node connectivities referred to mesh.pcg
    """

    ph = np.reshape(np.moveaxis(mesh.dgnodes, 2, 0), (-1,2))
    th = np.reshape(np.array([i for i in range(ph.shape[0])]), (mesh.dgnodes.shape[2], -1))

    _, ix, jx = np.unique(np.round(ph, 6), True, True, axis=0)
    ph = ph[ix,:]
    th = jx[th]

    pix, ix, jx = np.unique(th, True, True)
    th = np.reshape(jx, (-1, mesh.dgnodes.shape[0]))
    ph = ph[pix,:]

    mesh.pcg = ph
    mesh.tcg = th

    return mesh


def uniref(p, t, nref=1):
    """Uniformly refine simplicial mesh.

    Usage
    -----
    >>> p, t = uniref(p, t, nref)

    Parameters
    ----------
    p : array, shape (np, dim)
        Nodes
    t : array, shape (nt, dim+1)
        Triangulation
    nref : int, optional
        Number of uniform refinements
    """

    for i in range(nref):
        n = p.shape[0]
        nt = t.shape[0]

        pair = np.vstack((t[:,[0,1]],
                          t[:,[0,2]],
                          t[:,[1,2]]))
        pair.sort(1)
        pair, pairj = np.unique(pair, return_inverse=True, axis=0)
        pmid = (p[pair[:,0]] + p[pair[:,1]])/2
        t1 = t[:,[0]]
        t2 = t[:,[1]]
        t3 = t[:,[2]]
        t12 = pairj[0*nt:1*nt, None] + n
        t13 = pairj[1*nt:2*nt, None] + n
        t23 = pairj[2*nt:3*nt, None] + n
        t = np.vstack((np.hstack((t1,t12,t13)),
                       np.hstack((t12,t23,t13)),
                       np.hstack((t2,t23,t12)),
                       np.hstack((t3,t13,t23))))
        p = np.vstack((p,pmid))

    return p, t
    
    
def mkmesh_distort(mesh, wig=0.05):
    """
    mkmesh_distort distorts a unit square mesh keeping boundaries unchanged
    mesh = mkmesh_distort(mesh,wig)
 
       mesh:     mesh data structure
                    input: mesh for the unit square created with
                           mkmesh_square
                    output: distorted mesh
       wig:      amount of distortion (default: 0.05)
    """

    dx =  -wig*np.sin(2.0*np.pi*(mesh.p[:,1]-0.5))*np.cos(np.pi*(mesh.p[:,0]-0.5))
    dy =   wig*np.sin(2.0*np.pi*(mesh.p[:,0]-0.5))*np.cos(np.pi*(mesh.p[:,1]-0.5))
    mesh.p[:,0] = mesh.p[:,0] + dx
    mesh.p[:,1] = mesh.p[:,1] + dy

    dx =  -wig*np.sin(2.0*np.pi*(mesh.pcg[:,1]-0.5))*np.cos(np.pi*(mesh.pcg[:,0]-0.5))
    dy =   wig*np.sin(2.0*np.pi*(mesh.pcg[:,0]-0.5))*np.cos(np.pi*(mesh.pcg[:,1]-0.5))
    mesh.pcg[:,0] = mesh.pcg[:,0] + dx
    mesh.pcg[:,1] = mesh.pcg[:,1] + dy

    dx =  -wig*np.sin(2.0*np.pi*(mesh.dgnodes[:,1,:]-0.5))*np.cos(np.pi*(mesh.dgnodes[:,0,:]-0.5))
    dy =   wig*np.sin(2.0*np.pi*(mesh.dgnodes[:,0,:]-0.5))*np.cos(np.pi*(mesh.dgnodes[:,1,:]-0.5))
    mesh.dgnodes[:,0,:] = mesh.dgnodes[:,0,:] + dx
    mesh.dgnodes[:,1,:] = mesh.dgnodes[:,1,:] + dy

    mesh.fcurved = np.full((mesh.f.shape[0],), True)
    mesh.tcurved = np.full((mesh.t.shape[0],), True)

    return mesh

def squaremesh(m=10, n=10, parity=0):
    """ 
    squaremesh 2-d regular triangle mesh generator for the unit square
    p, t = squaremesh(m, n, parity)
 
      p:         node positions (np,2)
      t:         triangle indices (nt,3)
      parity:    flag determining the the triangular pattern
                  flag = 0 (diagonals sw - ne) (default)
                  flag = 1 (diagonals nw - se)
    """

    # Generate mesh for unit square
    n = int(n)
    m = int(m)

    x, y = np.meshgrid(np.linspace(0.0, 1.0, m), np.linspace(0.0, 1.0, n))
    p = np.vstack([x.flatten(), y.flatten()]).T

    t = np.zeros((2*(m-1)*(n-1),3), dtype = int)

    ii = 0
    for j in range(n-1):
        for i in range(m-1):
            i0 = i + j*m
            i1 = i0 + 1
            i2 = i0 + m
            i3 = i2 + 1
            if parity == 0:
                t[ii,:] = np.array([i0, i3, i2])
                t[ii+1,:] = np.array([i0, i1, i3])
            else:
                t[ii,:] = np.array([i0, i1, i2])
                t[ii+1,:] = np.array([i1, i3, i2])

            ii = ii + 2

    return p, t


def simpvol(p, t):
    """
    Signed volumes of the simplex elements in the mesh.
    """
        
    d01 = p[t[:,1]]-p[t[:,0]]
    d02 = p[t[:,2]]-p[t[:,0]]
    return (d01[:,0]*d02[:,1]-d01[:,1]*d02[:,0])/2


def fixmesh(p, t, ptol=2e-13):

    """
    Remove duplicated/unused nodes and fix element orientation.

    Parameters
     ----------
    p : array, 
    t : array, 

    Usage
    -----
    p, t = fixmesh(p, t, ptol)
    """

    #ii = np.where(abs(p) < snap)    # This is a little bit of hack because of the way unique_rows works
    #p[ii] = 0.0                     # it treats 0.0  and -0.0 differently !!!

    _, ix, jx = np.unique(np.round(p, 6), True, True, axis=0)

    p = p[ix]
    t = jx[t]

    flip = simpvol(p,t)<0
    t[flip, :2] = t[flip, 1::-1]

    return p, t


if __name__ == "__main__":
    porder = 3
