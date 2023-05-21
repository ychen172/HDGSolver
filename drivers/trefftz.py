import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import cmath

import numpy as np
import scipy

from mesh import *
from util import *
from master import *

def potential_trefftz( x, y, V=1.0, alpha=0.0, tparam=[0.1, 0.05, 1.98]):
    """
    potential_trefftz calculates the 2d potential flow for trefftz airfoil.
    [psi,velx,vely,gamma]=potential_trefftz(x,y,v,alpha,tparam)
 
       psi:       value of stream function at input points
       velx:      x component of the velocity at input points
       vely:      y component of the velocity at input points
       gamma:     circultation. lift force= v*gamma
       x:         x coordinates of input points 
       y:         y coordinates of input points 
       v:         free stream velocity magnitude (default=1)
       alpha:     angle of attack in degrees (default=0)
       tparam:    trefftz foil parameters
                  tparam[0] = left x-shift of circle center 
                              (trailing edge at (1,0)). (default=0.1)
                  tparam[1] = y-shift of circle center. (default=0.05)
                  tparam[2] = k-t exponent (=< 2) (2:jukowski). (default=1.98)                      
    """

    x0 = float(tparam[0])
    y0 = float(tparam[1])
    n  = float(tparam[2])

    # First Rotate to ensure that a point stays at the trailing edge
    rot = np.arctan2(y0, 1+x0)
    r = np.sqrt((1+x0)**2 + y0**2)

    zr = x
    zi = y
    z = zr + 1j * zi

    # First calcualte an approximate camber line (ugly code to ensure we are in the correct branch)
    cc = -x0 + 1j*y0
    th = np.linspace(0, 2*np.pi, 121)
    xc = (1-cc)*np.exp(1j*th)
    wd = cc + xc
    zd = ((wd-1)/(wd+1))**n
    wd = ((1+zd)/(1-zd))*n
    xle = min(wd.real)
    ii = np.argmin(wd.real)

    sup = scipy.interpolate.CubicSpline(np.flip(wd.real[:ii+1]), np.flip(wd.imag[:ii+1]))
    slo = scipy.interpolate.CubicSpline(wd.real[ii:], wd.imag[ii:])

    # Now K-T inverse
    A = ((z-n)/(z+n))
    anga = np.vectorize(cmath.phase)(A)

    il = np.where(
                (zr > xle) & 
                (zr <= n) &
                (zi < 0.5*(sup(zr) + slo(zr))) &
                (anga > 1.5)
            )
    iu = np.where(
                (zr > xle) & 
                (zr <= n) &
                (zi > 0.5*(sup(zr) + slo(zr))) &
                (anga < -1.5)
            )
    
    anga[il] = anga[il] - 2.0*np.pi
    anga[iu] = anga[iu] + 2.0*np.pi

    B = np.absolute(A**(1.0/n))*np.exp(1j*anga/n)
    v = (1.0+B)/(1.0-B)

    # scale back
    w = (1.0/r) * np.exp(1j*rot) * (v + x0 - 1j * y0)

    # Now we have a unit circle
    alphef = np.pi * alpha/180.0 + rot

    # Calculate circulation
    dphidw = -V * (np.exp(-1j*alphef) - 1.0/np.exp(-1j*alphef))
    dvortdw = 1j /(2*np.pi)
    Gamma = -(dphidw/dvortdw).real

    phi = -V * r* (w*np.exp(-1j*alphef) + 1.0/(w*np.exp(-1j*alphef)))
    vort = 1j * Gamma * np.log(w)/(2*np.pi)
    psi =  (phi+vort).imag

    # Find trailing edge
    ii = np.where(np.absolute(w-1) < 1.e-6)
    if len(ii) > 0:
        w[ii] =  complex(2.0, 0.0) 

    dphidw = -V * r * (np.exp(-1j*alphef) - 1.0/(w*w*np.exp(-1j*alphef)))
    dvortdw = 1j * Gamma/(2*np.pi*w)

    dwdv = (1/r) * np.exp(1j*rot)
    dvdB = 2.0/(1.0-B)**2

    aux = np.absolute(A)
    jj = np.where(aux > 1.0e-12)
    aux[jj] = aux[jj]**((1.0-n)/n)
    dBdz = (1.0/n) * aux * np.exp(1j*anga*(1.0-n)/n) * (1-A)/(z+n)

    dphi = (dphidw + dvortdw) * dwdv * dvdB * dBdz

    # set unbounded derivatieve at trailing edge to zero
    if len(ii) > 0:
        dphi[ii] =  complex(0.0, 0.0) 

    velx = -dphi.real
    vely =  dphi.imag

    return psi, velx, vely, Gamma


# def trefftz_points(tparam=[0.1, 0.05, 1.98], num=120):
#     """
#     trefftz_points calculates np points on trefftz airfoil surface.
#     [x,y,chord]=trefftz_points(tparam,np)
  
#         x:         x coordinates of generated points 
#         y:         y coordinates of generated points 
#         chord:     foil chord
#         tparam:    trefftz foil parameters
#                    tparam(1) = left x-shift of circle center 
#                                (trailing edge at (1,0)). (default=0.1)
#                    tparam(2) = y-shift of circle center. (default=0.05)
#                    tparam(3) = k-t exponent (=< 2) (2:jukowski). (default=1.98)
#         num:       number of points requested
#     """

#     x0 = float(tparam[0])
#     y0 = float(tparam[1])
#     n  = float(tparam[2])

#     # First Rotate to ensure that a point stays at the trailing edge
#     rot = np.arctan2(y0, 1+x0)
#     r = np.sqrt((1+x0)**2 + y0**2)

#     # K-T transform
#     cc = -x0 + 1j*y0
#     th = np.linspace(0, 2*np.pi, num+1)
#     xc = (1-cc)*np.exp(1j*th)
#     wd = cc + xc
#     zd = ((wd-1)/(wd+1))**n
#     wd = ((1+zd)/(1-zd))*n
#     xle = min(wd.real)

#     chord = n - xle

#     x = wd.real
#     y = wd.imag

#     return x, y, chord

def getVelocity(mesh, psi):
    # Calcaulte velocity by differentitating stream function

    # How many triangles and local points do we have?
    (npoints, nt) = psi.shape

    vx = np.zeros((npoints, nt), dtype=float)
    vy = np.zeros((npoints, nt), dtype=float)

    shapnodes = shape2d(mesh.porder, mesh.plocal, mesh.plocal[:,1:])

    for i in range(nt):
        # Calculate Jacobian
        J11 = mesh.dgnodes[:,0,i] @ shapnodes[:,1,:]     # DxDxi1
        J12 = mesh.dgnodes[:,0,i] @ shapnodes[:,2,:]     # DxDxi2
        J21 = mesh.dgnodes[:,1,i] @ shapnodes[:,1,:]     # DyDxi1
        J22 = mesh.dgnodes[:,1,i] @ shapnodes[:,2,:]     # DyDxi2

        detJ = J11*J22 - J12*J21

        Jinv11 =  J22/detJ
        Jinv12 = -J12/detJ
        Jinv21 = -J21/detJ
        Jinv22 =  J11/detJ

        vx[:,i] = - (np.diag(Jinv12) @ shapnodes[:,1,:].T  + np.diag(Jinv22) @ shapnodes[:,2,:].T) @ psi[:,i]
        vy[:,i] =   (np.diag(Jinv11) @ shapnodes[:,1,:].T  + np.diag(Jinv21) @ shapnodes[:,2,:].T) @ psi[:,i]

    return vx, vy

def getCoefficients(mesh, master, cp, alpha):
    cx = 0.0
    cy = 0.0
    cm = 0.0

    airfoilbc = 0
    ii = np.where(mesh.f[:,3] == -airfoilbc - 1)[0]

    xle =  100000.0
    xte = -100000.0
    for (j, i) in enumerate(ii):
        it = mesh.f[i,2]
        lc = np.where(abs(mesh.t2f[it,:]) - 1 == i)[0][0]

        xc = mesh.dgnodes[master.perm[:,lc,0],0,it]
        xle = np.minimum(xle, np.min(xc))
        xte = np.maximum(xte, np.max(xc))

        xg = mesh.dgnodes[master.perm[:,lc,0],0,it] @ master.sh1d[:,0,:]
        yg = mesh.dgnodes[master.perm[:,lc,0],1,it] @ master.sh1d[:,0,:]

        Dxdxig = mesh.dgnodes[master.perm[:,lc,0],0,it] @ master.sh1d[:,1,:]
        Dydxig = mesh.dgnodes[master.perm[:,lc,0],1,it] @ master.sh1d[:,1,:]

        # jac = np.sqrt(Dxdxig**2 + Dydxig**2)

        cpg = cp[master.perm[:,lc,0],it] @ master.sh1d[:,0,:]

        cx = cx + np.dot( cpg*Dydxig, master.gw1d)
        cy = cy + np.dot(-cpg*Dxdxig, master.gw1d)
        cm = cm + np.dot( cpg*Dydxig, master.gw1d*yg)  - np.dot(-cpg*Dxdxig, master.gw1d*xg)

    chord = xte - xle

    cm = (cm + cy*(xle + 0.25*chord)) / chord**2
    cx = cx / chord
    cy = cy / chord

    cd =  cx * np.cos(alpha*np.pi/180.0) + cy * np.sin(alpha*np.pi/180.0)
    cl = -cx * np.sin(alpha*np.pi/180.0) + cy * np.cos(alpha*np.pi/180.0)

    return cl, cd, cm, chord


def trefftz( Vinf, alpha, m=15, n=30, porder=3, type=0, tparam=[0.1, 0.05, 1.98]):

    mesh = mkmesh_trefftz(m, n, porder, type, tparam)
    master = mkmaster(mesh)

    # number of triangles
    nt = mesh.t.shape[0]
    # number of local points
    npoints = mesh.dgnodes.shape[0]

    # allocate psi structure
    psi = np.zeros((npoints, nt), dtype=float)

    velx = np.zeros((npoints, nt), dtype=float)
    vely = np.zeros((npoints, nt), dtype=float)

    for i in range(nt):
        psi[:,i], velx[:,i], vely[:,i], Gamma, = potential_trefftz(mesh.dgnodes[:,0,i], mesh.dgnodes[:,1,i], Vinf, alpha, tparam)

    # Now we need to calculate vx = -dpsi_dy and vy = dpsi_dx
    vx, vy = getVelocity( mesh, psi)

    cp = 1 - (vx**2 + vy**2)/Vinf**2
    scaplot(mesh, cp, limits=[-4.0, 1.0], show_mesh=True)

    cl, cd, cm, chord = getCoefficients(mesh, master, cp, alpha)

    cl_exact = -2.0*Gamma/(Vinf*chord)
    return cl, cd, cm, cl_exact


if __name__ == "__main__":
    cl, cd, cm, cl_exact = trefftz(1.0, 10., 30, 40, 3, 1)
    print("cl: \n", cl)
    print("cd: \n", cd)
    print("cm: \n", cm)
    print("cl_exact: \n", cl_exact)
