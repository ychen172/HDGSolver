import numpy as np

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np

from .app import App

__all__ = ['mkapp_euler', 'eulereval']

def euleri_roe(up, um, nor, p, param, time):
    """
    euleri_roe calculate interface roe flux for the euler equations.
 
       ul[np,4]:   np left (or plus) states
       ur[np,4]:   np right (or minus) states
       nor[np,2]:  np normal vectors (pointwing outwars the p element) 
       p[np,2]:    np x,y coordinates
       param:      dictionary containing the value of gamma
       time:       time
       fn[np,4]:   np normal fluxes (f plus)   
    """

    gam = param['gamma']
    gam1  = gam - 1.0
                                             
    nx   = nor[:,0]              
    ny   = nor[:,1]

    rr   = um[:,0]            
    rur  = um[:,1]
    rvr  = um[:,2]
    rEr  = um[:,3]

    rl   = up[:,0]
    rul  = up[:,1]
    rvl  = up[:,2]
    rEl  = up[:,3]

    rr1  = 1.0/rr
    ur   = rur*rr1
    vr   = rvr*rr1
    Er   = rEr*rr1
    u2r  = ur*ur + vr*vr
    pr   = gam1*(rEr - 0.5*rr*u2r)
    hr   = Er+pr*rr1
    unr  = ur*nx + vr*ny

    rl1  = 1.0/rl
    ul   = rul*rl1
    vl   = rvl*rl1
    El   = rEl*rl1
    u2l  = ul*ul + vl*vl
    pl   = gam1*(rEl - 0.5*rl*u2l)
    hl   = El+pl*rl1
    unl  = ul*nx + vl*ny

    fn = 0.5*np.column_stack((  (rr*unr + rl*unl), (rur*unr + rul*unl) + nx*(pr + pl), \
                                (rvr*unr + rvl*unl) + ny*(pr + pl), \
                                (rr*hr*unr + rl*hl*unl) ))
    
    di   = np.sqrt(rr*rl1)     
    d1   = 1.0/(di + 1.0)
    ui   = (di*ur + ul)*d1
    vi   = (di*vr + vl)*d1
    hi   = (di*hr + hl)*d1
    ci2  = gam1*(hi - 0.5*(ui*ui + vi*vi))
    ci   = np.sqrt(ci2)
    af   = 0.5*(ui*ui + vi*vi)
    uni  = ui*nx + vi*ny

    dr    = rr  - rl
    dru   = rur - rul
    drv   = rvr - rvl
    drE   = rEr - rEl

    rlam1 = np.abs(uni + ci)
    rlam2 = np.abs(uni - ci)
    rlam3 = np.abs(uni)

    s1    = 0.5*(rlam1 + rlam2)
    s2    = 0.5*(rlam1 - rlam2)
    al1x  = gam1*(af*dr - ui*dru - vi*drv + drE)
    al2x  = -uni*dr + dru*nx + drv*ny
    cc1   = ((s1 - rlam3)*al1x/ci2) + (s2*al2x/ci)
    cc2   = (s2*al1x/ci)+  (s1 - rlam3)*al2x
      
    fn[:,0]  = fn[:,0] - 0.5*(rlam3*dr  + cc1)
    fn[:,1]  = fn[:,1] - 0.5*(rlam3*dru + cc1*ui + cc2*nx)
    fn[:,2]  = fn[:,2] - 0.5*(rlam3*drv + cc1*vi + cc2*ny)
    fn[:,3]  = fn[:,3] - 0.5*(rlam3*drE + cc1*hi + cc2*uni)

    return fn

def eulerb(up, nor, ib, ui, p, param, time):
    """
    eulerb calculate the boundary flux for the euler equations.
 
       up[np,4]:   np plus states
       not[np,2]:  np normal plus vectors 
       ib:         boundary type
                   - ib: 0 far-field (radiation)
                   - ib: 1 solid wall(reflection)
                   - ib: 2 non homogenous far-filed (incoming wave)
       ui(3):      infinity state associated with ib
       p[np,2]:    np x,y coordinates
       param:      dictionary containing the value of gamma
       time:       time
       fn[np,4]:   np normal fluxes (f plus)  
    """  

    if ib == 0:                 # Far field
        um = np.matlib.repmat(ui, up[:,0].shape[0], 1)
    elif  ib == 1:              # Reflect
        un = up[:,1]*nor[:,0] + up[:,2]*nor[:,1]
        um = np.column_stack((up[:,0], up[:,1]-2.0*un*nor[:,0], up[:,2]-2.0*un*nor[:,1], up[:,3]))

    fn = euleri_roe(up, um, nor, p, param, time)

    return fn


def eulerv(u, p, param, time):
    """
    eulerv calculate the volume flux for the euler equations.
 
       u[np,4]:      np left (or plus) states
       p:            not used
       param:        dictionary containing the value of gamma
       time:         not used
       fx[np,4]:     np fluxes in the x direction (f plus)  
       fy[np,4]:     np fluxes in the y direction (f plus)  
    """

    gam = param['gamma']


    uv = u[:,1]/u[:,0]
    vv = u[:,2]/u[:,0]

    p = (gam-1.0)*(u[:,3] - 0.5*(u[:,1]*uv + u[:,2]*vv))

    fx = np.column_stack((u[:,1], u[:,1]*uv + p,   u[:,2]*uv,     uv*(u[:,3] + p)))
    fy = np.column_stack((u[:,2], u[:,1]*vv,       u[:,2]*vv + p, vv*(u[:,3] + p)))

    return fx, fy

def eulereval(u, str, gam):
    """
    eulerval calculates derived quantities for the euler equation variables.
 
       up[npl,4,nt]:   np plus states
       str:            string used to specify requested quantity
                       - str: 'r' density
                       - str: 'u' u_x velocity
                       - str: 'v' u_y velocity
                       - str: 'p' density
                       - str: 'm' density
                       - str; 's' entropy
       gam:            value of gamma
       sca[npl,4,nt]:  scalar field requested by str 
    """

    if str == 'r':
        sca = u[:,0,:]
    elif str == 'p':
        uv = u[:,1,:]/u[:,0,:]
        vv = u[:,2,:]/u[:,0,:]
        sca = (gam-1)*(u[:,3,:] - 0.5*(u[:,1,:]*uv + u[:,2,:]*vv))
    elif str == 'c':
        uv = u[:,1,:]/u[:,0,:]
        vv = u[:,2,:]/u[:,0,:]
        p = (gam-1)*(u[:,3,:] - 0.5*(u[:,1,:]*uv + u[:,2,:]*vv))
        sca = np.sqrt(gam*p/u[:,0,:])
    elif str == 'Jp':
        uv = u[:,1,:]/u[:,0,:]
        vv = u[:,2,:]/u[:,0,:]
        p = (gam-1)*(u[:,3,:] - 0.5*(u[:,1,:]*uv + u[:,2,:]*vv))
        c = np.sqrt(gam*p/u[:,0,:])
        sca = u[:,1,:] + 2*c/(gam-1)
    elif str == 'Jm':
        uv = u[:,1,:]/u[:,0,:]
        vv = u[:,2,:]/u[:,0,:]
        p = (gam-1)*(u[:,3,:] - 0.5*(u[:,1,:]*uv + u[:,2,:]*vv))
        c = np.sqrt(gam*p/u[:,0,:])
        sca = u[:,1,:] - 2*c/(gam-1)
    elif str == 'M':
        uv = u[:,1,:]/u[:,0,:]
        vv = u[:,2,:]/u[:,0,:]
        u2 = np.sqrt(uv**2 + vv**2)
        p = (gam-1)*(u[:,3,:] - 0.5*(u[:,1,:]*uv + u[:,2,:]*vv))
        sca = u2/np.sqrt(gam*p/u[:,0,:])
    elif str == 's':
        uv = u[:,1,:]/u[:,0,:]
        vv = u[:,2,:]/u[:,0,:]
        p = (gam-1)*(u[:,3,:] - 0.5*(u[:,1,:]*uv + u[:,2,:]*vv))
        sca = p/(u[:,0,:]**gam)
    elif str == 'u':
        sca = u[:,1,:]/u[:,0,:]
    elif str == 'v':
        sca = u[:,2,:]/u[:,0,:]
    else:
        print('unknonw case')
        exit(1)

    return sca


def mkapp_euler():
    """
    mkapp create application structure template for the linear convection equation.
       app:   application structure
    """

    app = App(4)
    app.pg  = False
    app.arg = {}

    app.finvi = euleri_roe
    app.finvb = eulerb
    app.finvv = eulerv

    return app