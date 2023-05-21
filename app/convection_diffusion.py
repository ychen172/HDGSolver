import numpy as np

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np

from .app import App

__all__ = ['mkapp_convection_diffusion']

def cdinvv(u, p, param, time):
    """
    cdinvv calculate the volume flux for the linear convection-diffusion equation.
 
       u[np]:      np left (or plus) states
       p[np,2]:    np x,y coordinates
       param:      dictionary containing either 
                   - a constant velocity field [u,v] = param['vf']
                   - a function that returns a velocity field as a function of p vvec =  param['vf'](p)
       time:       not used
       fx[np]:     np fluxes in the x direction (f plus)  
       fy[np]:     np fluxes in the y direction (f plus)  
    """
    vfield = param['vf'](p)
    fx = vfield[:,0][:,None]*u
    fy = vfield[:,1][:,None]*u

    return fx, fy


def cdinvi(up, um, nor, p, param, time):
    """
    cdinvi calculate interface upwind flux for the linear convection-diffusion equation.

       up[np]:     np plus states
       ur[np]:     np minus states
       nor[np,2]:  np normal plus vectors 
       p[np,2]:    np x,y coordinates
       param:      dictionary containing either 
                   - a constant velocity field [u,v] = param['vf']
                   - a function that returns a velocity field as a function of p vvec =  param['vf'](p)
       time:       not used
       fn[np]:     np normal fluxes (f plus)    
    """

    vfield = param['vf'](p)
    vn = np.sum(vfield * nor, axis=1)[:, None]
    avn = np.abs(vn)
    fn = 0.5*vn*(up + um) + 0.5*avn*(up - um)

    return fn

def cdinvb(up, nor, ib, ui, p, param, time):
    """
    cdinvb calculate the boundary flux for the linear convection-diffusion equation.
 
       up[np]:     np plus states
       nor[np,2]:  np normal plus vectors (pointing outwards the p element)
       ib:         boundary type
                   - ib: 1 far-field (radiation)
       ui[1]:      infinity state associated with ib
       p[np,2]:    np x,y coordinates
       param:      dictionary containing either 
                   - a constant velocity field [u,v] = param['vf']
                   - a function that returns a velocity field as a function of p vvec =  param['vf'](p)
       time:       not used
       fn[np]:     np normal fluxes (f plus)  
    """  

    if ib == 0:             # Dirichlet
        um = 0.0*up
    elif ib == 1:           # Neumann
        um = up

    fn = cdinvi(up, um, nor, p, param, time)

    return fn


def cdvisi(up, um, qp, qm, nor, p, param, time):
    """
    cdvisi calculate the viscous interface upwind flux for the linear convection-diffusion equation.

       up[np]:     np plus states
       ur[np]:     np minus states
       qp[np,2]:   npx2 plus q states
       qm[np,2]:   npx2 mins q states
       nor[np,2]:  np normal plus vectors 
       p[np,2]:    np x,y coordinates
       param:      dictionary containing either 
                   kappa = param['kappa']
                   c11int = param['c11int']
       time:       not used
       fn[np]:     np normal fluxes (f plus)    
    """

    kappa = param['kappa']
    c11int = param['c11int']
    fn = -kappa*(qm[:,0]*nor[:,0][:,None] + qm[:,1]*nor[:,1][:,None]) + c11int*(up - um) 

    return fn

def cdvisb(up, qp, nor, ib, ui, p, param, time):
    """
    cdvisb calculate the viscous boundary flux for the convection-diffusion equation.
 
       up[np]:     np plus states
       qp[np,2]:   npx2 plus q states
       nor[np,2]:  np normal plus vectors (pointing outwards the p element)
       ib:         boundary type
                   - ib: 1 far-field (radiation)
       ui:         infinity state associated with ib
       p[np,2]:    np x,y coordinates
       param:      dictionary containing either 
                   kappa = param['kappa']
                   c11 = param['c11']
       time:       not used
       fn[np]:     np normal fluxes (f plus)  
    """  

    kappa = param['kappa']
    c11 = param['c11']
    if ib == 0:             # Dirichlet
        fn = -kappa*(qp[:,0]*nor[:,0][:,None] + qp[:,1]*nor[:,1][:,None]) + c11*(up - ui) 
    elif ib == 1:           # Neumann
        fn = np.zeros_like(up) 

    return fn


def cdvisv(u, q, p, param, time):
    """
    cdvisv calculate the vsicous volume flux for the linear convection-diffusion equation.

       up[np]:     np plus states
       ur[np]:     np minus states
       qp[np,2]:   npx2 plus q states
       qm[np,2]:   npx2 mins q states
       nor[np,2]:  np normal plus vectors 
       p[np,2]:    np x,y coordinates
       param:      dictionary containing either 
                   kappa = param['kappa']
                   c11int = param['c11int']
       time:       not used
       fn[np]:     np normal fluxes (f plus)    
    """

    kappa = param['kappa']
    
    fx = -kappa*q[:,0]
    fy = -kappa*q[:,1]

    return fx, fy

def cdvisub(up, nor, ib, ui, p, param, time):
    """
    cdvisb calculate the viscous boundary flux for the convection-diffusion equation.
 
       up[np]:     np plus states
       nor[np,2]:  np normal plus vectors (pointing outwards the p element)
       ib:         boundary type
                   - ib: 1 far-field (radiation)
       ui:         infinity state associated with ib
       ub[np]:     np values of u at the boundary interface  
    """  

    if ib == 0:             # Dirichlet
        ub = np.zeros_like(up)
    elif ib == 1:           # Neumann
        ub = up

    return ub


def mkapp_convection_diffusion():
    """
    mkapp create application structure template for the linear convection-diffusionequation.
       app:   application structure
    """

    app = App(1)
    app.pg  = True
    app.arg = {}

    app.finvv = cdinvv
    app.finvi = cdinvi
    app.finvb = cdinvb
    app.fvisi = cdvisi
    app.fvisb = cdvisb
    app.fvisv = cdvisv
    app.fvisub = cdvisub

    return app