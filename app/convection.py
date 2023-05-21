import numpy as np

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np

from .app import App

__all__ = ['mkapp_convection']


def convectioni(up, um, nor, p, param, time):
    """
    convectioni calculate interface upwind flux for the linear convection equation.

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

    if callable(param['vf']): 
        vfield = param['vf'](p)
        vn = np.sum(nor * vfield, axis=1)
        avn = abs(vn) 
        fn = 0.5 * vn[:,None] * (up + um) + 0.5 * avn[:,None] * (up - um)
    else:
        vfield = param['vf']
        vn = nor[:,0] * vfield[0] + nor[:,1] * vfield[1]
        avn = abs(vn) 
        fn = 0.5 * vn[:,None] * (up + um) + 0.5 * avn[:,None] * (up - um)

    return fn

def convectionb(up, nor, ib, ui, p, param, time):

    """
    convectionb calculate the boundary flux for the linear convection equation.
 
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

    um = np.matlib.repmat(ui, up.shape[0] ,1)
    fn = convectioni(up, um, nor, p, param, time)

    return fn


def convectionv(u, p, param, time):
    """
    convectionv calculate the volume flux for the linear convection equation.
 
       u[np]:      np left (or plus) states
       p[np,2]:    np x,y coordinates
       param:      dictionary containing either 
                   - a constant velocity field [u,v] = param['vf']
                   - a function that returns a velocity field as a function of p vvec =  param['vf'](p)
       time:       not used
       fx[np]:     np fluxes in the x direction (f plus)  
       fy[np]:     np fluxes in the y direction (f plus)  
    """

    if callable(param['vf']): 
        vfield = param['vf'](p)
        fx = vfield[:,0,None]*u
        fy = vfield[:,1,None]*u
    else:
        vfield = param['vf']
        fx = vfield[0]*u
        fy = vfield[1]*u

    return fx, fy


def mkapp_convection():
    """
    mkapp create application structure template for the linear convection equation.
       app:   application structure
    """

    app = App(1)
    app.pg  = True
    app.arg = {}

    app.finvi = convectioni
    app.finvb = convectionb
    app.finvv = convectionv

    return app