import numpy as np

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np

from .app import App

__all__ = ['mkapp_wave']

def wavei_roe(up, um, nor, p, param, time):

    """
    wavei calculate interface roe flux for the wave equation.

       up[np,3]:   np plus states
       um[np,3]:   np minus states
       nor[np,2]:  np normal plus vectors 
       p[np,2]:    np x,y coordinates
       param:      dictionary containing the wave speed c=param['c']
       time:       not used
       fn[np,3]:   np normal fluxes (f plus)  
    """

    c = param['c']
    ca = np.abs(c)

    zer = np.zeros_like(up[:,0])

    fxl = -c*np.column_stack((up[:,2], zer, up[:,0]))
    fyl = -c*np.column_stack((zer, up[:,2], up[:,1]))
    fxr = -c*np.column_stack((um[:,2], zer, um[:,0]))
    fyr = -c*np.column_stack((zer, um[:,2], um[:,1]))

    fav = 0.5*np.diag(nor[:,0]) @ (fxl+fxr) + 0.5*np.diag(nor[:,1]) @ (fyl+fyr)

    qb = 0.5*ca*((up[:,0]-um[:,0]) * nor[:,0] + (up[:,1]-um[:,1]) * nor[:,1])
    ub = 0.5*ca*(up[:,2]-um[:,2])

    fn = np.zeros_like(fav)

    fn[:,0] = fav[:,0] + qb*nor[:,0]
    fn[:,1] = fav[:,1] + qb*nor[:,1]
    fn[:,2] = fav[:,2] + ub

    return fn

def waveb(up, nor, ib, ui, p, param, time):
    """ 
    waveb calculate the boundary flux for the wave equation

       up[np,3]:   np plus states
       nor[np,2]:  np normal plus vectors 
       ib:         boundary type
                   - ib: 0 far-field (radiation)
                   - ib: 1 solid wall(reflection)
                   - ib: 2 non homogenous far-filed (incoming wave)
        ui[3]:     infinity state associated with ib
        p[np,2]:    np x,y coordinates
        param:      dictionary containing 
                    - the wave speed c=param['c']
                    - the wave vector for incoming waves k=param['k']
                    - the wave fucntion f=param['f']
        time:       not used
        fn[np,3]:   np normal fluxes (f plus)  
    """

    if ib == 0:                # Far field
        um = np.matlib.repmat(ui, up[:,0].shape[0], 1)
    elif  ib == 1:            # Reflect
        un = up[:,0]*nor[:,0] + up[:,1]*nor[:,1]
        um = np.column_stack((up[:,0]-2.0*un*nor[:,0], up[:,1]-2.0*un*nor[:,1], up[:,2]))
    elif ib == 2:            # Non-homogenous far-field
        um = np.zeros_like(up)
        k = param['k']
        um[:,2] = param['f'](param['c'], k, p, time)
        kmod = np.sqrt(k[0]**2 + k[1]**2)
        um[:,0] = -k[0]*um[:,2]/kmod
        um[:,1] = -k[1]*um[:,2]/kmod

    fn = wavei_roe( up, um, nor, p, param, time)

    return fn


def wavev(u, p, param, time):
    
    """
    wavev calculate the volume flux for the wave equation.
 
       u[np,3]:    np left (or plus) states
       p:          not used
       param:      dictionary containing the wave speed c=param['c']
       time:       not used
       fx[np,3]:   np fluxes in the x direction (f plus)  
       fy[np,3]:   np fluxes in the y direction (f plus)  
    """
    zer = np.zeros_like(u[:,0])

    c = param['c']
    fx = -c*np.column_stack((u[:,2], zer, u[:,0]))
    fy = -c*np.column_stack((zer, u[:,2], u[:,1]))

    return fx, fy


def mkapp_wave():
    """
    mkapp create application structure template for the linear wave equation.
       app:   application structure
    """

    app = App(3)
    app.pg  = False
    app.arg = {}

    app.finvi = wavei_roe
    app.finvb = waveb
    app.finvv = wavev

    return app