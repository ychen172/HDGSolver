import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np
from master import *

__all__ = ['l2_error, initu']

def l2_error(mesh, uh, exact):
    """
    l2_error calculates the l2 norm of the erorr (squared)

    mesh:         mesh structure
    uh:           scalar variable with local numbering
    uexact:        exact solution function
    l2_error:     l2 norm of the error squared
    """

    # Use high order integration to calcualgte the error
    mst = mkmaster(mesh, 4*mesh.porder)
    
    l2error = 0.0
    for i in range(mesh.t.shape[0]):
        shap = mst.shap[:,0,:]
        shapxi = mst.shap[:,1,:]
        shapet = mst.shap[:,2,:]

        dg = mesh.dgnodes[:,:,i]
        xxi = dg[:,0] @ shapxi
        xet = dg[:,0] @ shapet
        yxi = dg[:,1] @ shapxi
        yet = dg[:,1] @ shapet

        jac = xxi * yet - xet * yxi
   
        ug = shap.T @ uh[:,i]
        ugexact = exact(shap.T @ dg)     # exact soilution at quadrature points
        ugerror = ug - ugexact
        l2error = l2error + ugerror.T @ np.diag(mst.gwgh * jac) @ ugerror

    return l2error


  
def initu(mesh, app, value):
    """
    initu initialize vector of unknowns

     mesh:             mesh structure
     app:              application structure
     value[app.nc]:    list containing
                       when value[i] is a float u[:,app.nc,:] = value[i]
                       when value[i] is a function,
                                    u[:,app.nc,:] = value[i](mesh.dgnodes)
    u[npl,app.nc,nt]: scalar fucntion to be plotted
    """

    u = np.zeros((mesh.dgnodes.shape[0], app.nc, mesh.dgnodes.shape[2]))

    for i in range(app.nc):
        if isinstance(value[i], float):
            u[:,i,:] = value[i]
        else:
            u[:,i,:] = value[i](mesh.dgnodes)

    return u

    