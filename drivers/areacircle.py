import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np

from mesh import *
from util import *
from master import *

def areacircle(siz, porder):
    """    
      areacircle calcualte the area and perimeter of a unit circle
      [area,perim]=areacircle(sp,porder)
   
         siz:       desired element size 
         porder:    polynomial order of approximation (default=1)
         area1:     area of the circle (\pi)
         area2:     area of the circle (\pi)
         perim:     perimeter of the circumference (2*\pi)
    """

    mesh = mkmesh_circle(siz, porder)
    master = mkmaster(mesh)

    # Zero out the return variables
    area1 = 0.0
    area2 = 0.0
    perim = 0.0

    # Lets do integral one
    for i in range(mesh.t.shape[0]):
        # Quadrature
        integral = 0.0
        for j in range(len(master.gwgh)):
            # Calculate Jacobian
            J = np.array([
                [master.shap[:,1,j] @ mesh.dgnodes[:,0,i], master.shap[:,1,j] @ mesh.dgnodes[:,1,i]],
                [master.shap[:,2,j] @ mesh.dgnodes[:,0,i], master.shap[:,2,j] @ mesh.dgnodes[:,1,i]],
                ])
            # Sum contribution
            integral = integral + 1 * master.gwgh[j] * np.linalg.det(J)

        area1 = area1 + integral

    # Lets do integral two
    boundaryIndex = np.where(mesh.f[:,3] == -1)[0]
    nBoundaryFaces = boundaryIndex.shape[0]
    for ii in range(nBoundaryFaces):
        # Look at each trace
        i = boundaryIndex[ii]
        # Which triangle are we on?
        t = mesh.f[i,2]
        # Quadrature
        integral = 0
        ngauss = master.gw1d.shape[0]
        for j in range(ngauss):
            # We need to populate nodes with all the nodes on the face
            nodes = np.full((master.perm.shape[0],2), 0.0)
            ff = np.where(abs(mesh.t2f[t,:])-1 == i)
            for k in range(master.perm.shape[0]):
                nodes[k,:] = mesh.dgnodes[master.perm[k,ff,1],:,t]

            J = np.linalg.norm(master.sh1d[:,1,j] @ nodes)
            d_dxi = master.sh1d[:,1,j] @ nodes / J
            P = master.sh1d[:,0,j] @ nodes / 2.0
            n = np.array([-d_dxi[1], d_dxi[0]])
            # Sum contribution
            integral = integral + P @ n * master.gw1d[j] * J

        area2 = area2 + integral

    # Need to implement the perimeter computation

    return area1, area2, perim


if __name__ == "__main__":
    area1, area2, perim = areacircle(0.5, 4)
    print("area1: \n", area1)
    print("area2: \n", area2)
    print("perim: \n", perim)
