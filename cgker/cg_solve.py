import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from .elemmat_cg import elemmat_cg

__all__ = ['cg_solve']

def cg_solve(mesh, master, source, param):
    """
    cg_solve solves the convection-diffusion equation using the cg method.
    [uh,qh,uhath]=hdg_solve(mesh,master,source,dbc,param)
  
        master:       master structure
        mesh:         mesh structure
        source:       source term
        param:        kappa:   = diffusivity coefficient
                        c      = convective velocity
                        s      = source coefficient
        u:            approximate scalar variable
        uh:           approximate scalar variable with local numbering
    """
    
    npl = mesh.plocal.shape[0]
    nt  = mesh.tcg.shape[0]
    nn  = mesh.pcg.shape[0]

    ae = np.empty((npl, npl, nt), dtype=float)
    fe = np.empty((npl, nt), dtype=float)

    for i in range(nt):
        ae[:,:,i], fe[:,i] = elemmat_cg(mesh.pcg[mesh.tcg[i,:],:], master, source, param)

    # Dirichlet boundary conditions
    bou = np.zeros((mesh.pcg.shape[0],), dtype=int)
    ii = np.where(mesh.f[:,3] < 0)[0]

    for i in ii:
        el = mesh.f[i,2]
        ipl = np.sum(mesh.t[el, :]) - np.sum(mesh.f[i, :2])
        isl = np.where(mesh.t[el, :] == ipl)[0]
        bou[mesh.tcg[el, master.perm[:,isl,0]]] = 1

    for i in range(nt):
        for j in range(npl):
            if bou[mesh.tcg[i,j]]:
                ae[j, :, i] = 0.0
                ae[j, j, i] = 1.0
                fe[j, i] = 0.0

    K = lil_matrix((nn, nn))          # should be using a sparse format !!!
    F = np.zeros((nn, 1))

    for i, elem in enumerate(mesh.tcg):
        K[elem[:,None], elem] += ae[:, :, i]
        F[elem, 0] += fe[:, i]

    u = spsolve(K.tocsr(), F)
    energy = 0.5 * u.T @ K @ u - u.T @ F

    # Output uh (DG format) to make it comaptible with scaplot
    uh = np.empty((mesh.dgnodes.shape[0], mesh.dgnodes.shape[2]), dtype=float)
    for i in range(mesh.tcg.shape[0]):
        uh[:,i] = u[mesh.tcg[i,:]]

    return uh, energy 

 

