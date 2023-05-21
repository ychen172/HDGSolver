import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from master import shape2d

__all__ = ['hdg_postprocess']

""" Legacy tried to implement conv-diffusion
def localLHS(dg, master, param):

    porder = master.porder

    kappa = param['kappa']
    c     = param['c']
    taud  = kappa

    nps   = porder+1 #nPoly1d
    npl   = dg.shape[0] #nPoly

    perm = master.perm[:,:,0] #nPoly1d X 3 corners CCW


    # Volume integral
    shap = master.shap[:,0,:]    #nPoly X nQuad
    shapxi = master.shap[:,1,:]  #nPoly X nQuad
    shapet = master.shap[:,2,:]  #nPoly X nQuad
    sh1d = np.squeeze(master.sh1d[:,0,:]) # sh1d [nPoly X nQuad]

    xxi = dg[:,0] @ shapxi     #dxdxi [nQuad X 1]
    xet = dg[:,0] @ shapet     #dxdet [nQuad X 1]
    yxi = dg[:,1] @ shapxi     #dydxi [nQuad X 1]
    yet = dg[:,1] @ shapet     #dydet [nQuad X 1]
    jac = xxi * yet - xet * yxi #Det(Jac) [nQuad X 1]

    M11 = (shap @ np.diag(master.gwgh * yet) @ shapxi.T)/kappa - (shap @ np.diag(master.gwgh * yxi) @ shapet.T)/kappa #[nPoly X nPoly]
    M12 = np.zeros([npl,npl]) 
    M13 = (shapet @ np.diag(master.gwgh * yxi) @ shap.T) - (shapxi @ np.diag(master.gwgh * yet) @ shap.T) # [nPoly X nPoly]
    M14 = np.zeros([npl,nps*3]) # [nPoly X 3nps]
    
    M21 = np.zeros([npl,npl])
    M22 = (shap @ np.diag(master.gwgh * xxi) @ shapet.T)/kappa - (shap @ np.diag(master.gwgh * xet) @ shapxi.T)/kappa #[nPoly X nPoly]
    M23 = (shapxi @ np.diag(master.gwgh * xet) @ shap.T) - (shapet @ np.diag(master.gwgh * xxi) @ shap.T) # [nPoly X nPoly]
    M24 = np.zeros([npl,nps*3]) # [nPoly X 3nps]
    
    M31 = -(shapxi @ np.diag(master.gwgh * yet) @ shap.T) + (shapet @ np.diag(master.gwgh * yxi) @ shap.T) # [nPoly X nPoly]
    # M31 = M31 - (shap @ np.diag(master.gwgh * yet) @ shapxi.T - shap @ np.diag(master.gwgh * yxi) @ shapet.T)
    M32 =  (shapxi @ np.diag(master.gwgh * xet) @ shap.T) - (shapet @ np.diag(master.gwgh * xxi) @ shap.T) # [nPoly X nPoly]
    # M32 = M32 - (shap @ np.diag(master.gwgh * xxi) @ shapet.T - shap @ np.diag(master.gwgh * xet) @ shapxi.T)
    M33 = -(shapxi @ np.diag(master.gwgh * yet) @ shap.T)*c[0] + (shapet @ np.diag(master.gwgh * yxi) @ shap.T)*c[0] + (shapxi @ np.diag(master.gwgh * xet) @ shap.T)*c[1] - (shapet @ np.diag(master.gwgh * xxi) @ shap.T)*c[1] # [nPoly X nPoly]
    M34 = np.zeros([npl,nps*3]) # [nPoly X 3nps]   

    M41 = np.zeros([nps*3,npl]) # [3nps X nPoly]
    M42 = np.zeros([nps*3,npl]) # [3nps X nPoly]
    M43 = np.zeros([nps*3,npl]) # [3nps X nPoly]
    M44 = np.zeros([nps*3,nps*3]) # [3nps X 3nps]

    M51 = np.zeros([1,npl]) # [1 X nPoly]
    M52 = np.zeros([1,npl]) # [1 X nPoly]
    M53 = np.zeros([1,npl]) # [1 X nPoly]
    M53[0,:] = (master.gwgh * jac) @ shap.T # [1X nPoly]
    M54 = np.zeros([1,nps*3]) # [1 X 3nps]
    
    for s in range(3):
        xxi = np.squeeze(master.sh1d[:,1,:]).T @ dg[perm[:,s], 0] #dxdxi1d [nQuad X 1]
        yxi = np.squeeze(master.sh1d[:,1,:]).T @ dg[perm[:,s], 1] #dydxi1d [nQuad X 1]
        dsdxi = np.sqrt(xxi**2 + yxi**2) #det(I) [nQuad X 1]
        nl = np.column_stack((yxi/dsdxi, -xxi/dsdxi)) #nx & ny [nQuad X 2]
        cnl = c[0]*nl[:,0] + c[1]*nl[:,1] #cx*nx + cy*ny [nQuad X 1]
    
        tauc = np.abs(cnl) #abs(cx*nx + cy*ny) [nQuad X 1]
        tau  = taud + tauc #tau [nQuad X 1]

        idx1dSav = np.arange(nps*s,nps*s+nps)
        M14[np.ix_(perm[:,s],idx1dSav)] +=  sh1d @ np.diag(master.gw1d * yxi) @ sh1d.T
        M24[np.ix_(perm[:,s],idx1dSav)] += -sh1d @ np.diag(master.gw1d * xxi) @ sh1d.T

        M31[np.ix_(perm[:,s],perm[:,s])] += sh1d @ np.diag(dsdxi*master.gw1d*nl[:,0]) @ sh1d.T 
        M32[np.ix_(perm[:,s],perm[:,s])] += sh1d @ np.diag(dsdxi*master.gw1d*nl[:,1]) @ sh1d.T
        M33[np.ix_(perm[:,s],perm[:,s])] += sh1d @ np.diag(dsdxi*master.gw1d*tau) @ sh1d.T 
        M34[np.ix_(perm[:,s],idx1dSav)] += sh1d @ np.diag(dsdxi*master.gw1d*(cnl-tau)) @ sh1d.T

        M41[np.ix_(idx1dSav,perm[:,s])] += sh1d @ np.diag(dsdxi*master.gw1d*nl[:,0]) @ sh1d.T
        M42[np.ix_(idx1dSav,perm[:,s])] += sh1d @ np.diag(dsdxi*master.gw1d*nl[:,1]) @ sh1d.T
        M43[np.ix_(idx1dSav,perm[:,s])] += sh1d @ np.diag(dsdxi*master.gw1d*tau) @ sh1d.T 
        M44[np.ix_(idx1dSav,idx1dSav)] += sh1d @ np.diag(dsdxi*master.gw1d*(cnl-tau)) @ sh1d.T

    LHS = np.vstack([np.hstack([M11,M12,M13,M14]),np.hstack([M21,M22,M23,M24]),np.hstack([M31,M32,M33,M34]),np.hstack([M41,M42,M43,M44]),np.hstack([M51,M52,M53,M54])])

    return LHS

"""

def hdg_postprocess(master, mesh, master1, mesh1, uh, qh):
    """
    hdg_postprocess postprocesses the hdg solution to obtain a better solution.
 
       master:       master structure of porder
       mesh:         mesh structure of porder
       master1:      master structure of porder+1
       mesh1:        mesh structure of porder+1
       uh:           approximate scalar variable
       qh:           approximate flux
       ustarh:       postprocessed scalar variable
    """
    #Forgot that in the code qh = -k*grad(U) the diffusion eqn postprocess is qh=k*grad(uh)
    qh = -qh #In-production fix

    ustarh = np.zeros((mesh1.dgnodes.shape[0], mesh1.t.shape[0]))

    shapTrans1_0 = shape2d(master1.porder, master1.plocal, master.gpts) #[nPoly(k+1)xnQuad(k)]

    nt  = mesh1.t.shape[0] # number of triangle
    shap0     = master.shap[:,0,:]   #[nPoly(k) X nQuad(k)]
    shap1     = master1.shap[:,0,:]  #[nPoly(k+1) X nQuad(k+1)]
    shapxi0   = master.shap[:,1,:]   #[nPoly(k) X nQuad(k)]
    shapxi1_0 = shapTrans1_0[:,1,:]     #[nPoly(k+1) X nQuad(k)]
    shapxi1   = master1.shap[:,1,:]  #[nPoly(k+1) X nQuad(k+1)]
    shapet0   = master.shap[:,2,:]   #[nPoly(k) X nQuad(k)]
    shapet1_0 = shapTrans1_0[:,2,:]     #[nPoly(k+1) X nQuad(k)]
    shapet1   = master1.shap[:,2,:]  #[nPoly(k+1) X nQuad(k+1)]


    #Alternative
    shapTrans0_1 = shape2d(master.porder, master.plocal, master1.gpts) #[nPoly(k)xnQuad(k+1)]
    shap0_1      = shapTrans0_1[:,0,:] #[nPoly(k) X nQuad(k+1)]
    #Alternative Over

    for i in range(nt):
        dgCur1 = mesh1.dgnodes[:,:,i]      #[nQuad(k+1) X 2]
        dgCur0 = mesh.dgnodes[:,:,i]       #[nQuad(k) X 2]     
        uhCur0 = uh[:,i]                   #[nQuad(k) X 1]
        qhxCur0 = qh[:,0,i]                #[nQuad(k) X 1]
        qhyCur0 = qh[:,1,i]                #[nQuad(k) X 1]

        xxiCur0 = dgCur0[:,0] @ shapxi0     #dxdxi [nQuad(k+1) X 1]
        xetCur0 = dgCur0[:,0] @ shapet0     #dxdet [nQuad(k+1) X 1]
        yxiCur0 = dgCur0[:,1] @ shapxi0     #dydxi [nQuad(k+1) X 1]
        yetCur0 = dgCur0[:,1] @ shapet0     #dydet [nQuad(k+1) X 1]
        jacCur0 = xxiCur0 * yetCur0 - xetCur0 * yxiCur0   #Det(Jac) [nQuad(k) X 1]

        xxiCur1 = dgCur1[:,0] @ shapxi1     #dxdxi [nQuad(k+1) X 1]
        xetCur1 = dgCur1[:,0] @ shapet1     #dxdet [nQuad(k+1) X 1]
        yxiCur1 = dgCur1[:,1] @ shapxi1     #dydxi [nQuad(k+1) X 1]
        yetCur1 = dgCur1[:,1] @ shapet1     #dydet [nQuad(k+1) X 1]
        jacCur1 = xxiCur1 * yetCur1 - xetCur1 * yxiCur1   #Det(Jac) [nQuad(k+1) X 1]
        xixCur1 = yetCur1/jacCur1           #dxidx [nQuad(k+1) X 1]
        etxCur1 = -yxiCur1/jacCur1          #detdx [nQuad(k+1) X 1]
        xiyCur1 = -xetCur1/jacCur1          #dxidy [nQuad(k+1) X 1]
        etyCur1 = xxiCur1/jacCur1           #detdy [nQuad(k+1) X 1]

        LHS  = (shapxi1 @ np.diag(xixCur1*yetCur1*master1.gwgh) @ shapxi1.T)
        LHS -= (shapet1 @ np.diag(etxCur1*yxiCur1*master1.gwgh) @ shapet1.T)
        LHS -= (shapet1 @ np.diag(xixCur1*yxiCur1*master1.gwgh) @ shapxi1.T)
        LHS += (shapxi1 @ np.diag(etxCur1*yetCur1*master1.gwgh) @ shapet1.T)

        LHS += (shapet1 @ np.diag(etyCur1*xxiCur1*master1.gwgh) @ shapet1.T)
        LHS -= (shapxi1 @ np.diag(xiyCur1*xetCur1*master1.gwgh) @ shapxi1.T) #[nPoly(k+1) x nPoly(k+1)]
        LHS += (shapet1 @ np.diag(xiyCur1*xxiCur1*master1.gwgh) @ shapxi1.T)
        LHS -= (shapxi1 @ np.diag(etyCur1*xetCur1*master1.gwgh) @ shapet1.T)

        LHS_C = (master1.gwgh*jacCur1) @ shap1.T #[1 X nPoly(k+1)]

        RHS  = (shapxi1_0 @ np.diag(master.gwgh*yetCur0) @ shap0.T @ qhxCur0) #[nPoly(k+1) x 1]
        RHS -= (shapet1_0 @ np.diag(master.gwgh*yxiCur0) @ shap0.T @ qhxCur0) #[nPoly(k+1) x 1]
        RHS -= (shapxi1_0 @ np.diag(master.gwgh*xetCur0) @ shap0.T @ qhyCur0) #[nPoly(k+1) x 1]
        RHS += (shapet1_0 @ np.diag(master.gwgh*xxiCur0) @ shap0.T @ qhyCur0) #[nPoly(k+1) x 1]

        RHS_C = (master.gwgh*jacCur0) @ shap0.T @ uhCur0 #[1X1]

        #Alternative
        RHS  = (shapxi1 @ np.diag(master1.gwgh*yetCur1) @ shap0_1.T @ qhxCur0) #[nPoly(k+1) x 1]
        RHS -= (shapet1 @ np.diag(master1.gwgh*yxiCur1) @ shap0_1.T @ qhxCur0) #[nPoly(k+1) x 1]
        RHS -= (shapxi1 @ np.diag(master1.gwgh*xetCur1) @ shap0_1.T @ qhyCur0) #[nPoly(k+1) x 1]
        RHS += (shapet1 @ np.diag(master1.gwgh*xxiCur1) @ shap0_1.T @ qhyCur0) #[nPoly(k+1) x 1]

        # RHS_C = (master1.gwgh*jacCur1) @ shap0_1.T @ uhCur0 #[1X1]
        #Alternative Over


        LHS[-1,:] = LHS_C
        RHS[-1] = RHS_C

        ustarh[:,i] = np.linalg.solve(LHS,RHS)

    return ustarh