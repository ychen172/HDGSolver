import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from master import shape2d

__all__ = ['hdg_solve']

def localprob(dg, master, m, source, param, taudLst, uhPrev=None):
    """
    localprob solves the local convection-diffusion problems for the hdg method
       dg:              dg nodes
       master:          master element structure
       m[3*nps, ncol]:  values of uhat at the element edges (ncol is the
                        number of right hand sides. i.e a different local 
                        problem is solved for each right hand side)
       source:          source term
       param:           param['kappa']= diffusivity coefficient
                        param['c'] = convective velocity
       umf[npl, ncol]:  uh local solution
       qmf[npl,2,ncol]: qh local solution
    """
    porder = master.porder

    kappa = param['kappa']
    c     = param['c']
    
    if 'dT' in param:
        dT = param['dT'] #Time dependent problem

    nps   = porder+1 #nPoly1d
    ncol  = m.shape[1] #Used to output a eigen matrix when input is an indentity matrix 
    npl   = dg.shape[0] #nPoly

    perm = master.perm[:,:,0] #nPoly1d X 3 corners CCW

    qmf = np.zeros((npl, 2, ncol)) #This is qx and qy for each column input: Each is nPoly X nColn

    Fx = np.zeros((npl, ncol))
    Fy = np.zeros((npl, ncol))
    Fu = np.zeros((npl, ncol))

    # Create Nx Ny K L
    Nx = np.zeros((nps*3,npl))
    Ny = Nx * 0.0
    K  = Nx * 0.0
    L  = np.zeros((nps*3,nps*3))

    # Volume integral
    shap = master.shap[:,0,:]                    #nPoly X nQuad
    shapxi = master.shap[:,1,:]                  #nPoly X nQuad
    shapet = master.shap[:,2,:]                  #nPoly X nQuad
    shapxig = shapxi @ np.diag(master.gwgh)      #shxi @ diag(wq) [nPoly X nQuad]
    shapetg = shapet @ np.diag(master.gwgh)      #shet @ diag(wq) [nPoly X nQuad]

    xxi = dg[:,0] @ shapxi     #dxdxi [nQuad X 1]
    xet = dg[:,0] @ shapet     #dxdet [nQuad X 1]
    yxi = dg[:,1] @ shapxi     #dydxi [nQuad X 1]
    yet = dg[:,1] @ shapet     #dydet [nQuad X 1]
    jac = xxi * yet - xet * yxi #Det(Jac) [nQuad X 1]
    shapx =   shapxig @ np.diag(yet) - shapetg @ np.diag(yxi) #   shxi @ diag(wq) @ diag(yet) - shet @ diag(wq) @ diag(yxi) [nPoly X nQuad]
    shapy = - shapxig @ np.diag(xet) + shapetg @ np.diag(xxi) # - shxi @ diag(wq) @ diag(xet) + shet @ diag(wq) @ diag(xxi) [nPoly X nQuad]
    M  = (shap @ np.diag(master.gwgh * jac) @ shap.T)/kappa   # shap @ np.diag(wq * det(Jac) / kappa) @ shap.T [nPoly X nPoly]  => Ax & Ay 
    Cx = shap @ shapx.T # Cx.T =>   shxi @ diag(wq) @ diag(yet) @ shap.T - shet @ diag(wq) @ diag(yxi) @ shap.T [nPoly X nPoly] => -Bx 
    Cy = shap @ shapy.T # Cy.T => - shxi @ diag(wq) @ diag(xet) @ shap.T + shet @ diag(wq) @ diag(xxi) @ shap.T [nPoly X nPoly] => -By 
                        # Cx   =>   shap @ diag(wq) @ diag(yet) @ shxi.T - shap @ diag(wq) @ diag(yxi) @ shet.T [nPoly X nPoly] => Dxk
                        # Cy   => - shap @ diag(wq) @ diag(xet) @ shxi.T + shap @ diag(wq) @ diag(xxi) @ shet.T [nPoly X nPoly] => Dyk

    D = -c[0]*Cx.T - c[1]*Cy.T #   shxi @ diag(wq) @ diag(yet) @ shap.T * (-cx) 
                               # - shet @ diag(wq) @ diag(yxi) @ shap.T * (-cx)
                               # - shxi @ diag(wq) @ diag(xet) @ shap.T * (-cy)
                               #   shet @ diag(wq) @ diag(xxi) @ shap.T * (-cy) [nPoly X nPoly] => Ek 
    if 'dT' in param:
        D += shap @ np.diag(jac * master.gwgh * (1/dT)) @ shap.T # [nPoly X nPoly] => Ek with time dependent

    if source:
        pg = shap.T @ dg #[nQuad X 2] (x and y coordinates)
        src = source( pg) # shap.T @ fmj [nQuad X 1]
        Fu[:,0] = shap @ np.diag(master.gwgh*jac) @ src # shap @ diag(wq) @ diag(det(Jac)) @ shap.T @ fmj [nPoly X 1] => Fk
        if 'dT' in param:
            Fu[:,0] += shap @ np.diag(jac * master.gwgh * (1/dT)) @ shap.T @ uhPrev #[nPoly X 1]

    sh1d = np.squeeze(master.sh1d[:,0,:]) # sh1d [nPoly X nQuad]
    for s in range(3):
        xxi = np.squeeze(master.sh1d[:,1,:]).T @ dg[perm[:,s], 0] #dxdxi1d [nQuad X 1]
        yxi = np.squeeze(master.sh1d[:,1,:]).T @ dg[perm[:,s], 1] #dydxi1d [nQuad X 1]
        dsdxi = np.sqrt(xxi**2 + yxi**2) #det(I) [nQuad X 1]
        nl = np.column_stack((yxi/dsdxi, -xxi/dsdxi)) #nx & ny [nQuad X 2]
        
        cnl = c[0]*nl[:,0] + c[1]*nl[:,1] #cx*nx + cy*ny [nQuad X 1]
    
        tauc = np.abs(cnl) #abs(cx*nx + cy*ny) [nQuad X 1]
        tau  = taudLst[s] + tauc #tau [nQuad X 1]

        D[np.ix_(perm[:,s], perm[:,s])] = D[np.ix_(perm[:,s], perm[:,s])] + sh1d @ np.diag(master.gw1d*dsdxi*tau) @ sh1d.T # sh1d @ diag(det(I)*wq*tau) @ sh1d.T [nPoly1d X nPoly1d]=> Edk
    
    for s in range(3):
        xxi = np.squeeze(master.sh1d[:,1,:]).T @ dg[perm[:,s], 0] #dxdxi1d [nQuad X 1]
        yxi = np.squeeze(master.sh1d[:,1,:]).T @ dg[perm[:,s], 1] #dydxi1d [nQuad X 1]
        dsdxi = np.sqrt(xxi**2 + yxi**2) #det(I) [nQuad X 1]
        nl = np.column_stack((yxi/dsdxi, -xxi/dsdxi)) #nx & ny [nQuad X 2]
       
        for icol in range(ncol):     # Loop over all the right-hand-sides
            ml = m[s*nps:(s+1)*nps, icol] # U_hat vector for one edge and one column [nPoly1d X 1]
       
            cnl = c[0]*nl[:,0] + c[1]*nl[:,1] #cx*nx + cy*ny [nQuad X 1]
    
            tauc = np.abs(cnl) #abs(cx*nx + cy*ny) [nQuad X 1]
            tau  = taudLst[s] + tauc #tau [nQuad X 1]
    
            Fx[perm[:,s], icol] = Fx[perm[:,s], icol] - sh1d @ np.diag(master.gw1d*dsdxi*nl[:,0]) @ sh1d.T @ ml # 0 - Cxdk @ Uhat [nPoly1d X 1]
            Fy[perm[:,s], icol] = Fy[perm[:,s], icol] - sh1d @ np.diag(master.gw1d*dsdxi*nl[:,1]) @ sh1d.T @ ml # 0 - Cydk @ Uhat [nPoly1d X 1]
       
            Fu[perm[:,s], icol] = Fu[perm[:,s], icol] - sh1d @ np.diag(master.gw1d*dsdxi*(cnl-tau)) @ sh1d.T @ ml #Fk - Idk @ Uhat [nPoly1d X 1]
        
        idx1dSav = np.arange(nps*s,nps*s+nps)
        Nx[np.ix_(idx1dSav, perm[:,s])] = sh1d @ np.diag(master.gw1d*dsdxi*nl[:,0]) @ sh1d.T #[3*nPoly1d, nPoly]
        Ny[np.ix_(idx1dSav, perm[:,s])] = sh1d @ np.diag(master.gw1d*dsdxi*nl[:,1]) @ sh1d.T #[3*nPoly1d, nPoly]
        K[np.ix_(idx1dSav, perm[:,s])]  = sh1d @ np.diag(master.gw1d*dsdxi*tau) @ sh1d.T #[3*nPoly1d, nPoly]
        L[np.ix_(idx1dSav, idx1dSav)]   = sh1d @ np.diag(master.gw1d*dsdxi*(cnl-tau)) @ sh1d.T #[3*nPoly1d, 3*nPoly1d]

    M1Fx = np.linalg.solve(M, Fx) # inv(Ax) @ {0 - Cxdk @ Uhat}
    M1Fy = np.linalg.solve(M, Fy) # inv(Ay) @ {0 - Cydk @ Uhat}

    M1   = np.linalg.inv(M)       # inv(Ax) or inv(Ay)

    umf = np.linalg.solve(D + Cx @ M1 @ Cx.T + Cy @ M1 @ Cy.T, Fu - Cx @ M1Fx - Cy @ M1Fy) # {Edk - Dxk @ inv(Ax) @ Bx - Dyk @ inv(Ay) @ By} @ umf = {Fu - Dxk @ M1Fx - Dyk @ M1Fy}  --> U
    qmf[:,0,:] = M1Fx + np.linalg.solve(M, Cx.T @ umf) # M1Fx - inv(Ax) @ Bx @ U --> Qx
    qmf[:,1,:] = M1Fy + np.linalg.solve(M, Cy.T @ umf) # M1Fy - inv(Ay) @ By @ U --> Qy

    return umf, qmf, Nx, Ny, K, L


def elemmat_hdg(dg, master, source, param, taudLst, uhPrev=None):
    """
    elemmat_hdg calcualtes the element and force vectors for the hdg method
 
       dg:              dg nodes
       master:          master element structure
       source:          source term
       param:           param['kappa']   = diffusivity coefficient
                        param['c'] = convective velocity
       ae[3*nps,3*nps]: element matrix (nps is nimber of points per edge)
       fe[3*nps,1]:     element forcer vector
    """
    nps   = master.porder + 1   #nPoly1d
    npl   = dg.shape[0]         #nPoly

    mu = np.identity(3*nps)
    m = np.zeros((3*nps,1))

    if 'dT' in param:
        um0, qm0, dummy1, dummy2, dummy3, dummy4 = localprob(dg, master, mu, None, param, taudLst, uhPrev=uhPrev) #um0: [nPoly X 3*nps] qm0: [nPoly X 2 X 3*nps] qx and qy
        u0f, q0f, Nx, Ny, K, L = localprob(dg, master, m, source, param, taudLst, uhPrev=uhPrev) #u0f: [nPoly X 1] q0f: [nPoly X 2 X 1] qx and qy
    else:
        um0, qm0, dummy1, dummy2, dummy3, dummy4 = localprob(dg, master, mu, None, param, taudLst) #um0: [nPoly X 3*nps] qm0: [nPoly X 2 X 3*nps] qx and qy
        u0f, q0f, Nx, Ny, K, L = localprob(dg, master, m, source, param, taudLst) #u0f: [nPoly X 1] q0f: [nPoly X 2 X 1] qx and qy


    m0 = np.zeros([3*npl,3*nps])
    Of = np.zeros([3*npl,1])
    NNK = np.zeros([3*nps,3*npl])

    m0[0:npl,:] = qm0[:,0,:]
    m0[npl:2*npl,:] = qm0[:,1,:]
    m0[2*npl:3*npl,:] = um0

    Of[0:npl,:] = q0f[:,0,:]
    Of[npl:2*npl,:] = q0f[:,1,:]
    Of[2*npl:3*npl,:] = u0f

    NNK[:,0:npl] = Nx
    NNK[:,npl:2*npl] = Ny 
    NNK[:,2*npl:3*npl] = K

    ae = L + NNK @ m0
    fe = -NNK @ Of

    fe = np.squeeze(fe)
    return ae, fe


def hdg_solve(master, mesh, source, dbc, param, taudInp, uhPrev=None):
    """
    hdg_solve solves the convection-diffusion equation using the hdg method.
    [uh,qh,uhath]=hdg_solve(mesh,master,source,dbc,param)
 
       master:       master structure
       mesh:         mesh structure
       source:       source term
       dbc:          dirichlet data 
       param:        param['kappa']   = diffusivity coefficient
                     param['c'] = convective velocity
       uh:           approximate scalar variable
       qh:           approximate flux
       uhath:        approximate trace
       taudInp:      input taud [0]: inner faces [1]: boundary faces                               
    """

    nps = mesh.porder + 1
    npl = mesh.dgnodes.shape[0]
    nt  = mesh.t.shape[0]
    nf  = mesh.f.shape[0]

    ae  = np.zeros((3*nps, 3*nps, nt))
    fe  = np.zeros((3*nps, nt))

    for i in range(nt):
        taudLst = np.zeros(3) #Dicated taud
        for j in range(3):
            numFace = abs(mesh.t2f[i,j])-1 #face number
            if mesh.f[numFace,3] < -0.5: #boundary
                taudLst[j] = taudInp[1]
            else:
                taudLst[j] = taudInp[0]
        if 'dT' in param:
            ae[:,:,i], fe[:,i] = elemmat_hdg( mesh.dgnodes[:,:,i], master, source, param, taudLst, uhPrev = uhPrev[:,i])
        else:
            ae[:,:,i], fe[:,i] = elemmat_hdg( mesh.dgnodes[:,:,i], master, source, param, taudLst)

    #Create Connectivity Matrix
    connect = np.ones([3*nps,nt])*1e10
    connect = connect.astype('int')
    for i in range(nt):
        for j in range(3):
            fNum = mesh.t2f[i,j]
            if fNum > 0: # CCW
                fNum -= 1
                connect[j*nps:j*nps+nps,i] = np.arange(fNum*nps,fNum*nps+nps)
            else: #CW
                fNum = -fNum - 1
                connect[j*nps:j*nps+nps,i] = np.flip(np.arange(fNum*nps,fNum*nps+nps))

    #Assemble Matrix
    H = np.zeros([nf*nps,nf*nps])
    G = np.zeros([nf*nps,1])
    for i in range(nt): #Loop through each element
        idxMap = connect[:,i]
        H[np.ix_(idxMap, idxMap)] += ae[:,:,i]
        G[idxMap,0] += fe[:,i]

    #Implement boundary condition
    for i in range(nf): #Loop through each face
        if mesh.f[i,3] < -0.5: #Boundary
            ipt  = np.sum(mesh.f[i,:2]) #Sum of the global indices of the two nodes of the face
            el  = mesh.f[i,2] #Owner element global index
            ipl = np.sum(mesh.t[el,:])-ipt #global index of the node opposite to that face
            isl = np.where(mesh.t[el,:] == ipl)[0][0] #local index of the node opposite to that face 0, 1, or 2
            dgCur =  mesh.dgnodes[master.perm[:,isl,0],:,el]
            bound = np.squeeze(dbc(dgCur))
            H[i*nps:i*nps+nps, :] = 0
            H[i*nps:i*nps+nps,i*nps:i*nps+nps] = np.identity(nps)
            G[i*nps:i*nps+nps,0] = bound

    uhath = np.linalg.solve(H,G)
    
    uh    = np.zeros((npl, nt))
    qh    = np.zeros((npl, 2, nt))
    for i in range(nt):
        uhathLoc = np.zeros([3*nps,1]) 
        for j in range(3):
            fNum = mesh.t2f[i,j]
            if fNum>0:
                fNum = fNum-1
                idxExt = np.arange(fNum*nps,fNum*nps+nps) #Extraction Indices 
            else:
                fNum = -fNum-1
                idxExt = np.flip(np.arange(fNum*nps,fNum*nps+nps)) #Extraction Indices
            uhathLoc[j*nps:j*nps+nps,:] =  uhath[idxExt]
        #Prepare tau_diff list
        taudLst = np.zeros(3) #Dicated taud
        for j in range(3):
            numFace = abs(mesh.t2f[i,j])-1 #face number
            if mesh.f[numFace,3] < -0.5: #boundary
                taudLst[j] = taudInp[1]
            else:
                taudLst[j] = taudInp[0]
        if 'dT' in param:
            uhCur, qhCur, dummy1, dummy2, dummy3, dummy4 = localprob(mesh.dgnodes[:,:,i], master, uhathLoc, source, param, taudLst, uhPrev = uhPrev[:,i]) #u: [nPoly X 1] q: [nPoly X 2 X 1] qx and qy
        else:
            uhCur, qhCur, dummy1, dummy2, dummy3, dummy4 = localprob(mesh.dgnodes[:,:,i], master, uhathLoc, source, param, taudLst)
        uh[:,i] = np.squeeze(uhCur)
        qh[:,:,i] = np.squeeze(qhCur)
    return uh, qh, uhath