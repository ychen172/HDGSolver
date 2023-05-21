import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from master import shape2d

__all__ = ['hdg_nonlinear_solve']

def residualUpd(BMat, dg, master, param, taudLst, uhTest=None, uhathTest=None):
    """
    localprob solves the local convection-diffusion problems for the hdg method
       dg:              dg nodes
       master:          master element structure
       param:           param['kappa']= diffusivity coefficient
                        param['c'] = convective velocity
    """
    porder = master.porder

    #For Residual Modification
    funCv = param['funCv']
    funCvX = funCv[0] #0.5*u**2
    funCvY = funCv[1] #0.5*u**2
    funCvXVal = funCvX(uhTest)
    funCvYVal = funCvY(uhTest) # [npl]
    funCvXHatVal = funCvX(uhathTest)
    funCvYHatVal = funCvY(uhathTest) # [nps*3]

    #For Jacobian Modification
    dFunCv  = param['dFunCv']
    dFunCvX = dFunCv[0] #u
    dFunCvY = dFunCv[1] #u
    dFunCvXVal = dFunCvX(uhTest) #dfx/du
    dFunCvYVal = dFunCvY(uhTest) #dfy/du [npl]
    dFunCvXHatVal = dFunCvX(uhathTest) #dfx/du_hat
    dFunCvYHatVal = dFunCvY(uhathTest) #dfy/du_hat [nps*3]
    
    nps   = porder+1 #nPoly1d
    npl   = dg.shape[0] #nPoly
    perm = master.perm[:,:,0] #nPoly1d X 3 corners CCW

    # Initialize L and I matrices For Residual Modification
    L  = np.zeros((nps*3,nps*3))
    IMatri = np.zeros((npl,nps*3)) 

    # Initialize L and I matrices For Jacobian Modification
    dL  = np.zeros((nps*3,nps*3))
    dIMatri = np.zeros((npl,nps*3)) 

    # Volume Integral
    shap = master.shap[:,0,:]                    #nPoly X nQuad
    shapxi = master.shap[:,1,:]                  #nPoly X nQuad
    shapet = master.shap[:,2,:]                  #nPoly X nQuad
    shapxig = shapxi @ np.diag(master.gwgh)      #shxi @ diag(wq) [nPoly X nQuad]
    shapetg = shapet @ np.diag(master.gwgh)      #shet @ diag(wq) [nPoly X nQuad]

    xxi = dg[:,0] @ shapxi     #dxdxi [nQuad X 1]
    xet = dg[:,0] @ shapet     #dxdet [nQuad X 1]
    yxi = dg[:,1] @ shapxi     #dydxi [nQuad X 1]
    yet = dg[:,1] @ shapet     #dydet [nQuad X 1]
    shapx =   shapxig @ np.diag(yet) - shapetg @ np.diag(yxi) #   shxi @ diag(wq) @ diag(yet) - shet @ diag(wq) @ diag(yxi) [nPoly X nQuad]
    shapy = - shapxig @ np.diag(xet) + shapetg @ np.diag(xxi) # - shxi @ diag(wq) @ diag(xet) + shet @ diag(wq) @ diag(xxi) [nPoly X nQuad]
    sh1d  = np.squeeze(master.sh1d[:,0,:]) # sh1d [nPoly X nQuad]
    Cx = shap @ shapx.T # Cx.T =>   shxi @ diag(wq) @ diag(yet) @ shap.T - shet @ diag(wq) @ diag(yxi) @ shap.T [nPoly X nPoly] => -Bx 
    Cy = shap @ shapy.T # Cy.T => - shxi @ diag(wq) @ diag(xet) @ shap.T + shet @ diag(wq) @ diag(xxi) @ shap.T [nPoly X nPoly] => -By 
                        # Cx   =>   shap @ diag(wq) @ diag(yet) @ shxi.T - shap @ diag(wq) @ diag(yxi) @ shet.T [nPoly X nPoly] => Dxk
                        # Cy   => - shap @ diag(wq) @ diag(xet) @ shxi.T + shap @ diag(wq) @ diag(xxi) @ shet.T [nPoly X nPoly] => Dyk

    #For Residual Modification
    D = - Cx.T @ np.diag(funCvXVal) - Cy.T @ np.diag(funCvYVal) #   shxi @ diag(wq) @ diag(yet) @ shap.T * (-fx) 
                                                                # - shet @ diag(wq) @ diag(yxi) @ shap.T * (-fx)
                                                                # - shxi @ diag(wq) @ diag(xet) @ shap.T * (-fy)
                                                                #   shet @ diag(wq) @ diag(xxi) @ shap.T * (-fy) [nPoly X nPoly] => Ek (change 1)
    
    #For Jacobian Modification
    dD = - Cx.T @ np.diag(dFunCvXVal) - Cy.T @ np.diag(dFunCvYVal)            #   shxi @ diag(wq) @ diag(yet) @ shap.T * (-dfx/du) 
                                                                              # - shet @ diag(wq) @ diag(yxi) @ shap.T * (-dfx/du)
                                                                              # - shxi @ diag(wq) @ diag(xet) @ shap.T * (-dfy/du)
                                                                              #   shet @ diag(wq) @ diag(xxi) @ shap.T * (-dfy/du) [nPoly X nPoly] => Ek (change 1)
    
    # Surface Integral    
    for s in range(3):
        xxi = np.squeeze(master.sh1d[:,1,:]).T @ dg[perm[:,s], 0] #dxdxi1d [nQuad X 1]
        yxi = np.squeeze(master.sh1d[:,1,:]).T @ dg[perm[:,s], 1] #dydxi1d [nQuad X 1]
        dsdxi = np.sqrt(xxi**2 + yxi**2) #det(I) [nQuad X 1]
        nl = np.column_stack((yxi/dsdxi, -xxi/dsdxi)) #nx & ny [nQuad X 2]
        tau  = taudLst[s] #tau [1]
        idx1dSav = np.arange(nps*s,nps*s+nps)

        #For Residual Modification
        funCvXHatValCurQ = sh1d.T @ funCvXHatVal[idx1dSav] # fxhat [Quad1d X 1]
        funCvYHatValCurQ = sh1d.T @ funCvYHatVal[idx1dSav] # fyhat [Quad1d X 1]
        uhathTestQ = sh1d.T @ uhathTest[idx1dSav] # uhat [Quad1d X 1]
        IMatri[np.ix_(perm[:,s], idx1dSav)]  += sh1d @ np.diag(master.gw1d*dsdxi*(funCvXHatValCurQ*nl[:,0] + funCvYHatValCurQ*nl[:,1] - tau*uhathTestQ)) @ sh1d.T # sh1d @ diag(|I|*wq*(fxhat*nx+fyhat*ny-tau*uhat)) @ sh1d.T => Idk [npl * nps*3] Change 2
        L[np.ix_(idx1dSav, idx1dSav)]        += sh1d @ np.diag(master.gw1d*dsdxi*(funCvXHatValCurQ*nl[:,0] + funCvYHatValCurQ*nl[:,1] - tau*uhathTestQ)) @ sh1d.T # Idk => Ldk [3*nPoly1d, 3*nPoly1d] Change 3

        #For Jacobian Modification
        dFunCvXHatValCurQ = sh1d.T @ dFunCvXHatVal[idx1dSav] # dfxhat/duhat [Quad1d X 1]
        dFunCvYHatValCurQ = sh1d.T @ dFunCvYHatVal[idx1dSav] # dfyhat/duhat [Quad1d X 1]
        dIMatri[np.ix_(perm[:,s], idx1dSav)] += sh1d @ np.diag(master.gw1d*dsdxi*(dFunCvXHatValCurQ*nl[:,0] + dFunCvYHatValCurQ*nl[:,1] - tau)) @ sh1d.T # sh1d @ diag(|I|*wq*(dfxhat/duhat*nx+dfyhat/duhat*ny-tau)) @ sh1d.T => Idk [npl * nps*3] Change 2
        dL[np.ix_(idx1dSav, idx1dSav)]       += sh1d @ np.diag(master.gw1d*dsdxi*(dFunCvXHatValCurQ*nl[:,0] + dFunCvYHatValCurQ*nl[:,1] - tau)) @ sh1d.T # Idk => Ldk [3*nPoly1d, 3*nPoly1d] Change 3

    #Turn the base matrix into residual matrix
    RMat = BMat*0.0 # Create a zero copy
    RMat[2*npl:3*npl,2*npl:3*npl] += D
    RMat[2*npl:3*npl,3*npl:] += IMatri
    RMat[3*npl:,3*npl:] += L

    #Turn the base matrix into jacobian matrix
    JMat = BMat*1.0 # Create a copy
    JMat[2*npl:3*npl,2*npl:3*npl] += dD
    JMat[2*npl:3*npl,3*npl:] += dIMatri
    JMat[3*npl:,3*npl:] += dL

    return RMat, JMat

def sourceIni(dg, master, source, param, uhPrev=None):
    """
    localprob solves the local convection-diffusion problems for the hdg method
       dg:              dg nodes
       master:          master element structure
       param:           param['kappa']= diffusivity coefficient
                        param['c'] = convective velocity
    """
    porder = master.porder
  
    if 'dT' in param:
        dT = param['dT'] #Time dependent problem

    nps   = porder+1 #nPoly1d
    npl   = dg.shape[0] #nPoly

    # Create Forcing terms
    Fqx = np.zeros((npl, 1))
    Fqy = np.zeros((npl, 1))
    Fuh = np.zeros((npl, 1))
    Fuhath = np.zeros((nps*3,1))

    # Volume integral
    shap = master.shap[:,0,:]                    #nPoly X nQuad
    shapxi = master.shap[:,1,:]                  #nPoly X nQuad
    shapet = master.shap[:,2,:]                  #nPoly X nQuad

    xxi = dg[:,0] @ shapxi     #dxdxi [nQuad X 1]
    xet = dg[:,0] @ shapet     #dxdet [nQuad X 1]
    yxi = dg[:,1] @ shapxi     #dydxi [nQuad X 1]
    yet = dg[:,1] @ shapet     #dydet [nQuad X 1]
    jac = xxi * yet - xet * yxi #Det(Jac) [nQuad X 1]

    # Calculate Forcing Term
    if source:
        pg = shap.T @ dg #[nQuad X 2] (x and y coordinates)
        src = source( pg) # shap.T @ fmj [nQuad X 1]
        Fuh[:,0] = shap @ np.diag(master.gwgh*jac) @ src # shap @ diag(wq) @ diag(det(Jac)) @ shap.T @ fmj [nPoly X 1] => Fk
        if 'dT' in param:
            Fuh[:,0] += shap @ np.diag(jac * master.gwgh * (1/dT)) @ shap.T @ uhPrev #[nPoly X 1]

    FVec = np.vstack([Fqx,Fqy,Fuh,Fuhath])
    return np.squeeze(FVec)

def baseMatIni(dg, master, param, taudLst):
    """
    localprob solves the local convection-diffusion problems for the hdg method
       dg:              dg nodes
       master:          master element structure
       param:           param['kappa']= diffusivity coefficient
                        param['c'] = convective velocity
    """
    porder = master.porder

    kappa = param['kappa']
    
    if 'dT' in param:
        dT = param['dT'] #Time dependent problem

    nps   = porder+1 #nPoly1d
    npl   = dg.shape[0] #nPoly
    perm = master.perm[:,:,0] #nPoly1d X 3 corners CCW

    # Create Nx Ny K L
    Nx = np.zeros((nps*3,npl))
    Ny = Nx * 0.0
    K  = Nx * 0.0
    L  = np.zeros((nps*3,nps*3))
    dummy = np.zeros((npl,npl)) #Placeholder zero matrix

    # Initialize I matrix
    D       = np.zeros((npl,npl)) #[nPoly X nPoly] => Ek
    IMatri  = np.zeros((npl,nps*3))
    CxMatri = np.zeros((npl,nps*3))
    CyMatri = np.zeros((npl,nps*3))  

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

    if 'dT' in param:
        D += shap @ np.diag(jac * master.gwgh * (1/dT)) @ shap.T # [nPoly X nPoly] => Ek with time dependent
    
    sh1d = np.squeeze(master.sh1d[:,0,:]) # sh1d [nPoly X nQuad]
    for s in range(3):
        xxi = np.squeeze(master.sh1d[:,1,:]).T @ dg[perm[:,s], 0] #dxdxi1d [nQuad X 1]
        yxi = np.squeeze(master.sh1d[:,1,:]).T @ dg[perm[:,s], 1] #dydxi1d [nQuad X 1]
        dsdxi = np.sqrt(xxi**2 + yxi**2) #det(I) [nQuad X 1]
        nl = np.column_stack((yxi/dsdxi, -xxi/dsdxi)) #nx & ny [nQuad X 2]
        tau  = taudLst[s] #tau [nQuad X 1]
        idx1dSav = np.arange(nps*s,nps*s+nps)

        D[np.ix_(perm[:,s], perm[:,s])]      += sh1d @ np.diag(master.gw1d*dsdxi*tau) @ sh1d.T # sh1d @ diag(det(I)*wq*tau) @ sh1d.T [nPoly1d X nPoly1d]=> Edk
        CxMatri[np.ix_(perm[:,s], idx1dSav)] += sh1d @ np.diag(master.gw1d*dsdxi*nl[:,0]) @ sh1d.T # Cxdk [npl * nps*3]
        CyMatri[np.ix_(perm[:,s], idx1dSav)] += sh1d @ np.diag(master.gw1d*dsdxi*nl[:,1]) @ sh1d.T # Cydk [npl * nps*3]
        Nx[np.ix_(idx1dSav, perm[:,s])]      += sh1d @ np.diag(master.gw1d*dsdxi*nl[:,0]) @ sh1d.T #[3*nPoly1d, nPoly]
        Ny[np.ix_(idx1dSav, perm[:,s])]      += sh1d @ np.diag(master.gw1d*dsdxi*nl[:,1]) @ sh1d.T #[3*nPoly1d, nPoly]
        K[np.ix_(idx1dSav, perm[:,s])]       += sh1d @ np.diag(master.gw1d*dsdxi*tau) @ sh1d.T #[3*nPoly1d, nPoly]
        

    BMat = np.vstack([np.hstack([M,dummy,-Cx.T,CxMatri]),np.hstack([dummy,M,-Cy.T,CyMatri]),np.hstack([Cx,Cy,D,IMatri]),np.hstack([Nx,Ny,K,L])])
    return BMat

def elemmat_nonlinear_hdg(BMat, FVec, dg, master, param, taudLst, qhInter=None, uhInter=None, uhathInter=None):
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
    npl   = dg.shape[0]         #nPoly

    #Convert the base matrix into residual and jacobian matrices based on intermediate state
    RMat, JMat = residualUpd(BMat, dg, master, param, taudLst, uhTest=uhInter, uhathTest=uhathInter)
    
    #Intermediate solution vector
    UVecInter = np.hstack([qhInter[:,0],qhInter[:,1],uhInter,uhathInter])
    
    #Total forcing vector
    FVecT = FVec - (BMat @ UVecInter + RMat @ (1+(UVecInter*0)))
    
    #Decompose the matrix and the vector
    matABDE  = JMat[0:(3*npl),0:(3*npl)]
    matCI    = JMat[0:(3*npl),(3*npl):]
    matNK    = JMat[(3*npl):,0:(3*npl)]
    matL     = JMat[(3*npl):,(3*npl):] 
    FTqxqyuh = FVecT[0:(3*npl)]
    FTuhath  = FVecT[(3*npl):]
    
    #Inver ABDEmatrix
    matABDEInv = np.linalg.inv(matABDE) #High cost need to be done in each newton iteration
    matNK_ABDEInv = matNK @ matABDEInv
    
    #Write out H ang G matrix
    ae = matL - matNK_ABDEInv @ matCI
    fe = FTuhath - matNK_ABDEInv @ FTqxqyuh
    fe = np.squeeze(fe)
    
    return ae, fe, matABDEInv, FTqxqyuh, matCI

def back_Comp_qhuh(matABDEInv, FTqxqyuh, matCI, duhath):
    dqxqyuhVec = matABDEInv @ (FTqxqyuh - matCI @ duhath)
    npl = int(len(dqxqyuhVec)/3)
    dqh = np.zeros((npl,2))
    duh = np.zeros((npl))
    dqh[:,0] = dqxqyuhVec[0:npl]
    dqh[:,1] = dqxqyuhVec[npl:(2*npl)]
    duh[:] = dqxqyuhVec[(2*npl):]
    return dqh, duh

def hdg_nonlinear_solve(master, mesh, source, param, taudInp, BMatC, uhPrev=None, qhInter=None, uhInter=None, uhathInter=None):
    """                          
    """
    #As boundary conditions are set by initial conditions
    dbc    = lambda p: np.zeros((p.shape[0],1))
    #Maybe wrong but let's see

    nps = mesh.porder + 1
    npl = mesh.dgnodes.shape[0]
    nt  = mesh.t.shape[0]
    nf  = mesh.f.shape[0]

    ae  = np.zeros((3*nps, 3*nps, nt))
    fe  = np.zeros((3*nps, nt))
    matABDEInvC = np.zeros((3*npl,3*npl,nt))
    FTqxqyuhC = np.zeros((3*npl,nt))
    matCIC = np.zeros((3*npl,3*nps,nt))

    #And across all time steps, based matrix is only generated once
    #In each time step, forcing is generated only once
    FVecC = np.zeros(((npl+nps)*3,nt))
    taudLst = np.zeros((3,nt))
    for i in range(nt):
        #Compute forcing
        if 'dT' in param:
            FVecC[:,i] = sourceIni(mesh.dgnodes[:,:,i], master, source, param, uhPrev=uhPrev[:,i])
        else:
            FVecC[:,i] = sourceIni(mesh.dgnodes[:,:,i], master, source, param)
        #Assign stabilizer
        for j in range(3):
            numFace = abs(mesh.t2f[i,j])-1 #face number
            if mesh.f[numFace,3] < -0.5: #boundary
                taudLst[j,i] = taudInp[1]
            else:
                taudLst[j,i] = taudInp[0]

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

    #Start Newton inner loop
    #Define L2 norm np.linalg.norm(A,2)/np.sqrt(len(A))
    iniErr = 1 # initial error
    counter = 1
    relax = 1.0 #change is modified by this relaxation factor
    reduc = 0.7 #every time the error increases, reduce this relax factor by 10 percent if put 0.9
    while ((iniErr > 1e-14) & (counter <= 100)):
        # Adjust relaxation factor
        if (counter >= 2):
            if (iniErr > preErr): #If the current error is bigger than previous error
                relax = relax*reduc
        # Loop through each triangle to assemble H and G matrices
        for i in range(nt):
            ae[:,:,i], fe[:,i], matABDEInvC[:,:,i], FTqxqyuhC[:,i], matCIC[:,:,i] = elemmat_nonlinear_hdg(BMatC[:,:,i], FVecC[:,i], mesh.dgnodes[:,:,i], master, param, taudLst[:,i], qhInter=qhInter[:,:,i], uhInter=uhInter[:,i], uhathInter=uhathInter[:,i])

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
        #Solve for duhath
        duhath = np.linalg.solve(H,G)
        #Recollect duhath for each element
        duhathStack = np.zeros((nps*3,nt))
        for i in range(nt):
            for j in range(3):
                fNum = mesh.t2f[i,j]
                if fNum>0:
                    fNum = fNum-1
                    idxExt = np.arange(fNum*nps,fNum*nps+nps) #Extraction Indices 
                else:
                    fNum = -fNum-1
                    idxExt = np.flip(np.arange(fNum*nps,fNum*nps+nps)) #Extraction Indices
                duhathStack[j*nps:j*nps+nps,i] =  np.squeeze(duhath[idxExt])
        #Recalculate duh and dqh
        duh = np.zeros((npl, nt))
        dqh = np.zeros((npl, 2, nt))
        for i in range(nt):
            dqh[:,:,i], duh[:,i] = back_Comp_qhuh(matABDEInvC[:,:,i], FTqxqyuhC[:,i], matCIC[:,:,i], duhathStack[:,i])
        #Update all the intermediate solutions
        qhInter += relax*dqh
        uhInter += relax*duh
        uhathInter += relax*duhathStack
        #Store previous error
        preErr = iniErr
        #Update error
        iniErr = np.mean(abs(dqh)) + np.mean(abs(duh)) + np.mean(abs(duhathStack))
        #Print
        print('Counter: '+str(counter)+'\n')
        print('Errorqh: '+str(iniErr))
        #Update Counter
        counter += 1 
    return uhInter, qhInter, uhathInter