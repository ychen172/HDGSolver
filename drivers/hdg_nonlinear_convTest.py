# Driver for Contaminant Dispersion Problem

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np
import matplotlib.pyplot as plt

from mesh import *
from util import *
from master import *
from hdgker import hdg_nonlinear_solve, hdg_postprocess, baseMatIni


def HDGCompute(ngrid = 8,porder=3,taud = 1):
    #Mesh
    wig    = 0.0  # amopunt of mesh distortion
    #Equations
    kappa   = 1.0
    taudInp = [taud,taud] #[Inner Face, Boundary Face]
    funCv   = [lambda u: 0.5*(u**2),lambda u: 0.5*(u**2)] #convection function fx and fy
    dFunCv  = [lambda u: u,lambda u: u]

    def source(p, kappa = kappa):
        u = np.sin(np.pi*p[:,0])*np.sin(np.pi*p[:,1]) 
        dudx = np.pi*np.cos(np.pi*p[:,0])*np.sin(np.pi*p[:,1])
        dudy = np.pi*np.sin(np.pi*p[:,0])*np.cos(np.pi*p[:,1])
        d2udx2 = -np.pi**2*np.sin(np.pi*p[:,0])*np.sin(np.pi*p[:,1])
        d2udy2 = -np.pi**2*np.sin(np.pi*p[:,0])*np.sin(np.pi*p[:,1])
        f = u*dudx + u*dudy - kappa*d2udx2 - kappa*d2udy2 
        return f

    uhInitF = lambda p: np.zeros((p.shape[0],1)) # uh initial condition function
    #Time
    time_total  = 1e6#0.04
    dT          = 1e6#0.001 #0.0005 #2.0e-03
    nstep       = 1
    ##########################User Control

    ##Mesh Assembly
    #Coarse Mesh
    mesh   = mkmesh_square(ngrid, ngrid, porder)
    mesh   = mkmesh_distort(mesh, wig)            # Mesh distortion
    master = mkmaster(mesh, 2*porder)
    npl    = mesh.dgnodes.shape[0]
    nt     = mesh.t.shape[0]
    nps    = master.porder+1 #nPoly1d
    #Fine Mesh
    mesh1   = mkmesh_square(ngrid, ngrid, porder+1)
    mesh1   = mkmesh_distort(mesh1, wig)
    master1 = mkmaster(mesh1, 2*(porder+1))

    ##Time Assembly
    ncycl = int(np.ceil(time_total/(nstep*dT)))

    ##Equations Assembly
    param  = {'kappa': kappa, 'funCv': funCv, 'dFunCv': dFunCv, 'dT': dT} #Add dT as an entry only if you want to run time depend but not steady state
    uhInit = np.zeros((npl,nt)) 
    for i in range(nt):
        uhInit[:,i] = np.squeeze(uhInitF(mesh.dgnodes[:,:,i])) #[nPoly X nElement]
    uhPrev  = uhInit #Update n-1 state Need to be Accurate
    uhInter = uhInit #Intermediate step uh
    qhInter = np.zeros((npl, 2, nt)) #Can be Inaccurate
    uhathInter = np.zeros((nps*3,nt)) #Can be Inaccurate (This is very crucial for the boundary condition Need to do more carefully)
    
    ##Generate based matrix
    BMatC = np.zeros(((npl+nps)*3,(npl+nps)*3,nt))
    for i in range(nt):
        taudLst = np.zeros((3))
        for j in range(3):
            numFace = abs(mesh.t2f[i,j])-1 #face number
            if mesh.f[numFace,3] < -0.5: #boundary
                taudLst[j] = taudInp[1]
            else:
                taudLst[j] = taudInp[0]
        BMatC[:,:,i] = baseMatIni(mesh.dgnodes[:,:,i], master, param, taudLst)

    ##Simulation
    #Initialization
    tm = 0.0
    #Start iteration
    for i in range(ncycl):
        tm = tm + nstep*dT
        for j in range(nstep):
            # HDG Solution
            uh, qh, uhath = hdg_nonlinear_solve(master, mesh, source, param, taudInp, BMatC, uhPrev=uhPrev, qhInter=qhInter, uhInter=uhInter, uhathInter=uhathInter)
            # HDG postprocessing
            ustarh  = hdg_postprocess(master, mesh, master1, mesh1, uh, qh/kappa)
            # Update n-1 state
            uhPrev     = uh
            qhInter    = qh
            uhInter    = uh
            uhathInter = uhath #Should I update this, let me think???
    return ustarh, uh, qh, mesh, mesh1


if __name__ == "__main__":
    ngrid  = np.array([8,15,22])
    porder = np.array([1,2,3])
    hGap   = 1/(ngrid-1)
    taud   = np.ones(len(ngrid)) #1/hGap #hGap #np.ones(len(ngrid)) # 1, hGap, or 1/hGap
    
    exactu = lambda p: np.sin(np.pi*p[:,0])*np.sin(np.pi*p[:,1])
    exactqx = lambda p: -np.cos(np.pi*p[:,0])*np.sin(np.pi*p[:,1])*np.pi
    exactqy = lambda p: -np.sin(np.pi*p[:,0])*np.cos(np.pi*p[:,1])*np.pi
    
    Errtot = np.ones((4,len(porder),len(ngrid)))*99999

    for i in range(len(porder)):
        for j in range(len(ngrid)):
            ustarh, uh, qh, mesh0, mesh1 = HDGCompute(ngrid = ngrid[j],porder = porder[i],taud = taud[j])
            ustarhErr = np.sqrt(l2_error(mesh1, ustarh, exactu))
            qhxErr = np.sqrt(l2_error(mesh0, qh[:,0,:], exactqx))
            qhyErr = np.sqrt(l2_error(mesh0, qh[:,1,:], exactqy))
            uhErr = np.sqrt(l2_error(mesh0, uh, exactu))
            Errtot[0,i,j] = ustarhErr
            Errtot[1,i,j] = qhxErr
            Errtot[2,i,j] = qhyErr
            Errtot[3,i,j] = uhErr
    


    #Plot Options
    linFitSta = 0
    kPlusWhatExp = 1
    idxVar = 3 #0->uh* 1->qx 2->qy 3->uh
    varPlo = 'uh'
    taudIs = '1'
    #Plot Option Over

    ErrExp = 'p+'+str(kPlusWhatExp)
    fig = plt.figure(1,dpi = 300)
    ax  = fig.add_subplot(1,1,1)
    for idx in range(len(porder)):
        const = np.log(Errtot[idxVar][idx][linFitSta])-(porder[idx]+kPlusWhatExp)*np.log(hGap[linFitSta])
        errExp = hGap**(porder[idx]+kPlusWhatExp)*np.exp(const) 
        plt.loglog(hGap,Errtot[idxVar,idx],'.',label=(varPlo + " p="+str(porder[idx])))
        plt.loglog(hGap,errExp,'--',label=(ErrExp+"for p="+str(porder[idx])))
    plt.ylabel('L2 Error with taud = '+taudIs)
    plt.xlabel('h')
    plt.grid(True)
    plt.legend()
    plt.savefig(varPlo+'_'+taudIs+".jpg")
    
    print('end')

        


  