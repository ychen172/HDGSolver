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

if __name__ == "__main__":
    ##########################User Control
    #Mesh
    ngrid  = 15 #8,15,22
    porder = 4
    wig    = 0.1  # amopunt of mesh distortion
    #Equations
    kappa   = 1.0
    taudInp = [1,1] #[Inner Face, Boundary Face]
    funCv   = [lambda u: 10*u,lambda u: 10*u] #convection function fx and fy
    dFunCv  = [lambda u: u*0+10,lambda u: u*0+10]

    # def source(p, kappa = kappa):
    #     u = np.sin(np.pi*p[:,0])*np.sin(np.pi*p[:,1]) 
    #     dudx = np.pi*np.cos(np.pi*p[:,0])*np.sin(np.pi*p[:,1])
    #     dudy = np.pi*np.sin(np.pi*p[:,0])*np.cos(np.pi*p[:,1])
    #     d2udx2 = -np.pi**2*np.sin(np.pi*p[:,0])*np.sin(np.pi*p[:,1])
    #     d2udy2 = -np.pi**2*np.sin(np.pi*p[:,0])*np.sin(np.pi*p[:,1])
    #     f = u*dudx + u*dudy - kappa*d2udx2 - kappa*d2udy2 
    #     return f

    source = lambda p: 0*p[:,0]+1 # uh initial condition function

    uhInitF = lambda p: np.zeros((p.shape[0],1)) # uh initial condition function
    #Time
    time_total  = 0.12#0.04
    dT          = 0.0001#0.001 #0.0005 #2.0e-03
    nstep       = 10
    #Option
    display = False
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
    pause = lambda : input('(press enter to continue)')
    
    ##Equations Assembly
    param  = {'kappa': kappa, 'funCv': funCv, 'dFunCv': dFunCv, 'dT': dT} #Add dT as an entry only if you want to run time depend but not steady state
    uhInit = np.zeros((npl,nt)) 
    for i in range(nt):
        uhInit[:,i] = np.squeeze(uhInitF(mesh.dgnodes[:,:,i])) #[nPoly X nElement]
    uhPrev  = uhInit #Update n-1 state Need to be Accurate
    uhInter = uhInit #Intermediate step uh
    qhInter = np.zeros((npl, 2, nt)) #Can be Inaccurate
    uhathInter = np.zeros((nps*3,nt)) #Can be Inaccurate (This is very crucial for the boundary condition Need to do more carefully)
    
    ##Output Assembly
    if os.path.isdir(os.getcwd() + '/' + 'Movie' + '/') == False:
        os.mkdir(os.getcwd() + '/' + 'Movie' + '/')
    
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
    counterTime = 0
    #Intial Plot
    fig, axs = plt.subplots(2,figsize=(5,8))
    plt.ion()
    scaplot_raw(axs[0], mesh, uhInit, show_mesh=True, pplot=porder+2, title='HDG Solution')
    fig.suptitle('Time: '+str(tm)+' nGrid: '+str(ngrid))
    print("Time:", tm)
    #Post-process
    if display:
        pause()
    else:
        counterTime += 1
        plt.savefig(os.getcwd() + '/' + 'Movie' + '/' +'nGrid'+str(int(ngrid))+'_T'+str(counterTime)+".jpg")
    plt.close("all") #This make color bar change in each iteration
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
        #Intermediate Plot
        fig, axs = plt.subplots(2,figsize=(5,8))
        plt.ion()
        scaplot_raw(axs[0], mesh, uh, show_mesh=True, pplot=porder+2, title='HDG Solution')
        scaplot_raw(axs[1], mesh1, ustarh, show_mesh=True, pplot=porder+3, title='Postprocessed Solution')
        fig.suptitle('Time: '+str(tm)+' nGrid: '+str(ngrid))
        print("Time:", tm)
        #Post-process
        if display:
            pause()
        else:
            counterTime += 1
            plt.savefig(os.getcwd() + '/' + 'Movie' + '/' +'nGrid'+str(int(ngrid))+'_T'+str(counterTime)+".jpg")
        plt.close("all")
    plt.ioff()


