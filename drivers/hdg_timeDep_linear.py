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
from hdgker import hdg_solve, hdg_postprocess

if __name__ == "__main__":
    ngrid  = 15 #8,15,22
    porder = 4
    taud   = 1
    wig    = 0.1  # amopunt of mesh distortion

    #Coarse Mesh
    mesh = mkmesh_square(ngrid, ngrid, porder)
    mesh = mkmesh_distort(mesh, wig)            # Mesh distortion
    master = mkmaster(mesh, 2*porder)
    #Fine Mesh
    mesh1   = mkmesh_square(ngrid, ngrid, porder+1)
    mesh1   = mkmesh_distort(mesh1, wig)
    master1 = mkmaster(mesh1, 2*(porder+1))

    #Add time control
    time_total  = 0.04
    dt          = 0.001 #0.0005 #2.0e-03
    nstep       = 1
    ncycl       = int(np.ceil(time_total/(nstep*dt)))
    pause = lambda : input('(press enter to continue)')
    
    # Parameters control
    kappa = 1.0
    c = [100, 100] #[1,1] #[10,10]
    
    # Assemble Inputs
    param  = {'kappa': kappa, 'c': c, 'dT': dt} #Add dT as an entry only if you want to run time depend but not steady state
    source = lambda p: 0*p[:,0]+1
    dbc    = lambda p: np.zeros((p.shape[0],1))
    uhInitF = lambda p: np.zeros((p.shape[0],1)) # uh initial condition function
    npl     = mesh.dgnodes.shape[0]
    nt      = mesh.t.shape[0]
    uhInit  = np.zeros((npl,nt)) 
    for i in range(nt):
        uhInit[:,i] = np.squeeze(uhInitF(mesh.dgnodes[:,:,i])) #[nPoly X nElement]
    taudInp = [taud,taud] #[Inner Face, Boundary Face]

    ##Create Movie Folder for Printout
    if os.path.isdir(os.getcwd() + '/' + 'Movie' + '/') == False:
        os.mkdir(os.getcwd() + '/' + 'Movie' + '/')
    ##Simulation
    counterTime = 0
    #Initial state
    tm    = 0.0
    fig, axs = plt.subplots(2,figsize=(5,8))
    plt.ion()
    scaplot_raw(axs[0], mesh, uhInit, show_mesh=True, pplot=porder+2, title='HDG Solution')
    fig.suptitle('Time: '+str(tm)+' Cx: '+str(c[0])+' Cy: '+str(c[1])+' nGrid: '+str(ngrid))
    print("Time:", tm)
    #####Interact
    # pause()
    #####Save
    counterTime += 1
    plt.savefig(os.getcwd() + '/' + 'Movie' + '/' + 'Cx'+str(int(c[0]))+'nGrid'+str(int(ngrid))+'_T'+str(counterTime)+".jpg")
    #####Over
    plt.close("all")
    uhPrev = uhInit #Update n-1 state
    #Start iteration
    for i in range(ncycl):
        tm = tm + nstep*dt
        for j in range(nstep):
            # HDG Solution
            uh, qh, uhath = hdg_solve(master, mesh, source, dbc, param, taudInp, uhPrev=uhPrev)
            # HDG postprocessing
            ustarh  = hdg_postprocess(master, mesh, master1, mesh1, uh, qh/kappa)
            # Update n-1 state
            uhPrev  = uh
        #Plot intermediate results
        fig, axs = plt.subplots(2,figsize=(5,8))
        plt.ion()
        scaplot_raw(axs[0], mesh, uh, show_mesh=True, pplot=porder+2, title='HDG Solution')
        scaplot_raw(axs[1], mesh1, ustarh, show_mesh=True, pplot=porder+3, title='Postprocessed Solution')
        fig.suptitle('Time: '+str(tm)+' Cx: '+str(c[0])+' Cy: '+str(c[1])+' nGrid: '+str(ngrid))
        print("Time:", tm)
        #####Interact
        # pause()
        #####Save
        counterTime += 1
        plt.savefig(os.getcwd() + '/' + 'Movie' + '/' + 'Cx'+str(int(c[0]))+'nGrid'+str(int(ngrid))+'_T'+str(counterTime)+".jpg")
        #####Over
        plt.close("all")
    plt.ioff()
    # plt.savefig('Cx'+str(int(c[0]))+'nGrid'+str(int(ngrid))+".jpg")


