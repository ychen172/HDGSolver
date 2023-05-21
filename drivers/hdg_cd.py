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


def HDGCompute(ngrid = 8,porder=3,taud = 1):
    wig = 0.0       # amopunt of mesh distortion

    mesh = mkmesh_square(ngrid, ngrid, porder)
    mesh = mkmesh_distort(mesh, wig)            # Mesh distortion
    master = mkmaster(mesh, 2*porder)

    kappa = 1.0
    c = [0.0, 0.0]

    param = {'kappa': kappa, 'c': c}
    source = lambda p: 2*(np.pi**2)*np.sin(np.pi*p[:,0])*np.sin(np.pi*p[:,1]) # f = 2*pi^2*sin(pi*x)*sin(pi*y)
    dbc    = lambda p: np.zeros((p.shape[0],1))

    taudInp = [taud,taud] #[Inner Face, Boundary Face]
    # HDG Solution
    uh, qh, uhath = hdg_solve(master, mesh, source, dbc, param, taudInp)

    # HDG postprocessing
    mesh1   = mkmesh_square(ngrid, ngrid, porder+1)
    mesh1   = mkmesh_distort(mesh1, wig)
    master1 = mkmaster(mesh1, 2*(porder+1))
    ustarh  = hdg_postprocess(master, mesh, master1, mesh1, uh, qh/kappa)

    # fig, axs = plt.subplots(2)
    
    # pause = lambda : input('(press enter to continue)')
    # plt.ion()
    # scaplot_raw(axs[0], mesh, uh, show_mesh=True, pplot=porder+2, title='HDG Solution')
    # scaplot_raw(axs[1], mesh1, ustarh, show_mesh=True, pplot=porder+3, title='Postprocessed Solution')
    # pause()

    # pause = lambda : input('(press enter to continue)')
    # plt.ion()
    # scaplot(mesh, uh, show_mesh=False, pplot=porder+2, interactive=True, title='HDG Solution')
    # pause()
    # scaplot(mesh1, ustarh, show_mesh=False, pplot=porder+3, interactive=True, title='Postprocessed Solution')
    # pause() 
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
    idxVar = 1 #0->uh* 1->qx 2->qy 3->uh
    varPlo = 'qx'
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

        


  