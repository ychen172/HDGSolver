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

    mesh = mkmesh_square(ngrid, ngrid, porder)
    mesh = mkmesh_distort(mesh, wig)            # Mesh distortion
    master = mkmaster(mesh, 2*porder)

    kappa = 1.0
    c = [10, 10] #[1,1] #[10,10]

    param = {'kappa': kappa, 'c': c}
    source = lambda p: 0*p[:,0]+1
    dbc    = lambda p: np.zeros((p.shape[0],1))

    taudInp = [taud,taud] #[Inner Face, Boundary Face]
    # HDG Solution
    uh, qh, uhath = hdg_solve(master, mesh, source, dbc, param, taudInp)

    # HDG postprocessing
    mesh1   = mkmesh_square(ngrid, ngrid, porder+1)
    mesh1   = mkmesh_distort(mesh1, wig)
    master1 = mkmaster(mesh1, 2*(porder+1))
    ustarh  = hdg_postprocess(master, mesh, master1, mesh1, uh, qh/kappa)

    fig, axs = plt.subplots(2,figsize=(5,8))
    # pause = lambda : input('(press enter to continue)')
    plt.ion()
    scaplot_raw(axs[0], mesh, uh, show_mesh=True, pplot=porder+2, title='HDG Solution')
    scaplot_raw(axs[1], mesh1, ustarh, show_mesh=True, pplot=porder+3, title='Postprocessed Solution')
    fig.suptitle('Cx: '+str(c[0])+' Cy: '+str(c[1])+' nGrid: '+str(ngrid))
    plt.savefig('Cx'+str(int(c[0]))+'nGrid'+str(int(ngrid))+".jpg")
    # pause()

    # pause = lambda : input('(press enter to continue)')
    # plt.ion()
    # scaplot(mesh, uh, show_mesh=False, pplot=porder+2, interactive=True, title='HDG Solution')
    # pause()
    # scaplot(mesh1, ustarh, show_mesh=False, pplot=porder+3, interactive=True, title='Postprocessed Solution')
    # pause() 


