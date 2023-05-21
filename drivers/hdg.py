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
    porder = 3
    siz = 0.2

    mesh = mkmesh_circle(siz, porder)
    master = mkmaster(mesh, 2*porder)

    kappa = 5.0
    c = [50.0, 0.0]

    param = {'kappa': kappa, 'c': c}
    source = lambda p: 10.0*np.ones((p.shape[0],1))
    dbc    = lambda p: np.zeros((p.shape[0],1))

    taudInp = [kappa,kappa]
    # HDG Solution
    uh, qh, uhath = hdg_solve(master, mesh, source, dbc, param, taudInp)

    # HDG postprocessing
    mesh1   = mkmesh_circle(siz, porder+1)
    master1 = mkmaster(mesh1, 2*(porder+1))
    ustarh  = hdg_postprocess(master, mesh, master1, mesh1, uh, qh/kappa)

    fig, axs = plt.subplots(2)
    
    pause = lambda : input('(press enter to continue)')
    plt.ion()
    scaplot_raw(axs[0], mesh, uh, show_mesh=True, pplot=porder+2, title='HDG Solution')
    scaplot_raw(axs[1], mesh1, ustarh, show_mesh=True, pplot=porder+3, title='Postprocessed Solution')
    pause()

    # pause = lambda : input('(press enter to continue)')
    # plt.ion()
    # scaplot(mesh, uh, show_mesh=False, pplot=porder+2, interactive=True, title='HDG Solution')
    # pause()
    # scaplot(mesh1, ustarh, show_mesh=False, pplot=porder+3, interactive=True, title='Postprocessed Solution')
    # pause()  

        


  