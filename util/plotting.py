#-----------------------------------------------------------------------------
#  Adapted from:
# 
#  Copyright (C) 2004-2012 Per-Olof Persson
#  Copyright (C) 2012 Bradley Froehle
#  Distributed under the terms of the GNU General Public License. You should
#  have received a copy of the license along with this program. If not,
#  see <http://www.gnu.org/licenses/>.
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.collections import PathCollection
from matplotlib.path import Path

__all__ = ['meshplot', 'scaplot', 'scaplot_raw', 'meshplot_curved', 'meshplot_gauss']

from master import shape2d, uniformlocalpnts

colorbars = {}

#-----------------------------------------------------------------------------
# Classes
#-----------------------------------------------------------------------------

class SimplexCollection(PathCollection):
    """A collection of triangles."""
    def __init__(self, simplices, **kwargs):
        kwargs.setdefault('linewidths', 0.5)
        kwargs.setdefault('edgecolors', 'k')
        kwargs.setdefault('facecolors', (0.8, 0.9, 1.0))
        PathCollection.__init__(self, [], **kwargs)
        p, t = simplices
        code = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
        self.set_paths([Path(edge, code) for edge in p[t[:,[0,1,2,0]]]])

class DGCollection(PathCollection):
    """A collection of triangles."""
    def __init__(self, dgnodes, bou, **kwargs):
        kwargs.setdefault('linewidth', 0.5)
        kwargs.setdefault('edgecolor', 'k')
        kwargs.setdefault('facecolor', (0.8, 0.9, 1.0))
        PathCollection.__init__(self, [], **kwargs)
        code = [Path.MOVETO] + [Path.LINETO]*(len(bou)-2) + [Path.CLOSEPOLY]
        self.set_paths([Path(edge, code) for edge in np.transpose(dgnodes[bou,:,:], (2,0,1))])
        

def _draw_curved_mesh(ax, mesh, pplot, **kwargs):
    if pplot == 0:
        pplot = mesh.porder
    
    plocal, _ = uniformlocalpnts(pplot)
    perm = np.zeros((pplot+1, 3), dtype=int)
    aux = [0,1,2,0,1]
    for i in [0,1,2]:
        ii = np.where(plocal[:,i] < 1.0e-6)[0]
        jj = np.argsort(plocal[ii,aux[i+2]])
        perm[:,i] = ii[jj]

    bou = perm[:,2].tolist() + perm[1:,0].tolist() + perm[1:,1].tolist()

    shapnodes = shape2d(mesh.porder, mesh.plocal, plocal[:,1:])
    dgnodes = np.zeros((plocal.shape[0], 2, mesh.t.shape[0]), dtype=float)
    for i in range(mesh.t.shape[0]):
        dgnodes[:,0,i] = mesh.dgnodes[:,0,i] @ shapnodes[:,0,:]     
        dgnodes[:,1,i] = mesh.dgnodes[:,1,i] @ shapnodes[:,0,:]    

    c = DGCollection(dgnodes, bou, **kwargs)
    ax.add_collection(c)


def meshplot(mesh, nodes=False, annotate=''):
    """Plot a simplicial mesh

    Parameters
    ----------
    mesh : Mesh strcuture

    Additional parameters
    ------------------------
    nodes : bool, optional
        draw a marker at each node
    annotate : str, optional
        'p' : annotate nodes
        't' : annotate simplices

    """

    fig = plt.gcf()
    ax = fig.gca()

    print("entered meshplot")
    p = mesh.p
    t = mesh.t

    c = SimplexCollection((p, t))
    ax.add_collection(c)
    if nodes:
        ax.plot(mesh.dgnodes[:,0,:].flatten(), mesh.dgnodes[:,1,:].flatten(), '.k', markersize=8)
    if 'p' in annotate:
        for i in range(len(p)):
            ax.annotate(str(i), p[i]+[0,0], color='red', ha='center', va='center')
    if 't' in annotate:
        for it in range(len(t)):
            pmid = p[t[it]].mean(0)
            ax.annotate(str(it), pmid, ha='center', va='center')
            
    ax.set_aspect('equal')
    ax.autoscale_view(scalex=True, scaley=True)

    #ax.set_title('title')
    #ax.set_xlabel('x label')
    #ax.set_ylabel('y label')
    #ax.set_axis_off()

    xmin = np.min(mesh.p[:,0])
    xmax = np.max(mesh.p[:,0])
    ymin = np.min(mesh.p[:,1])
    ymax = np.max(mesh.p[:,1])
    dx = xmax - xmin
    dy = ymax - ymin
    ax.set_xlim(xmin=xmin-0.02*dx, xmax=xmax+0.02*dx)
    ax.set_ylim(ymin=ymin-0.02*dy, ymax=ymax+0.02*dy)

    plt.show()
    
def meshplot_gauss(mesh, master, nodes=False, gauss=True):
    """Plot a simplicial mesh

    Parameters
    ----------
    mesh : Mesh strcuture

    """

    fig = plt.gcf()
    ax = fig.gca()

    print("entered meshplot")
    p = mesh.p
    t = mesh.t

    xg = np.zeros((master.shap.shape[2], t.shape[0]))
    yg = np.zeros((master.shap.shape[2], t.shape[0]))
    for i in range(mesh.t.shape[0]):
        # Quadrature
        for j in range(len(master.gwgh)):
            # Calculate Jacobian
            xg[j,i] = np.array(master.shap[:,0,j] @ mesh.dgnodes[:,0,i])
            yg[j,i] = np.array(master.shap[:,0,j] @ mesh.dgnodes[:,1,i])


    c = SimplexCollection((p, t))
    ax.add_collection(c)
    if nodes:
        ax.plot(mesh.dgnodes[:,0,:].flatten(), mesh.dgnodes[:,1,:].flatten(), '.k', markersize=8)

    if gauss:
        ax.plot(xg, yg, '.b', markersize=8)

    ax.set_aspect('equal')
    ax.autoscale_view(scalex=True, scaley=True)

    #ax.set_title('title')
    #ax.set_xlabel('x label')
    #ax.set_ylabel('y label')
    #ax.set_axis_off()

    xmin = np.min(mesh.p[:,0])
    xmax = np.max(mesh.p[:,0])
    ymin = np.min(mesh.p[:,1])
    ymax = np.max(mesh.p[:,1])
    dx = xmax - xmin
    dy = ymax - ymin
    ax.set_xlim(xmin=xmin-0.02*dx, xmax=xmax+0.02*dx)
    ax.set_ylim(ymin=ymin-0.02*dy, ymax=ymax+0.02*dy)

    plt.show()


def scaplot(mesh, c, limits=None, show_mesh=False, pplot=0, interactive=False, title=None):
    """
    Plot countours of a scalar field

    Parameters
    ----------
    mesh : Mesh strcuture
    c    : Scalar field of dimension (number_of_local_nodes_per_element, number_of_elements)

    Additional parameters
    ------------------------
    limits = [cmin, cmax]: limits for thresholding
    show_mesh : flag to plot mesh over contours
    pplot: order of the polynomial used to display mesh
    interactive: flag used for interactive plotting
    """

    fig = plt.gcf()
    ax = fig.gca()

    scaplot_raw(ax, mesh, c, limits, show_mesh, pplot)

    if title:
        plt.title(title)

    if interactive:
        fig.canvas.draw()
    else:
        plt.show()


def scaplot_raw(ax, mesh, c, limits=None, show_mesh=False, pplot=0, title=None):
    c = np.squeeze(c)

    N = 31  # number of contours
    cmap = 'viridis' #'plasma', 'RdYlBu', 'inferno', 'viridis', 'magma'

    ax.set_aspect('equal')

    if limits is None:    
        cmin = min(np.reshape(c, (-1,)))
        cmax = max(np.reshape(c, (-1,)))
    else:
        cmin = limits[0]
        cmax = limits[1]

    # print("cmin: ", cmin)
    # print("cmax: ", cmax)

    if pplot > 0:
        plocal, tlocal = uniformlocalpnts(pplot)
        shapnodes = shape2d(mesh.porder, mesh.plocal, plocal[:,1:])
        dgnodes = np.zeros((plocal.shape[0], 2), dtype=float)
        sca = np.zeros((plocal.shape[0]), dtype=float)

    for i in range(mesh.t.shape[0]):
        if pplot > 0:
            dgnodes[:,0] = mesh.dgnodes[:,0,i] @ shapnodes[:,0,:]     
            dgnodes[:,1] = mesh.dgnodes[:,1,i] @ shapnodes[:,0,:]    
            sca = c[:,i] @ shapnodes[:,0,:]
        else:
            dgnodes = mesh.dgnodes[:,:,i]
            sca = c[:,i]
            tlocal = mesh.tlocal

        triang = tri.Triangulation(dgnodes[:,0], dgnodes[:,1], tlocal)
        ax.tricontourf(triang, sca, vmin=cmin, vmax=cmax, levels=N, cmap=cmap)
        #ax.tripcolor(mesh.dgnodes[:,0,i], mesh.dgnodes[:,1,i], mesh.tlocal, c[:,i], vmin=cmin, vmax=cmax, shading='gouraud')

    if show_mesh:
        _draw_curved_mesh(ax, mesh, pplot, facecolor='none')
        #ax.triplot(mesh.p[:,0], mesh.p[:,1], mesh.t, 'k-', lw=0.5)

    #ax.set_title('title')
    #ax.set_xlabel('x label')
    #ax.set_ylabel('y label')
    #ax.set_axis_off()

    xmin = np.min(mesh.dgnodes[:,0,:])
    xmax = np.max(mesh.dgnodes[:,0,:])
    ymin = np.min(mesh.dgnodes[:,1,:])
    ymax = np.max(mesh.dgnodes[:,1,:])
    dx = xmax - xmin
    dy = ymax - ymin
    ax.set_xlim(xmin=xmin-0.02*dx, xmax=xmax+0.02*dx)
    ax.set_ylim(ymin=ymin-0.02*dy, ymax=ymax+0.02*dy)
    
    norm = mpl.colors.Normalize(vmin=cmin, vmax=cmax)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    if title:
        ax.set_title(title)

    if not ax in colorbars:
    # create a colorbar
        colorbars[ax] = plt.colorbar(sm, ticks=np.linspace(cmin, cmax, 11), ax=ax)
        plt.show(block=False)





def meshplot_curved(mesh, nodes=False, annotate='', pplot=0):
    """
    Plot a curved mesh of triangles mesh

    Parameters
    ----------
    mesh : Mesh strcuture

    Additional parameters
    ------------------------
    nodes : bool, optional
        draw a marker at each node
    annotate : str, optional
        'p' : annotate nodes
        't' : annotate simplices

    pplot : order of the polynomial used to display mesh
    """

    fig = plt.gcf()
    ax = fig.gca()

    p = mesh.p
    t = mesh.t

    _draw_curved_mesh(ax, mesh, pplot)

    if nodes:
        ax.plot(mesh.dgnodes[:,0,:].flatten(), mesh.dgnodes[:,1,:].flatten(), '.k', markersize=10)
    if 'p' in annotate:
        for i in range(len(p)):
            ax.annotate(str(i), p[i]+[0,0.0], color='red', ha='center', va='center')
    if 't' in annotate:
        for it in range(len(t)):
            pmid = p[t[it]].mean(0)
            ax.annotate(str(it), pmid, ha='center', va='center')
            
    ax.set_aspect('equal')
    ax.autoscale_view(scalex=True, scaley=True)

    #ax.set_title('title')
    #ax.set_xlabel('x label')
    #ax.set_ylabel('y label')
    #ax.set_axis_on()

    xmin = np.min(mesh.dgnodes[:,0,:])
    xmax = np.max(mesh.dgnodes[:,0,:])
    ymin = np.min(mesh.dgnodes[:,1,:])
    ymax = np.max(mesh.dgnodes[:,1,:])
    dx = xmax - xmin
    dy = ymax - ymin
    ax.set_xlim(xmin=xmin-0.02*dx, xmax=xmax+0.02*dx)
    ax.set_ylim(ymin=ymin-0.02*dy, ymax=ymax+0.02*dy)

    plt.show()
