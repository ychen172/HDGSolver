import numpy.matlib
import numpy as np

__all__ = ['rldgexpl']


def getq(master, mesh, app, u, time):
    """
    getq calculates the gradient quantities q = \nabla u for use in the the ldg method.

        master:           master structure
        mesh:             mesh structure
        app:              application structure
        u[npl,nc,nt]:     vector of unknowns
                         npl = size(mesh.plocal,1)
                         nc = app.nc (number of equations in system)
                         nt = size(mesh.t,1)
        time:         time
        q[npl,2,nc,nt]: q vector    
    """


    nt   = mesh.t.shape[0]
    nc   = app.nc
    npl  = u.shape[0]
    ng1d = master.gw1d.shape[0]

    q = np.zeros((npl, 2, nc, nt))

    # Interfaces
    sh1d = np.squeeze(master.sh1d[:,0,:])
    perm = master.perm
    ni = np.where(mesh.f[:,3] < 0)[0][0]

    # Interior first
    for i in range(ni):
        ipt  = np.sum(mesh.f[i,:2])
        el  = mesh.f[i,2]
        er  = mesh.f[i,3]
    
        ipl = np.sum(mesh.t[el,:])-ipt
        isl = np.where(mesh.t[el,:] == ipl)
        if mesh.t2f[el,isl] > 0:
            iol = 0
        else:
            iol = 1

        ipr = np.sum(mesh.t[er,:])-ipt
        isr = np.where(mesh.t[er,:] == ipr)
        if mesh.t2f[er,isr] > 0:
            ior = 0
        else:
            ior = 1
    
        perml = np.squeeze(perm[:,isl,iol])
        permr = np.squeeze(perm[:,isr,ior])
    
        if mesh.fcurved[i]:
            xxi = np.squeeze(master.sh1d[:,1,:]).T @ np.squeeze(mesh.dgnodes[perml,0,el]) 
            yxi = np.squeeze(master.sh1d[:,1,:]).T @ np.squeeze(mesh.dgnodes[perml,1,el])
            dsdxi = np.sqrt(xxi**2 + yxi**2)
            nl = np.column_stack((yxi/dsdxi, -xxi/dsdxi))
        else:
            dx = mesh.p[mesh.f[i,1],:] - mesh.p[mesh.f[i,0],:]
            dsdxi = np.sqrt(dx[0]**2 + dx[1]**2)
            nl = np.matlib.repmat(np.array([dx[1], -dx[0]]/dsdxi), ng1d ,1)

        ul = u[perml,:,el]                  # Sample u from the right
 
        cntx = sh1d @ np.diag(master.gw1d * dsdxi * nl[:,0]) @ sh1d.T @ ul
        cnty = sh1d @ np.diag(master.gw1d * dsdxi * nl[:,1]) @ sh1d.T @ ul
   
        q[perml,0,:,el] = q[perml,0,:,el] + cntx
        q[perml,1,:,el] = q[perml,1,:,el] + cnty
        q[permr,0,:,er] = q[permr,0,:,er] - cntx
        q[permr,1,:,er] = q[permr,1,:,er] - cnty

    # Now boundary
    for i in range(ni, mesh.f.shape[0]):
        ipt  = np.sum(mesh.f[i,:2])
        el  = mesh.f[i,2]
        ib  = -mesh.f[i,3]-1

        ipl = np.sum(mesh.t[el,:])-ipt
        isl = np.where(mesh.t[el,:] == ipl)
        if mesh.t2f[el,isl] > 0:
            iol = 0
        else:
            iol = 1

        perml = np.squeeze(perm[:,isl,iol])

        if mesh.fcurved[i]:
            xxi = np.squeeze(master.sh1d[:,1,:]).T @ np.squeeze(mesh.dgnodes[perml,0,el]) 
            yxi = np.squeeze(master.sh1d[:,1,:]).T @ np.squeeze(mesh.dgnodes[perml,1,el])
            dsdxi = np.sqrt(xxi**2 + yxi**2)
            nl = np.column_stack((yxi/dsdxi, -xxi/dsdxi))
        else:
            dx = mesh.p[mesh.f[i,1],:] - mesh.p[mesh.f[i,0],:]
            dsdxi = np.sqrt(dx[0]**2 + dx[1]**2)
            nl = np.matlib.repmat(np.array([dx[1], -dx[0]]/dsdxi), ng1d ,1)

        ul = u[perml,:,el]
        ub = app.fvisub( ul, nl, app.bcm[ib], app.bcs[app.bcm[ib],:], None, app.arg, time)

        cntx = sh1d @ np.diag(master.gw1d * dsdxi * nl[:,0]) @ sh1d.T @ ub
        cnty = sh1d @ np.diag(master.gw1d * dsdxi * nl[:,1]) @ sh1d.T @ ub
   
        q[perml,0,:,el] = q[perml,0,:,el] + cntx
        q[perml,1,:,el] = q[perml,1,:,el] + cnty

    # Volume integral
    shap = master.shap[:,0,:]
    shapxi = master.shap[:,1,:]
    shapet = master.shap[:,2,:]
    shapxig = shapxi @ np.diag(master.gwgh)
    shapetg = shapet @ np.diag(master.gwgh)

    for i in range(nt):
        if mesh.tcurved[i]:
            xxi = mesh.dgnodes[:,0,i] @ shapxi
            xet = mesh.dgnodes[:,0,i] @ shapet
            yxi = mesh.dgnodes[:,1,i] @ shapxi
            yet = mesh.dgnodes[:,1,i] @ shapet
            jac = xxi * yet - xet * yxi
            shapx =   shapxig @ np.diag(yet) - shapetg @ np.diag(yxi)
            shapy = - shapxig @ np.diag(xet) + shapetg @ np.diag(xxi)
            M  = shap @ np.diag(master.gwgh * jac) @ shap.T
            Cx = shap @ shapx.T
            Cy = shap @ shapy.T
        else:
            xxi = mesh.p[mesh.t[i,1],0] - mesh.p[mesh.t[i,0],0]
            xet = mesh.p[mesh.t[i,2],0] - mesh.p[mesh.t[i,0],0]
            yxi = mesh.p[mesh.t[i,1],1] - mesh.p[mesh.t[i,0],1]
            yet = mesh.p[mesh.t[i,2],1] - mesh.p[mesh.t[i,0],1]
            jac = xxi * yet - xet * yxi
            shapx =   shapxig * yet - shapetg * yxi
            shapy = - shapxig * xet + shapetg * xxi 
            M = master.mass * jac
            Cx =   master.conv[:,:,0]*yet - master.conv[:,:,1]*yxi
            Cy = - master.conv[:,:,0]*xet + master.conv[:,:,1]*xxi

        q[:,0,:,i] = q[:,0,:,i] - Cx.T @ u[:,:,i]
        q[:,1,:,i] = q[:,1,:,i] - Cy.T @ u[:,:,i]
   
        q[:,:,:,i] = np.linalg.solve(M, q[:,:,:,i].reshape((npl, 2*nc))).reshape((npl, 2, nc))

    return q

def rldgexpl(master, mesh, app, u, time):
    """
    rldgexpl calculates the residual vector for explicit time stepping using the LDG method
 
       master:       master structure
       mesh:         mesh structure
       app:          application structure
       u[npl,nc,nt]: vector of unknowns
                     npl = size(mesh.plocal,1)
                     nc = app.nc (number of equations in system)
                     nt = size(mesh.t,1)
       time:         time
       r[npl,nc,nt]: residual vector (=du/dt) (already divided by mass
                     matrix)   
    """

    nt   = mesh.t.shape[0]
    nc   = app.nc
    npl  = u.shape[0]
    np1d = master.perm.shape[0]
    ng   = master.gwgh.shape[0]
    ng1d = master.gw1d.shape[0]

    if app.fvisv:
        q = getq(master, mesh, app, u, time)

    r = np.zeros_like(u)

    # Interfaces
    sh1d = np.squeeze(master.sh1d[:,0,:])
    perm = master.perm
    ni = np.where(mesh.f[:,3] < 0)[0][0]

    # Interior first
    for i in range(ni):
        ipt  = np.sum(mesh.f[i,:2])
        el  = mesh.f[i,2]
        er  = mesh.f[i,3]
    
        ipl = np.sum(mesh.t[el,:])-ipt
        isl = np.where(mesh.t[el,:] == ipl)
        if mesh.t2f[el,isl] > 0:
            iol = 0
        else:
            iol = 1

        ipr = np.sum(mesh.t[er,:])-ipt
        isr = np.where(mesh.t[er,:] == ipr)
        if mesh.t2f[er,isr] > 0:
            ior = 0
        else:
            ior = 1
    
        perml = np.squeeze(perm[:,isl,iol])
        permr = np.squeeze(perm[:,isr,ior])

        if app.pg:
            plg = sh1d.T @ mesh.dgnodes[perml,:,el]
        else:
            plg = []
    
        if mesh.fcurved[i]:
            xxi = np.squeeze(master.sh1d[:,1,:]).T @ np.squeeze(mesh.dgnodes[perml,0,el]) 
            yxi = np.squeeze(master.sh1d[:,1,:]).T @ np.squeeze(mesh.dgnodes[perml,1,el])
            dsdxi = np.sqrt(xxi**2 + yxi**2)
            nl = np.column_stack((yxi/dsdxi, -xxi/dsdxi))
            dws = master.gw1d * dsdxi
        else:
            dx = mesh.p[mesh.f[i,1],:] - mesh.p[mesh.f[i,0],:]
            dsdxi = np.sqrt(dx[0]**2 + dx[1]**2)
            nl = np.matlib.repmat(np.array([dx[1], -dx[0]]/dsdxi), ng1d ,1)
            dws = master.gw1d * dsdxi

        ul = u[perml,:,el]
        ulg = sh1d.T @ ul
        
        ur = u[permr,:,er]
        urg = sh1d.T @ ur
 
        fng = app.finvi( ulg, urg, nl, plg, app.arg, time)

        if app.fvisv:
            qr = q[permr,:,:,er]
            qrg = np.reshape(sh1d.T @ qr.reshape((np1d,2*nc)), (ng1d,2,nc))
            fnvg = app.fvisi( ulg, urg, None, qrg, nl, plg, app.arg, time)
            fng = fng + fnvg

        cnt = sh1d @ np.diag(dws) @ fng 
   
        ci = cnt.reshape((np1d, nc))
        r[perml,:,el] = r[perml,:,el] - ci
        r[permr,:,er] = r[permr,:,er] + ci

    # Now Boundary
    for i in range(ni, mesh.f.shape[0]):
        ipt  = np.sum(mesh.f[i,:2])
        el  = mesh.f[i,2]
        ib  = -mesh.f[i,3]-1

        ipl = np.sum(mesh.t[el,:])-ipt
        isl = np.where(mesh.t[el,:] == ipl)
        if mesh.t2f[el,isl] > 0:
            iol = 0
        else:
            iol = 1

        perml = np.squeeze(perm[:,isl,iol])

        if app.pg:
            plg = sh1d.T @ mesh.dgnodes[perml,:,el]
        else:
            plg = []

        if mesh.fcurved[i]:
            xxi = np.squeeze(master.sh1d[:,1,:]).T @ np.squeeze(mesh.dgnodes[perml,0,el]) 
            yxi = np.squeeze(master.sh1d[:,1,:]).T @ np.squeeze(mesh.dgnodes[perml,1,el])
            dsdxi = np.sqrt(xxi**2 + yxi**2)
            nl = np.column_stack((yxi/dsdxi, -xxi/dsdxi))
            dws = master.gw1d * dsdxi
        else:
            dx = mesh.p[mesh.f[i,1],:] - mesh.p[mesh.f[i,0],:]
            dsdxi = np.sqrt(dx[0]**2 + dx[1]**2)
            nl = np.matlib.repmat(np.array([dx[1], -dx[0]]/dsdxi), ng1d ,1)
            dws = master.gw1d * dsdxi

        ul = u[perml,:,el]
        ulg = sh1d.T @ ul
        
        fng = app.finvb( ulg, nl, app.bcm[ib], app.bcs[app.bcm[ib],:], plg, app.arg, time)

        if app.fvisv:
            ql = q[perml,:,:,el]
            qlg = np.reshape(sh1d.T @ ql.reshape((np1d,2*nc)), (ng1d,2,nc))
            fnvg = app.fvisb( ulg, qlg, nl, app.bcm[ib], app.bcs[app.bcm[ib],:], plg, app.arg, time)
            fng = fng + fnvg
        
        cnt = sh1d @ np.diag(dws) @ fng 
   
        ci = cnt.reshape((np1d, nc))
        r[perml,:,el] = r[perml,:,el] - ci

    # Volume integral
    shap = master.shap[:,0,:]
    shapxi = master.shap[:,1,:]
    shapet = master.shap[:,2,:]
    shapxig = shapxi @ np.diag(master.gwgh)
    shapetg = shapet @ np.diag(master.gwgh)

    for i in range(nt):

        if app.pg:
            pg = shap.T @ mesh.dgnodes[:,:,i]
        else:
            pg = []

        if mesh.tcurved[i]:
            xxi = mesh.dgnodes[:,0,i] @ shapxi
            xet = mesh.dgnodes[:,0,i] @ shapet
            yxi = mesh.dgnodes[:,1,i] @ shapxi
            yet = mesh.dgnodes[:,1,i] @ shapet
            jac = xxi * yet - xet * yxi
            shapx =   shapxig @ np.diag(yet) - shapetg @ np.diag(yxi)
            shapy = - shapxig @ np.diag(xet) + shapetg @ np.diag(xxi)
            M  = shap @ np.diag(master.gwgh * jac) @ shap.T
        else:
            xxi = mesh.p[mesh.t[i,1],0] - mesh.p[mesh.t[i,0],0]
            xet = mesh.p[mesh.t[i,2],0] - mesh.p[mesh.t[i,0],0]
            yxi = mesh.p[mesh.t[i,1],1] - mesh.p[mesh.t[i,0],1]
            yet = mesh.p[mesh.t[i,2],1] - mesh.p[mesh.t[i,0],1]
            jac = xxi * yet - xet * yxi
            shapx =   shapxig * yet - shapetg * yxi
            shapy = - shapxig * xet + shapetg * xxi 
            M = master.mass * jac

        ug = shap.T @ u[:,:,i]
   
        if app.src:
            src = app.src( ug, [], pg, app.arg, time)
            r[:,:,i] = r[:,:,i] + shap @ np.diag(master.gwgh * jac) @ src
   
        fgx, fgy = app.finvv( ug, pg, app.arg, time)

        if app.fvisv:
            qg = np.reshape(shap.T @ q[:,:,:,i].reshape((npl, 2*nc)), (ng, 2, nc))
            fxvg, fyvg = app.fvisv( ug, qg, pg, app.arg, time)
            fgx = fgx + fxvg
            fgy = fgy + fyvg

        r[:,:,i] =  r[:,:,i] + shapx @ fgx + shapy @ fgy
   
        r[:,:,i] = np.linalg.solve(M, r[:,:,i])

    return r