__all__ = ['rk4']

def rk4(residexpl, master, mesh, app, u, time, dt, nstep):
    """
    rk4 time integrator using a 4 stage runge-kutta scheme.
 
       residexpl:    pointer to residual evaluation function
                     r=residexpl(master,mesh,app,u,time)
       master:       master structure
       mesh:         mesh structure
       app:          application structure
       u(npl,nc,nt): vector of unknowns
                     npl = size(mesh.plocal,1)
                     nc = app.nc (number of equations in system)
                     nt = size(mesh.t,1)
       time:         time
       dt:           time step
       nstep:        number of steps to be performed
       r(npl,nc,nt): residual vector (=du/dt) (already divided by mass
                     matrix)                             
    """
    for i in range(nstep):   
        k1 = dt*residexpl( master, mesh, app, u       , time       )
        k2 = dt*residexpl( master, mesh, app, u+0.5*k1, time+0.5*dt)
        k3 = dt*residexpl( master, mesh, app, u+0.5*k2, time+0.5*dt)
        k4 = dt*residexpl( master, mesh, app, u+    k3, time+    dt)
        u = u + k1/6 + k2/3 + k3/3 + k4/6
        time = time + dt

    return u