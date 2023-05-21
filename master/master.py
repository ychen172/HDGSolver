import numpy as np
from scipy.special import jacobi
import math

class Master:
    def __init__(self, mesh):
        self.porder = mesh.porder
        self.plocal = mesh.plocal
        self.corner = np.zeros((3,), dtype=int)
        self.perm   = np.zeros((mesh.porder+1, 3, 2), dtype=int)

        self.gp1d   = []
        self.gpts   = []
        self.sh1d   = []
        self.shap   = []

        self.mass   = [] 
        self.conv   = []


def mkmaster(mesh, pgauss=None):
    """         
    mkmaster  initialize master element structures
    master=mkmaster(mesh)

      mesh:      mesh data structure
      pgauss:    degree of the polynomial to be integrated exactly
                 (default: pgauss = 4*mesh.porder)
    """

    master = Master(mesh)

    for i in range(3):
        master.corner[i] = np.where(master.plocal[:,i] > 1-1.0e-6)[0][0]

    aux = [0,1,2,0,1]
    for i in [0,1,2]:
        ii = np.where(master.plocal[:,i] < 1.0e-6)[0]
        jj = np.argsort(master.plocal[ii,aux[i+2]])
        master.perm[:,i,0] = ii[jj]
        if i == 2:
            master.ploc1d = master.plocal[ii[jj],0:2]

    master.perm[:,:,1] = np.flipud(master.perm[:,:,0])

    if not pgauss:
        pgauss = max(4*mesh.porder, 1)

    master.gp1d, master.gw1d = gaussquad1d(pgauss)
    master.gpts, master.gwgh = gaussquad2d(pgauss)

    master.sh1d = shape1d(master.porder, master.ploc1d, master.gp1d)
    master.shap = shape2d(master.porder, master.plocal, master.gpts)

    master.conv = np.empty((master.shap.shape[0], master.shap.shape[0], 2), dtype = float)
    master.mass = master.shap[:,0,:] @ np.diag(master.gwgh) @ master.shap[:,0,:].T
    master.conv[:,:,0] = master.shap[:,0,:] @ np.diag(master.gwgh) @ master.shap[:,1,:].T
    master.conv[:,:,1] = master.shap[:,0,:] @ np.diag(master.gwgh) @ master.shap[:,2,:].T

    return master


def uniformlocalpnts(porder):
    """
   uniformlocalpnts 2-d mesh generator for the master element.
   [plocal,tlocal]=uniformlocalpnts(porder)

      plocal:    node positions (npl,2)
      tlocal:    triangle indices (nt,3)
      porder:    order of the complete polynomial 
                 npl = (porder+1)*(porder+2)/2
    """

    u, v = np.meshgrid(np.linspace(0,1,porder+1), np.linspace(0,1,porder+1))
    uf = u.flatten()
    vf = v.flatten()
    plocal = np.stack([1-uf-vf, uf, vf], axis=0).T
    ind = np.where(plocal[:,0] > -1.0e-10)
    plocal = plocal[ind,:][0]

    shf = 0
    tlocal = np.zeros((0,3), dtype=int)
    for jj in range(porder):
        ii = porder-jj
        l1 = np.vstack([[i, i+1, ii+i+1]  for i in range(ii)]) + shf
        tlocal = np.append(tlocal, l1, axis=0)
        if ii > 1:
            l2 = np.vstack([[i+1, ii+i+2, ii+i+1] for i in range(ii-1)]) + shf
            tlocal = np.append(tlocal, l2, axis=0)

        shf = shf + ii + 1
    
    return plocal, tlocal

    
def rotate_plocal(plocal, porder):
    m = np.full((porder+1, porder+1), -1, dtype=int)
    k = 0
    for i in reversed(range(porder+1)):
        for j in range(porder+1):
            if j<= i:
                m[i,j] = k
                k = k + 1

    rm = np.copy(m)
    rm = np.flipud(rm.T)
    for i in reversed(range(porder)):
        rm[i,:] = np.roll(rm[i,:], i+1)

    plocal_n = np.zeros(plocal.shape, dtype=float)
    for i in range(porder+1):
        for j in range(porder+1):
            if m[i,j] > -1:
                plocal_n[m[i,j],:] = np.roll(plocal[rm[i,j],:],1)

    return plocal_n

def localpnts1d(porder, type=0):
    """
  	porder:     Polynomial order
    nodeType:   Flag determining node distribution:
                     - nodeType = 0: Uniform distribution (default)
                     - nodeType = 1: Extended Chebyshev nodes of the first kind
                     - nodeType = 2: Extended Chebyshev nodes of the second kind
  	plocal:  Node positions on the master volume element
    """

    match type:
        case 0:
            plocal = np.linspace(0,1,porder+1)
        case 1:
            k = np.array([float(i) for i in range(porder+1)])
            plocal = -np.cos((2.0*k + 1.0)*np.pi/(2.0*float(len(k)))) / np.cos(np.pi/(2.0*float(len(k))))
            plocal = 0.5 + 0.5 * plocal
        case 2:
            k = np.array([float(porder-i) for i in range(porder+1)])
            plocal = np.cos(np.pi*k/porder)
            plocal = 0.5 + 0.5 * plocal

    return plocal

def localpnts(porder, type=0):
    """
   uniformlocalpnts 2-d mesh generator for the master element.
   [plocal,tlocal]=uniformlocalpnts(porder)

      plocal:    node positions (npl,2)
      tlocal:    triangle indices (nt,3)
      porder:    order of the complete polynomial 
                 npl = (porder+1)*(porder+2)/2
    """
    ploc1d = localpnts1d(porder, type)
    u, v = np.meshgrid(ploc1d, ploc1d)
    uf = u.flatten()
    vf = v.flatten()
    plocal = np.stack([1-uf-vf, uf, vf], axis=0).T
    ind = np.where(plocal[:,0] > -1.0e-10)
    plocal = plocal[ind,:][0]

    # rotate and averge to preserve symmetry
    plocal_1 = rotate_plocal(plocal, porder)
    plocal_2 = rotate_plocal(plocal_1, porder)
    plocal = (plocal + plocal_1 + plocal_2)/3.0

    shf = 0
    tlocal = np.zeros((0,3), dtype=int)
    for jj in range(porder):
        ii = porder-jj
        l1 = np.vstack([[i, i+1, ii+i+1]  for i in range(ii)]) + shf
        tlocal = np.append(tlocal, l1, axis=0)
        if ii > 1:
            l2 = np.vstack([[i+1, ii+i+2, ii+i+1] for i in range(ii-1)]) + shf
            tlocal = np.append(tlocal, l2, axis=0)

        shf = shf + ii + 1
    
    return plocal, tlocal
    
def shape1d(porder, plocal, pts):
    """     
    shape1d calculates the nodal shapefunctions and its derivatives for 
             the master 1d element [0,1]
    nfs=shape1d(porder,plocal,pts)
 
       porder:    polynomial order
       plocal:    node positions (np) (np=porder+1)
       pts:       coordinates of the points where the shape fucntions
                   and derivatives are to be evaluated (npoints)
       nsf:       shape function adn derivatives (np,2,npoints)
                  nsf[:,0,:] shape functions 
                  nsf[:,1,:] shape fucntions derivatives w.r.t. x 
    """

    f, fx = koornwinder1d(pts, porder)
    A, _ = koornwinder1d(plocal[:,1], porder)
    nf = np.linalg.solve(A.T, f.T)
    nfx = np.linalg.solve(A.T, fx.T)

    nfs = np.full((porder+1, 2, pts.shape[0]), 0.0)
    nfs[:,0,:] = nf
    nfs[:,1,:] = nfx

    return nfs

def shape2d(porder, plocal, pts):
    """     
    shape2d calculates the nodal shapefunctions and its derivatives for 
            the master triangle [0,0]-[1,0]-[0,1]
    nfs=shape2d(porder,plocal,pts)

    porder:    polynomial order
    plocal:    node positions (np,2) (np=(porder+1)*(porder+2)/2)
    pts:       coordinates of the points where the shape fucntions
                 and derivatives are to be evaluated (npoints,2)
    nfs:       shape function adn derivatives (np,3,npoints)
                 nsf[:,0,:] shape functions 
                 nsf[:,1,:] shape fucntions derivatives w.r.t. x
                 nsf[:,2,:] shape fucntions derivatives w.r.t. y
    """

    f, fx, fy = koornwinder2d(pts, porder)
    A, _, _ = koornwinder2d(plocal[:,1:3], porder)
    nf = np.linalg.solve(A.T, f.T)
    nfx = np.linalg.solve(A.T, fx.T)
    nfy = np.linalg.solve(A.T, fy.T)

    nfs = np.full((A.shape[0], 3, pts.shape[0]), 0.0)
    nfs[:,0,:] = nf
    nfs[:,1,:] = nfx
    nfs[:,2,:] = nfy

    return nfs


def koornwinder1d(x,p):
    """      
    koornwinder1d vandermonde matrix for legenedre polynomials in [0,1]
    [f,fx]=koornwinder(x,p)
 
       x:         coordinates of the points wherethe polynomials 
                  are to be evaluated (npoints)
       p:         maximum order of the polynomials consider. that
                  is all polynomials of degree up to p, npoly=p+1
       f:         vandermonde matrix (npoints,npoly)
       fx:        vandermonde matrix for the derivative of the koornwinder
                  polynomials w.r.t. x (npoints,npoly) 
    """
 
    x = 2*x - 1.0

    f = np.zeros((x.shape[0], p+1), dtype=float)
    fx = np.zeros((x.shape[0], p+1), dtype=float)

    for ii in range(p+1):
        pp = jacobi(ii,0,0)
        # Normalization factor to ensure integration to one
        pp = pp*math.sqrt(2*ii+1.0)
        dpp = np.polyder(pp)
        pval = np.polyval(pp, x)
        dpval = np.polyval(dpp, x)
        f[:,ii] = pval
        fx[:,ii] = dpval

    fx = 2*fx

    return f, fx

def koornwinder2d(x,p):
    """     
    koornwinder2d vandermonde matrix for koornwinder polynomials in 
               the master triangle [0,0]-[1,0]-[0,1]
    [f,fx,fy]=koornwinder(x,p)
 
       x:         coordinates of the points wherethe polynomials 
                  are to be evaluated (npoints,dim)
       p:         maximum order of the polynomials consider. that
                  is all polynomials of complete degree up to p,
                  npoly = (porder+1)*(porder+2)/2
       f:         vandermonde matrix (npoints,npoly)
       fx:        vandermonde matrix for the derivative of the koornwinder
                  polynomials w.r.t. x (npoints,npoly)
       fy:        vandermonde matrix for the derivative of the koornwinder
                  polynomials w.r.t. y (npoints,npoly)
    """
    x = 2*x - 1.0
    npol = ((p+1) * (p+2)) // 2
    f = np.zeros((x.shape[0], npol), dtype=float)
    fx = np.zeros((x.shape[0], npol), dtype=float)
    fy = np.zeros((x.shape[0], npol), dtype=float)

    pq = pascalindex(npol)
    
    xc = x
    xc[:,1] = np.minimum(0.99999999, xc[:,1]) # Displace coordinate for singular node

    e = np.full(xc.shape, 0.0)
    e[:,0] = 2*(1.0+xc[:,0]) / (1-xc[:,1]) - 1.0
    e[:,1] = xc[:,1]

    ii = np.where(x[:,1] == 1.0)
            # Correct values for function evaluation
    e[ii,0] = -1.0
    e[ii,1] =  1.0

    for ii in range(npol):
        pp = jacobi(pq[ii,0], 0,0)
        qp = jacobi(pq[ii,1], 2*pq[ii,0]+1, 0)
        for i in range(pq[ii,0]):
            qp = np.convolve([-0.5, 0.5], qp)

        pval = np.polyval(pp, e[:,0])
        qval = np.polyval(qp, e[:,1])

        fc = math.sqrt((2.0*pq[ii,0]+1.0)*2.0*(pq[ii,0]+pq[ii,1]+1.0))

        f[:,ii] = fc*pval*qval

    de1 = np.full(xc.shape, 0.0)
    de1[:,0] = 2.0 / (1-xc[:,1])
    de1[:,1] = 2*(1.0+xc[:,0]) / (1-xc[:,1])**2

    for ii in range(npol):
        pp = jacobi(pq[ii,0], 0,0)
        qp = jacobi(pq[ii,1], 2*pq[ii,0]+1, 0)
        for i in range(pq[ii,0]):
            qp = np.convolve([-0.5, 0.5], qp)

        dpp = np.polyder(pp)
        dqp = np.polyder(qp)

        pval = np.polyval(pp, e[:,0])
        qval = np.polyval(qp, e[:,1])

        dpval = np.polyval(dpp, e[:,0])
        dqval = np.polyval(dqp, e[:,1])

        fc = math.sqrt((2.0*pq[ii,0]+1.0)*2.0*(pq[ii,0]+pq[ii,1]+1.0))

        fx[:,ii] = fc*dpval*qval*de1[:,0]
        fy[:,ii] = fc*(dpval*qval*de1[:,1] + pval*dqval)

    fx = 2.0*fx
    fy = 2.0*fy

    return f, fx, fy

def pascalindex(npol):
    pq = np.full((npol,2), 0)
    l = 0
    for i in range(npol):
        for j in range(i+1):
            pq[l,0] = i-j
            pq[l,1] = j
            l = l + 1
            if l == npol: 
                return pq

    return pq


def gaussquad1d(pgauss):

    """     
    gaussquad1d calculates the gauss integration points in 1d for [0,1]
    [x,w]=gaussquad1d(pgauss)

      x:         coordinates of the integration points 
      w:         weights  
      pgauss:         order of the polynomila integrated exactly 
    """

    n = math.ceil((pgauss+1)/2)
    P = jacobi(n, 0, 0)
    x = np.sort(np.roots(P))

    A = np.zeros((n,n))
    for i in range(1,n+1):
        P = jacobi(i-1,0,0)
        A[i-1,:] = np.polyval(P,x)

    r = np.zeros((n,), dtype=float)
    r[0] = 2.0
    w = np.linalg.solve(A,r)

    x = (x + 1.0)/2.0
    w = w/2.0

    return x, w

def gaussquad2d(pgauss):

    """    
    gaussquad2d calculates the gauss integration points in 2d for [0,1]
    [x,w]=gaussquad2d(p)

      x:         coordinates of the integration points 
      w:         weights  
      pgauss:    order of the polynomila integrated exactly 
    """
    match pgauss:
        case 0, 1:
            w = np.array([
                5.00000000000000000E-01
            ])
            x = np.array([
                [3.33333333333333333E-01,  3.33333333333333333E-01]
            ])
        case 2:
            w = np.array([
                1.66666666666666666E-01,  1.66666666666666667E-01,  1.66666666666666667E-01
            ])
            x = np.array([
                [6.66666666666666667E-01,  1.66666666666666667E-01],
                [1.66666666666666667E-01,  6.66666666666666667E-01],
                [1.66666666666666667E-01,  1.66666666666666667E-01] 
            ])
        case 3:
            w = np.array([
                -2.81250000000000000E-01,  2.60416666666666667E-01, 2.60416666666666667E-01,  2.60416666666666666E-01 
            ])
            x = np.array([
                [3.33333333333333333E-01,  3.33333333333333333E-01],
                [6.00000000000000000E-01,  2.00000000000000000E-01],
                [2.00000000000000000E-01,  6.00000000000000000E-01],
                [2.00000000000000000E-01,  2.00000000000000000E-01]
            ])
        case 4:
            w = np.array([
                1.116907948390055E-01,  1.116907948390055E-01,  1.116907948390055E-01,  5.497587182766100E-02,
                5.497587182766100E-02,  5.497587182766100E-02 
            ])
            x = np.array([
                [1.081030181680700E-01,  4.459484909159650E-01],
                [4.459484909159650E-01,  1.081030181680700E-01],
                [4.459484909159650E-01,  4.459484909159650E-01],
                [8.168475729804590E-01,  9.157621350977100E-02],
                [9.157621350977100E-02,  8.168475729804590E-01],
                [9.157621350977100E-02,  9.157621350977100E-02] 
            ])
        case 5:
            w = np.array([
                1.12500000000000E-01,   6.61970763942530E-02,   6.61970763942530E-02,   6.61970763942530E-02,
                6.29695902724135E-02,   6.29695902724135E-02,   6.29695902724135E-02 
            ])
            x = np.array([
                [3.33333333333333E-01,  3.33333333333333E-01],
                [5.97158717897700E-02,  4.70142064105115E-01],
                [4.70142064105115E-01,  5.97158717897700E-02],
                [4.70142064105115E-01,  4.70142064105115E-01],
                [7.97426985353087E-01,  1.01286507323456E-01],
                [1.01286507323456E-01,  7.97426985353087E-01],
                [1.01286507323456E-01,  1.01286507323456E-01]
            ])
        case 6:
            w = np.array([
                5.83931378631895E-02,   5.83931378631895E-02,   5.83931378631895E-02,   2.54224531851035E-02,
                2.54224531851035E-02,   2.54224531851035E-02,   4.14255378091870E-02,   4.14255378091870E-02,
                4.14255378091870E-02,   4.14255378091870E-02,   4.14255378091870E-02,   4.14255378091870E-02 
            ])
            x = np.array([
                [5.01426509658179E-01,  2.49286745170910E-01],
                [2.49286745170910E-01,  5.01426509658179E-01],
                [2.49286745170910E-01,  2.49286745170910E-01],
                [8.73821971016996E-01,  6.30890144915020E-02],
                [6.30890144915020E-02,  8.73821971016996E-01],
                [6.30890144915020E-02,  6.30890144915020E-02],
                [5.31450498448170E-02,  3.10352451033784E-01],
                [6.36502499121399E-01,  5.31450498448170E-02],
                [3.10352451033784E-01,  6.36502499121399E-01],
                [5.31450498448170E-02,  6.36502499121399E-01],
                [6.36502499121399E-01,  3.10352451033784E-01],
                [3.10352451033784E-01,  5.31450498448170E-02]
            ])
        case 7:
            w = np.array([
                -7.47850222338410E-02,  8.78076287166040E-02,   8.78076287166040E-02,   8.78076287166040E-02,
                2.66736178044190E-02,   2.66736178044190E-02,   2.66736178044190E-02,   3.85568804451285E-02,
                3.85568804451285E-02,   3.85568804451285E-02,   3.85568804451285E-02,   3.85568804451285E-02,
                3.85568804451285E-02 
            ])
            x = np.array([
                [3.33333333333333E-01,  3.33333333333333E-01],
                [4.79308067841920E-01,  2.60345966079040E-01],
                [2.60345966079040E-01,  4.79308067841920E-01],
                [2.60345966079040E-01,  2.60345966079040E-01],
                [8.69739794195568E-01,  6.51301029022160E-02],
                [6.51301029022160E-02,  8.69739794195568E-01],
                [6.51301029022160E-02,  6.51301029022160E-02],
                [4.86903154253160E-02,  3.12865496004874E-01],
                [6.38444188569810E-01,  4.86903154253160E-02],
                [3.12865496004874E-01,  6.38444188569810E-01],
                [4.86903154253160E-02,  6.38444188569810E-01],
                [6.38444188569810E-01,  3.12865496004874E-01],
                [3.12865496004874E-01,  4.86903154253160E-02] 
            ])
        case 8:
            w = np.array([
                7.21578038388935E-02,   4.75458171336425E-02,   4.75458171336425E-02,   4.75458171336425E-02,
                5.16086852673590E-02,   5.16086852673590E-02,   5.16086852673590E-02,   1.62292488115990E-02,
                1.62292488115990E-02,   1.62292488115990E-02,   1.36151570872175E-02,   1.36151570872175E-02,
                1.36151570872175E-02,   1.36151570872175E-02,   1.36151570872175E-02,   1.36151570872175E-02 
            ])
            x = np.array([
                [3.33333333333333E-01,  3.33333333333333E-01],
                [8.14148234145540E-02,  4.59292588292723E-01],
                [4.59292588292723E-01,  8.14148234145540E-02],
                [4.59292588292723E-01,  4.59292588292723E-01],
                [6.58861384496480E-01,  1.70569307751760E-01],
                [1.70569307751760E-01,  6.58861384496480E-01],
                [1.70569307751760E-01,  1.70569307751760E-01],
                [8.98905543365938E-01,  5.05472283170310E-02],
                [5.05472283170310E-02,  8.98905543365938E-01],
                [5.05472283170310E-02,  5.05472283170310E-02],
                [8.39477740995800E-03,  2.63112829634638E-01],
                [7.28492392955404E-01,  8.39477740995800E-03],
                [2.63112829634638E-01,  7.28492392955404E-01],
                [8.39477740995800E-03,  7.28492392955404E-01],
                [7.28492392955404E-01,  2.63112829634638E-01],
                [2.63112829634638E-01,  8.39477740995800E-03] 
            ])
        case 9:
            w = np.array([
                4.85678981413995E-02,   1.56673501135695E-02,   1.56673501135695E-02,   1.56673501135695E-02,   
                3.89137705023870E-02,   3.89137705023870E-02,   3.89137705023870E-02,   3.98238694636050E-02,   
                3.98238694636050E-02,   3.98238694636050E-02,   1.27888378293490E-02,   1.27888378293490E-02,   
                1.27888378293490E-02,   2.16417696886445E-02,   2.16417696886445E-02,   2.16417696886445E-02,   
                2.16417696886445E-02,   2.16417696886445E-02,   2.16417696886445E-02 
            ])
            x = np.array([
                [3.33333333333333E-01,  3.33333333333333E-01],
                [2.06349616025250E-02,  4.89682519198738E-01],
                [4.89682519198738E-01,  2.06349616025250E-02],
                [4.89682519198738E-01,  4.89682519198738E-01],
                [1.25820817014127E-01,  4.37089591492937E-01],
                [4.37089591492937E-01,  1.25820817014127E-01],
                [4.37089591492937E-01,  4.37089591492937E-01],
                [6.23592928761935E-01,  1.88203535619033E-01],
                [1.88203535619033E-01,  6.23592928761935E-01],
                [1.88203535619033E-01,  1.88203535619033E-01],
                [9.10540973211095E-01,  4.47295133944530E-02],
                [4.47295133944530E-02,  9.10540973211095E-01],
                [4.47295133944530E-02,  4.47295133944530E-02],
                [3.68384120547360E-02,  2.21962989160766E-01],
                [7.41198598784498E-01,  3.68384120547360E-02],
                [2.21962989160766E-01,  7.41198598784498E-01],
                [3.68384120547360E-02,  7.41198598784498E-01],
                [7.41198598784498E-01,  2.21962989160766E-01],
                [2.21962989160766E-01,  3.68384120547360E-02]
            ])
        case 10:
            w = np.array([
                4.54089951913770E-02,   1.83629788782335E-02,   1.83629788782335E-02,   1.83629788782335E-02,   
                2.26605297177640E-02,   2.26605297177640E-02,   2.26605297177640E-02,   3.63789584227100E-02,   
                3.63789584227100E-02,   3.63789584227100E-02,   3.63789584227100E-02,   3.63789584227100E-02,   
                3.63789584227100E-02,   1.41636212655285E-02,   1.41636212655285E-02,   1.41636212655285E-02,   
                1.41636212655285E-02,   1.41636212655285E-02,   1.41636212655285E-02,   4.71083348186650E-03,   
                4.71083348186650E-03,   4.71083348186650E-03,   4.71083348186650E-03,   4.71083348186650E-03,   
                4.71083348186650E-03
            ])
            x = np.array([
                [3.33333333333333E-01,  3.33333333333333E-01],
                [2.88447332326850E-02,  4.85577633383657E-01],
                [4.85577633383657E-01,  2.88447332326850E-02],
                [4.85577633383657E-01,  4.85577633383657E-01],
                [7.81036849029926E-01,  1.09481575485037E-01],
                [1.09481575485037E-01,  7.81036849029926E-01],
                [1.09481575485037E-01,  1.09481575485037E-01],
                [1.41707219414880E-01,  3.07939838764121E-01],
                [5.50352941820999E-01,  1.41707219414880E-01],
                [3.07939838764121E-01,  5.50352941820999E-01],
                [1.41707219414880E-01,  5.50352941820999E-01],
                [5.50352941820999E-01,  3.07939838764121E-01],
                [3.07939838764121E-01,  1.41707219414880E-01],
                [2.50035347626860E-02,  2.46672560639903E-01],
                [7.28323904597411E-01,  2.50035347626860E-02],
                [2.46672560639903E-01,  7.28323904597411E-01],
                [2.50035347626860E-02,  7.28323904597411E-01],
                [7.28323904597411E-01,  2.46672560639903E-01],
                [2.46672560639903E-01,  2.50035347626860E-02],
                [9.54081540029900E-03,  6.68032510122000E-02],
                [9.23655933587500E-01,  9.54081540029900E-03],
                [6.68032510122000E-02,  9.23655933587500E-01],
                [9.54081540029900E-03,  9.23655933587500E-01],
                [9.23655933587500E-01,  6.68032510122000E-02],
                [6.68032510122000E-02,  9.54081540029900E-03] 
            ])
        case _:
            x1, w1 = gaussquad1d(pgauss)
            x1 = 2*x1 - 1
            w1 = 2*w1
            x2, y2 = np.meshgrid(x1, x1)
            xf = (1.0 + x2 - y2 - x2*y2).flatten()/4.0
            yf = (1.0 + y2).flatten()/2
            w0 = np.outer(w1, w1)
            w = (1-y2).flatten() * w0.flatten() / 8.0
            x = np.vstack([xf, yf]).T


    return x, w

if __name__ == "__main__":

    plocal = localpnts(5,1)

    print("plocal \n", plocal)
 