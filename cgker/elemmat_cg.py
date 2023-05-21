import numpy as np

def elemmat_cg(pcg, master, source, param):
    """
    elemmat_cg computes the elemental stiffness matrix and force vector.
   
        pcg:           element node coordinates
        mesh:         master structure
        source:       forcing function
        param:        kappa:   = diffusivity coefficient
                        c      = convective velocity
                        s      = source coefficient
        ae(npl,npl):  local element matrix (npl - nodes perelement)
        fe(npl):      local element force vector
    """

    kappa = param['kappa']
    c = param['c']
    s = param['s']

    npl = master.plocal.shape[0]

    ae = np.zeros((npl, npl), dtype=float)
    fe = np.zeros((npl, ), dtype=float)

    shap = master.shap[:,0,:]
    shapxi = master.shap[:,1,:]
    shapet = master.shap[:,2,:]

    xxi = pcg[:,0] @ shapxi
    xet = pcg[:,0] @ shapet
    yxi = pcg[:,1] @ shapxi
    yet = pcg[:,1] @ shapet

    jac = xxi * yet - xet * yxi

    shapx =   shapxi @ np.diag(yet) - shapet @ np.diag(yxi)
    shapy = - shapxi @ np.diag(xet) + shapet @ np.diag(xxi)

    Cx = shap @ np.diag(master.gwgh) @ shapx.T
    Cy = shap @ np.diag(master.gwgh) @ shapy.T
    K  = shapx @ np.diag(master.gwgh / jac) @ shapx.T + shapy @ np.diag(master.gwgh / jac) @ shapy.T
    M  = shap @ np.diag(master.gwgh * jac) @ shap.T

    Conv = -c[0]*Cx.T - c[1]*Cy.T

    ae = kappa*K + Conv + s*M

    if source:
        pg = shap.T @ pcg
        src = source(pg)
        fe = shap @ np.diag(master.gwgh * jac) @ src

    return ae, fe

