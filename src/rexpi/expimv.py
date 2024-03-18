import numpy as np
from scipy.special import jv

#######################################################################
#######################################################################
#######################################################################


##########################################################################
##########################################################################
##########################################################################

def chebyshev(op,t,v,m):
    """
    Clenshaw Algorithm
    polynomial chebychev approximation return p(A)*v ~ exp(1j*t*A)*v  
    op(v) = A*v applies operator on v,
    the eigenvalues of A are assumed to be located in [-1,1]
    m .. degree of p
    """

    cm1 = (1j)**m*jv(m,t)
    dkp1 = cm1 * v
    dkp2 = 0
    for k in range(m-1,-1,-1):
        ck = (1j)**k*jv(k,t)
        dk = ck * v + 2*op(dkp1) - dkp2
        if (k>0):
            dkp2 = dkp1
            dkp1 = dk
    return dk - dkp2

##########################################################################
##########################################################################
##########################################################################

def evalr_product(op, opinv, v, poles, c0):
    """
    c0 = 1 for unitary best approximation with approximation error<2
    best approximation for skew-Hermitian matrix A with spectrum in i*[-w,w] 
    compute (prod_j (A-conj(sj)*I)*inv(A-sj*I))*v
    for poles sj in C\R
    """
    n = len(poles)
    v2 = v.copy()
    for s in poles:
        v1 = opinv(s,v2)
        v2 = op(v1)+np.conj(s)*v1
    return c0*v2

   
##########################################################################
##########################################################################

def evalr_partialfraction(opinv, v, c0, coef, poles, v0=None):
    if v0 is None:
        v0 = v+0j
    y2 = c0*v0
    for (c,s) in zip(coef,poles):
        psi=opinv(s,v)
        y2 += c*psi
    return y2

   