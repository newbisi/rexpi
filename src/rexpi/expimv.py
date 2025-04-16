import numpy as np

##########################################################################
##########################################################################

def evalr_product_scalar(z, poles, r0=1):
    zs = np.asanyarray(z).ravel()
    mvconj = lambda b : zs*b
    mvSI = lambda s,b : b/(zs-s)
    k = len(zs)
    b = np.ones(k)
    y = evalr_product(mvconj, mvSI, b, poles, r0=r0)
    
    if np.isscalar(z): return y[0]
    y.shape = np.shape(z)
    return y
    
def evalr_product(op, opSI, v, poles, r0=1):
    """
    opconj: v -> conj(A)*v
    opSI: s, v -> inv(A-s)
    A is skew-Hermitian (spectrum resides on imaginary axis)
    best approximation for skew-Hermitian matrix A with spectrum in i*[-w,w] 
    compute (prod_j (A-conj(sj)*I)*inv(A-sj*I))*v
    for poles sj in C without imaginary axis
    assume r is symmetric s.t. constant factor is r0 = r(0)
    r0 = 1 for unitary best approximation with approximation error<2
    """
    n = len(poles)
    v2 = (-1)**n*r0*v
    for s in poles:
        v1 = opSI(s,v2)
        v2 = op(v1)+np.conj(s)*v1
    return v2

   
##########################################################################
##########################################################################

def evalr_partialfraction_scalar(z, coef, poles, rinf=None):
    zs = np.asanyarray(z).ravel()
    mvSI = lambda s,b : b/(zs-s)
    k = len(zs)
    b = np.ones(k)
    y = evalr_partialfraction(mvSI, b, coef, poles, rinf=rinf)
    if np.isscalar(z): return y[0]
    y.shape = np.shape(z)
    return y
    
def evalr_partialfraction(opinv, v, coef, poles, rinf=None, v0=None):
    if v0 is None:
        v0 = v+0j
    if rinf is None:
        n = len(poles)
        rinf = (-1)**n
    y2 = rinf*v0
    for (c,s) in zip(coef,poles):
        psi=opinv(s,v)
        y2 += c*psi
    return y2

   