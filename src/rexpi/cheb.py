import numpy as np
#import scipy.linalg
from scipy.special import jv

def PositiveChebyshevNodes(n):
    """
    return the strictly positive entries from the 2n+1 Chebyshev nodes, sorted in ascending order
    """
    cheb_nodes_pos = np.cos((2*np.arange(n)+1.)/2/(2*n+1)*np.pi)
    return np.sort(cheb_nodes_pos)

def eval_ratfrompolchebyshev(x, omega, n):
    v = np.ones(np.shape(x))
    op = lambda v : x*v
    u = eval_polynomial_chebyshev(x, omega/2, n)
    return u/np.conjugate(u)

def eval_polynomial_chebyshev(x, t, n):
    
    """
    # Clenshaw Algorithm
    # polynomial Chebyshev approximation to y ~ exp(1j*t*x) 
    # we use p(A)*v ~ exp(1j*t*A)*v  
    # where op(v) = A*v applies operator on v,
    # the eigenvalues of A are assumed to be located in [-1,1]
    # m .. degree of p
    """
    v = np.ones(np.shape(x))
    op = lambda v : x*v

    cm1 = (1j)**n*jv(n,t)
    dkp1 = cm1 * v
    dkp2 = 0
    for k in range(n-1,-1,-1):
        ck = (1j)**k*jv(k,t)
        dk = ck * v + 2*op(dkp1) - dkp2
        if (k>0):
            dkp2 = dkp1
            dkp1 = dk
    return dk - dkp2

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