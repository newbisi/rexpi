import numpy as np
from scipy.special import lambertw


def buerrest(n, w):
    """
    error estimate for best unitary approximation
    Returns:
        estimate for given n and w
    """
    nfacx = np.sum(np.log(np.arange(n+1,2*n+1)))
    efaclog = -2*nfacx-np.log(2*n+1)
    return 2*np.exp(efaclog+(2*n+1)*np.log(w/2))

def buerrest_getw(n, tol):
    """
    error estimate for best unitary approximation
    Returns:
        frequency w s.t. the approximation to exp(iwx) of degree (n,n) has an error < tol
    """
    nfacx = np.sum(np.log(np.arange(n+1,2*n+1)))
    efaclog = -2*nfacx-np.log(2*n+1)
    logtolh = np.log(tol/2)
    return 2*np.exp((logtolh-efaclog)/(2*n+1))

def buerrest_getn(w, tol):
    """
    error estimate for best unitary approximation
    Returns:
        n s.t. the approximation to exp(iwx) of degree (n,n) has an error < tol
    """
    
    m=-np.log(tol)/lambertw(-4*np.exp(-1)*np.log(tol/2)/w)
    return int(np.ceil((m.real-1)/2))
