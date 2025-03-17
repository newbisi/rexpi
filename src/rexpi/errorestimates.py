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
    nfacx = np.sum(np.log2(np.arange(n+1,2*n+1)))
    return 2**((np.log2(tol*(2*n+1)/2)+2*nfacx)/(2*n+1)+1)
    
def buerrest_getn(w, tol):
    """
    error estimate for best unitary approximation
    Returns:
        n s.t. the approximation to exp(iwx) of degree (n,n) has an error < tol
    """
    m=-np.log(tol)/lambertw(-4*np.exp(-1)*np.log(tol/2)/w)
    return int(np.ceil((m.real-1)/2))
    
def buerrest2_getw(n, tol):
    """
    estimate on frequency s.t. error is approximately equal to tolerance
    based on experimental data
    """
    if tol>1e-14:
        coefsa = [-7.121569685837123e-13,
                  -1.2386694533170105e-10,
                  -9.304919862544988e-09,
                  -3.9545380249348943e-07,
                  -1.0467338695044733e-05,
                  -0.0001792107115710027,
                  -0.0020017032381431966,
                  -0.014498935965331125,
                  -0.06860343132683391,
                  -0.5777408873924058,
                  0.7732573374862906,
                 ]
        coefsb = [-5.1523270545898124e-15,
                  -9.84139635152686e-13,
                  -8.237224551239085e-11,
                  -3.971747584379515e-09,
                  -1.2203258361585594e-07,
                  -2.4972665972026705e-06,
                  -3.4599720415307025e-05,
                  -0.00032440829161667405,
                  -0.0020382018252632795,
                  -0.00854706119111975,
                  -0.024713673601660883,
                  -0.9296235152950845,
                  ]
    else:
        # extrapolation
        coefsa = [-0.34960298585304206,
                  1.2653161350741573,]
        coefsb = [0.00028332004893961965,
                  -0.876285182160704,]

    polyxa = np.poly1d(coefsa)
    ause = polyxa(np.log(tol))
    polyxb = np.poly1d(coefsb)
    buse = polyxb(np.log(tol))
    return np.exp(-ause*n**buse)*(n+1)*np.pi
