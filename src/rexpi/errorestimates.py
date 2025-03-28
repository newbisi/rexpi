import numpy as np
from scipy.special import lambertw

def errest(n, w, type=None, info=0):
    """
    a priori error estimate for unitary best approximation
    using an asymptotic estimate or an estimate based on experimental data
    Returns:
        error estimate for given n and w
    """
    if type=='asym':
        eout =  _errest_asym(n, w)
    elif type=='expe':
        eout = _errest_expe(n, w)
    else:
        erresta = _errest_asym(n, w)
        erreste = _errest_expe(n, w)
        # errest_asym tends to overestimate the error, so we take the min for the error objective
        eo = min(erresta,erreste)
        if (eo < 10**(-2/3*(n-4))):
            eout, type = erresta, 'asym'
        else:
            eout, type = erreste, 'expe'
    if info==1:
        return eout, {'type': type}
    else:
        return eout
def west(n, eo, type=None, info=0):
    """
    a priori estimate for frequency w (frequency can also be understood as a time-step)
    based on an asymptotic estimate or on experimental data
    Returns:
        frequency w s.t. the approximation to exp(iwx) of degree (n,n) attains the error objective eo
    """
    if type=='asym':
        wout = _west_asym(n, w)
    elif type=='expe':
        wout = _west_expe(n, w)
    else:
        if (eo < 10**(-2/3*(n-4))):
            wout, type = _west_asym(n, eo), 'asym'
        else:
            wout, type = _west_expe(n, eo), 'expe'
    if info==1:
        return wout, {'type': type}
    else:
        return wout
    
def nest(w, eo, type=None, info=0):
    """
    a priori estimate for degree n
    based on an asymptotic estimate or on experimental data
    Returns:
        n s.t. the approximation to exp(iwx) of degree (n,n) attains the error objective eo
    """
    if type=='asym':
        nout = _nest_asym(w, eo)
    elif type=='expe':
        nout = _nest_expe(w, eo)
    else:
        nesta = _nest_asym(w, eo)
        nestb = _nest_expe(w, eo)
        ntest = min(nesta,nestb)
        if (eo < 10**(-2/3*(ntest-4))):
            nout, type = nesta, 'asym'
        else:
            nout, type = nestb, 'expe'
    if info==1:
        return nout, {'type': type}
    else:
        return nout

# estimates based on asymptotic limits
def _errest_asym(n, w):
    """
    a version of errest(n, w) using asymptotic error estimate for the limit w->0
    Returns:
        error estimate for given n and w
    """
    nfacx = np.sum(np.log(np.arange(n+1,2*n+1)))
    efaclog = -2*nfacx-np.log(2*n+1)
    return 2*np.exp(efaclog+(2*n+1)*np.log(w/2))

def _west_asym(n, eo):
    """
    a version of west(n, err) based on an asymptotic estimate for the limit w->0
    Returns:
        frequency w s.t. the approximation to exp(iwx) of degree (n,n) attains the error objective eo
    """
    nfacx = np.sum(np.log2(np.arange(n+1,2*n+1)))
    return 2**((np.log2(eo*(2*n+1)/2)+2*nfacx)/(2*n+1)+1)
    
def _nest_asym(w, eo):
    """
    estimate for degree n based on an asymptotic estimate for the limit w->0
    Returns:
        n s.t. the approximation to exp(iwx) of degree (n,n) attains the error objective eo
    """
    m=-np.log(eo)/lambertw(-4*np.exp(-1)*np.log(eo/2)/w)
    return int(np.ceil((m.real-1)/2))

# estimates based on experimental data
def _west_expe(n, eo):
    """
    a version of west(n, err) based on experimental data
    Returns:
        w s.t. the unitary best approximant to exp(iwx) of degree (n,n) attains the error objective eo
    """
    if eo>1e-14:
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
                  0.7732573374862906,]
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
                  -0.9296235152950845,]
    else:
        # extrapolation
        coefsa = [-0.34960298585304206,
                  1.2653161350741573,]
        coefsb = [0.00028332004893961965,
                  -0.876285182160704,]

    polyxa = np.poly1d(coefsa)
    ause = polyxa(np.log(eo))
    polyxb = np.poly1d(coefsb)
    buse = polyxb(np.log(eo))
    return np.exp(-ause*n**buse)*(n+1)*np.pi
    
def _errest_expe(n, w):
    """
    a version of errest(n, w) based on experimental data
    Returns:
        errest s.t. errest and the given n and w satisfy w = _west_expe(n, errest)
    """
    if w>=(n+1)*np.pi:
        return 2
    ke=104
    for j in range(1,ke):
        errest = 2**(-j)
        wtest = _west_expe(n,errest)
        if w>wtest:
            w1, err1 = wtest, errest
            w2, err2 = _west_expe(n,2*errest), 2*errest
            for iruns in np.arange(20):
                step = (w-w1)/(w2-w1)
                loge1, loge2 = np.log10(err1), np.log10(err2)
                errnew = 10**(loge1+step*(loge2-loge1))
                wnew = _west_expe(n,errnew)
                if (abs(w-wnew)/wnew < 1e-12):
                    break
                if w < wnew:
                    w2, err2 = wnew, errnew
                else:
                    w1, err1 = wnew, errnew
            return(errnew)
            break
    return 2**(-ke)
    
def _nest_expe(w, eo):
    """
    estimate for degree n based on experimental data
    Returns:
        the smallest degree n s.t. _west_expe(n,eo) > w
    """
    info = 0
    if eo>=2:
        return 1
    # compute largest n for which we can attain w < (n+1)*np.pi for given w
    # n has to be at least that large
    n0f = w/np.pi-1
    n0 = max(1,int(np.ceil(n0f)))
    k=1000
    for n in range(n0,n0+k):
        wtest = _west_expe(n,eo)
        if w<wtest:
            return n
    return n0+k
