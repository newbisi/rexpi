import numpy as np
import scipy.linalg

import gmpy2
import flamp

import logging

from .cheb import PositiveChebyshevNodes
from .errorestimates import west

class barycentricratfct():
    def __init__(self, y, alpha, beta):
        self.y, self.alpha, self.beta = y, alpha, beta
        
    def __call__(self, x, oniR = False):
        y, alpha, beta = self.y, self.alpha, self.beta
        if oniR:
            return evalr_unitary_oniR(x, y, beta)
        else:
            return evalr(x, y, alpha, beta)

    def getpoles(self,sym=False,withgenEig=True):
        y, wj = self.y, self.beta
        m = len(y)
        if (m<=1): return []
        if (wj.dtype=='complex128' and withgenEig):
            # with generalized eigenvalue problem, Kle12, NST18
            # this is only implemented for double numpy vectors
            if sym:
                return _getpoles_geneig_sym(y,wj)
            else:
                return _getpoles_geneig(y,wj)
        else:
            # with std eigenvalue problem, Kno08
            if sym:
                return _getpoles_stdeig_sym(y,wj)
            else:
                return _getpoles_stdeig(y,wj)

    def getpartialfractioncoef(self,sym=False, withint=False, usebform = False):
        '''
        compute coefficients and poles for partial fraction decomposition
        y(z) = a0 + sum_j aj/(z-sj)
        withint.. using contour integrals to compute coefficients aj
                  for partial fraction decomposition
        usebform.. use barycentric rational form
        sym.. compute poles with symmetrized eigenvalue problem
        '''
        y, alpha, beta = self.y, self.alpha, self.beta
        n = len(y)-1
        a0 = sum(alpha)/sum(beta)
        if (n<=0):
            return a0,[],[]
        sj = self.getpoles(sym)

        if withint:
            npts = 100
            pts = np.exp(1j*2*np.pi*np.arange(npts)/npts)
            aj = np.zeros(sj.size,dtype=np.cdouble)
            for j in range(n):
            	snow = sj[j]
            	ds = abs(sj[j]-sj)
            	ds[j] = max(ds)
            	d = min(ds)
            	ra = d/2
            	aj[j] = ra/npts * np.sum(self(sj[j]+ra*pts)*pts)
        elif usebform:
            one = y[0]*0 + 1
            C_pol = np.divide(one, sj[:,None] - y[None,:])
            N_pol = C_pol.dot(beta.conj())
            Ddiff_pol = (-C_pol**2).dot(beta)
            aj = -N_pol / Ddiff_pol
        else:
            P=sj[:,None]+sj.conj()[None,:]
            Q=sj[:,None]-sj[None,:]
            np.fill_diagonal(Q, 1)
            aj = (-1)**n*np.prod(P/Q,1)
        return a0, aj, sj

    def coef(self):
        '''
        return support nodes and coefficients of r in barycentric rational form
        '''
        return self.y, self.alpha, self.beta
def _use_flamp(x):
    if isinstance(x, gmpy2.mpc)|isinstance(x, gmpy2.mpfr): return True
    if isinstance(x, complex): return False
    return x.dtype == 'O'

def _nullspace(A):
    '''
    A .. numpy or flamp matrix
    return a vector in the nullspace of A
    '''
    if _use_flamp(A):
        Q, _ = flamp.qr(A.T, mode='full')
        return Q[:, -1].conj()
    else:
        Q, _ = scipy.linalg.qr(A.T, mode='full')
        return Q[:, -1].conj()
    # alternative option
    # [U,S,V]=np.linalg.svd(A,full_matrices=True)
    # gam = V[-1,:]
def _rotvecsLoewner(x):
    if _use_flamp(x):
        # gmpy2 expm1 not implemented for complex arguments
        #expm1 = np.vectorize(gmpy2.expm1, otypes=[object])
        Fmo = 1-flamp.exp(-1j*x) # 1 - F.conj()
        Fmo[Fmo == 0.0] = 1j
        Rr = (1-flamp.cos(x))/abs(Fmo)
        Ri = flamp.sin(x)/abs(Fmo)
        Rr[Fmo == 1j] = flamp.to_mp(0.0)
        Ri[Fmo == 1j] = flamp.to_mp(1.0)
        return Rr, Ri
    else:
        Fmo = -np.expm1(-1j*x) # 1 - F.conj()
        Fmo[Fmo == 0.0] = 1j
        R = Fmo/abs(Fmo)
        return R.real, R.imag

def _exp(z):
    if _use_flamp(z):
        return flamp.exp(z)
    else:
        return np.exp(z)
def _log(z):
    if _use_flamp(z):
        return flamp.log(z)
    else:
        return np.log(z)

_flampangle = np.vectorize(gmpy2.phase, otypes=[object])
def _angle(z):
    if _use_flamp(z):
        return _flampangle(z)
    else:
        return np.angle(z)
_real = np.vectorize(np.real)
_imag = np.vectorize(np.imag)
def _solve(A, b):
    if  _use_flamp(A):
        return flamp.lu_solve(A, b)
    else:
        return np.linalg.solve(A, b)
def _eigvals(A):
    """
    return eigenvalues of a matrix A
    """
    if _use_flamp(A):
        return flamp.eig(A, left=False, right=False)
    else:
        return scipy.linalg.eigvals(A)
        
#######################################

def _getpoles_geneig(y, wj):
    """
    Klein thesis 2012 and NST18
    """
    m = len(y)
    B = np.eye(m+1)
    B[0,0] = 0
    E = np.block([[0, wj],
                [np.ones([m,1]), np.diag(y)]])
    lam = scipy.linalg.eigvals(E, B)
    lam = lam[np.isfinite(lam)]
    return lam
def _getpoles_geneig_sym(y, b):
    """
    symmetrized version of generalized eigenvalue problem
    """
    m = len(y)
    ij=np.argsort(abs(y.imag))
    b, y = b[ij], y[ij]

    B = np.eye(m+1)
    B[0,0] = 0

    M= np.zeros((m+1,m+1))
    M[1:,0]=1
    M[0,1:]=(b.real+b.imag)
    
    yp = y[m%2:]
    n2 = m-m%2
    ij=np.zeros(n2,dtype=np.int32)
    ij[::2]=np.arange(1,n2,2)
    ij[1::2]=np.arange(0,n2,2)
    ix=(m+1)*(np.arange(n2)+m%2+1)+m%2+1
    
    K = M.reshape((m+1)**2)
    K[ix+ij] += -yp.imag

    lam = scipy.linalg.eigvals(M, B)
    lam = lam[np.isfinite(lam)]
    return lam
    
def _getpoles_stdeig(y, wj):
    """
    compute poles using the approach of Kno08, with standard eigenvalue problem
    """
    ak = wj / (wj.sum())
    M = np.diag(y) - np.outer(ak, y)
    lam = _eigvals(M)
    lam = np.delete(lam, np.argmin(abs(lam)))
    return lam
def _getpoles_stdeig_sym(y, wj):
    """
    symmetrized version of _getpoles_stdeig
    can be used with (flamp datatypes)
    """

    m, n = len(y), len(y)-1
    n2=m-m%2
    
    ij=np.argsort(abs(_imag(y)))
    y=y[ij]
    ak = wj[ij] / (wj.sum().real)
    
    y=y[m%2:]
    ak=ak[m%2:]
    ij=np.zeros(n2,dtype=np.int32)
    ij[::2]=np.arange(1,n2,2)
    ij[1::2]=np.arange(0,n2,2)
    ix=n2*np.arange(n2)
    
    M = -np.outer(_real(ak)-_imag(ak), _imag(y))
    K = M.reshape(n2*n2)
    K[ix+ij] += _imag(y[ij])

    lam = _eigvals(M)
    if n%2==1:
        lam = np.delete(lam, np.argmin(abs(lam)))
    return lam

def interpolate_std(allnodes, omega=1):
    """
    allnodes are 2n+1 distinct nodes on the real axis
    interpolates exp(1j*x) for x <- allnodes
    """
    y = allnodes[::2]
    xs = allnodes[1::2]
    F = _exp(1j*omega*xs)
    f = _exp(1j*omega*y)
    C = 1./(xs[:,None]-y)
    L = F[:,None]*C-C*f[None,:]
    wjs = _nullspace(L)
    n=len(y)-1
    if n%2==1: wjs*=1j
    alpha = f*wjs
    return barycentricratfct(1j*y,alpha,wjs)

def interpolate_unitary(allnodes, omega=1):
    """
    allnodes are 2n+1 distinct nodes on the real axis
    interpolates exp(1j*omega*x) for x <- allnodes
    from allnodes nodes choose
    n nodes as 'test nodes' x
    n+1 nodes as 'support nodes' y
    interpolant r has degree n
    """
    y=allnodes[::2]
    xs=allnodes[1::2]
    
    # nodes_pos are all strictly positive nodes
    n = len(y)-1 # total number of nodes is 2n+1 <- always odd!!

    one = xs[0]*0+1
    C = one/(xs[:,None]-y[None,:])

    Rr, Ri = _rotvecsLoewner(omega*xs)
    Kr, Ki = _rotvecsLoewner(omega*y)
    
    A = Ri[:,None]*C*Kr[None,:] - Rr[:,None]*C*Ki[None,:]
    v0 = _nullspace(A)
    b = (-Ki+1j*Kr)*v0 # iKw
    if n%2==1: b*=1j
    alpha = (-1)**n*b.conj()
    return barycentricratfct(1j*y, alpha, b)

def interpolate_unitarysym(nodes_pos, omega=1):
    """
    nodes_pos are n strictly positive, distinct and real-valued nodes, total number of nodes is 2n+1 <- always odd!!
    thus, zero is always in the set of interpolation nodes but not in nodes_pos
    """
    n = len(nodes_pos)
    zero = 0*nodes_pos[0]

    # of 2n+1 nodes, n+1 are support nodes
    ys_pos = nodes_pos[(n+1)%2::2]
    # of 2n+1 nodes, n are test nodes
    xs_pos = nodes_pos[n%2::2]
    
    Rr, Ri = _rotvecsLoewner(omega*xs_pos)
    Kr, Ki = _rotvecsLoewner(omega*ys_pos)
    
    C2 = 1./(xs_pos[:,None]-ys_pos[None,:])
    C2m = 1./(xs_pos[:,None]+ys_pos[None,:])
    
    B1 = Ri[:,None]*C2*Kr[None,:] - Rr[:,None]*C2*Ki[None,:]
    B2 = Ri[:,None]*C2m*Kr[None,:] + Rr[:,None]*C2m*Ki[None,:]
    if ((n+1)%2 == 1):
        b0 = (Rr/xs_pos)[:,None] # no minus sign needed here since we also remove minus below
        B = np.concatenate((b0, B1 - B2), axis=1)
    else:
        B = B1 + B2
    v0 = _nullspace(B) # nullspace
    if ((n+1)%2 == 1):
        # n%2==0
        # Bm has k x k+1 dimensional and always has non-trivial nullspace, no need to look at Bp !!
        # b2sub = 1j*Kz*v0
        b2sub = (1j*Kr*v0[1:] - Ki*v0[1:])
        b = np.concatenate(([v0[0]], b2sub, b2sub.conj()))
        y = np.concatenate(([zero], ys_pos, -ys_pos))
    else:
        # Bp has k x k+1 dimensional and always has non-trivial nullspace, no need to look at Bm !!
        # b2sub = 1j*Kz*v0         #b = 1j*np.concatenate((b2sub, -b2sub.conj()))
        b2sub = -(Kr*v0 + 1j*Ki*v0)
        b = np.concatenate((b2sub, b2sub.conj()))
        y = np.concatenate((ys_pos, -ys_pos))
    alpha = (-1)**n*b.conj()
    return barycentricratfct(1j*y, alpha, b)

    
def evalr(z, zj, alpha, beta):
    """
    evaluate z -> r(z) for weights alpha, beta
    or only beta and alpha = (-1)**(m-1)*beta.conj() for the unitary casy r=d*/d 
    """
    m = len(zj)

    zv = np.asanyarray(z).ravel()
    D = zv[:,None] - zj[None,:]
    
    # find indices where x is exactly on a node
    (node_xi, node_zi) = np.nonzero(D == 0)
    one = zj[0]*0+1.0
    D[node_xi, node_zi] = one

    # beta is scaled by 1j for odd n=m-1
    #alpha = (-1)**(m-1)*beta.conj()
    
    C = np.divide(one, D)
    numx = C.dot(alpha)
    denomx = C.dot(beta)
    r = numx / denomx
    
    r[node_xi] = alpha[node_zi]/beta[node_zi]

    if np.isscalar(z): return r[0]
    r.shape = np.shape(z)
    return r
    
def evalr_unitary_oniR(x, zj, wj):
    """
    evaluate x -> r(ix) for unitary r=d*/d with weights alpha = (-1)**(m-1)*beta*
    currently only works for numpy standard dtypes
    """
    m=len(zj)
    xv = np.asanyarray(x).ravel()
    xj = zj.imag
    D = xv[:,None] - xj[None,:]
    one = 1.0
    
    # find indices where x is exactly on a node
    (node_xi, node_zi) = np.nonzero(D == 0)
    # set divisor to 1 to avoid division by zero
    D[node_xi, node_zi] = one
    
    #with np.errstate(divide='ignore', invalid='ignore'):
    #if len(node_xi) == 0:       # no zero divisors
    C = np.divide(one, D)
    denomx = C.dot(wj)
    r = (-1)**(m-1)*np.conj(denomx) / denomx

    # fix evaluation at support nodes
    r[node_xi] = (-1)**(m-1)*np.conj(wj[node_zi])/wj[node_zi]

    if np.isscalar(x): return r[0]
    r.shape = np.shape(x)
    return r


def riCheb(w, n, syminterp=True):
    """
    compute (n,n) rational function which interpolates exp(1j*w*x) at 2n+1 Chebyshev nodes
    """
    cheb_nodes_pos = PositiveChebyshevNodes(n)
    if syminterp:
        return interpolate_unitarysym(cheb_nodes_pos, w)
    else:
        return interpolate_unitary(cheb_nodes_pos, w)
        
#######################################

def brib(w=10.0, n=6, nodes_pos=None, syminterp=True,
         maxiter=None, tolequi=1e-3,
         info=0,
         tolstagnation=0, # if devation is smaller than this tol and stagnates, break iteration
         kstag=10,
         tolerr=0, #  if error is smaller tolerr and phis are alternating
         optcorrection=None, # adapt nodes
         npi=None, npigolden=-20, npisample=9,
         Maehlylog=0.05, useDunhamformula = 1,
         step_factor=2.2, max_step_size=0.1, hstep=0.1):
    """
    best rational interpolation based (brib) approximation, r(ix) \approx exp(iwx)
    n .. compute (n,n) unitary rational best approximation
    tol .. stop the iteration 
    y or n, mirrored nodes, flat, sorted
    w or tol, tol only used to specify w if needed, no stopping criteria
    nodes_pos .. initial interpolation nodes, assuming nodes mirrored around zero, only needs nodes>0
    info[0] ..  INFO = 0: successful termination
                INFO < 0: illegal value of one or more arguments -- no computation performed
                INFO > 0: failure in the course of computation 
    """
    success = 1
    ###### set parameters
    if (w >= np.pi*(n+1)):
        # return constant one function
        logging.warning("for w>=(n+1)pi = %f the best approximartion is trivial" % (n+1)*np.pi)
        allpoints=[[0.0],[0.0]]
        allerr = [2.0]
        success = 0
        return barycentricratfct([0.0],[1.0],[1.0]), [success, allpoints ,allerr]
        
    if nodes_pos is None:
        xi = w/(n+1)/np.pi
        nodes_pos = (1-xi)*PositiveChebyshevNodes(n) + xi*np.arange(1,n+1)/(n+1)
    else:
        nodes_pos = np.sort(nodes_pos[nodes_pos>0].flatten())
        n = len(nodes_pos)
    if maxiter is None: maxiter = 4*n
    if npi is not None: npiuse = npi
    ######
    errors = []
    stepsize = np.nan
    devstagnation = False
    besterr = 2
    besterrdev = 1
    besterrnodes = []
    max_err = 2
    phisignsum = n
    lastusedcorrection = 0
    zero = 0*nodes_pos[0]
    one = zero + 1
    for ni in range(maxiter):
        #### rational interpolation
        if syminterp:
            r = interpolate_unitarysym(nodes_pos, omega=w)
        else:
            nodes_all = np.concatenate((-nodes_pos[::-1], [zero], nodes_pos))
            r = interpolate_unitary(nodes_all, omega=w)

        nodes_sub = np.concatenate(([zero], nodes_pos, [one]))
        errfun = lambda x: abs( r(1j*x) - _exp(1j*w*x) )

        ##### find nodes with largest error
        if npi is None:
            npiuse = npigolden if ((max_err<2)&(phisignsum==0)) else npisample
        
        if npiuse > 0:
            local_max_x, local_max = local_maxima_sample(errfun, nodes_sub, npiuse)
        else:
            local_max_x, local_max = local_maxima_golden(errfun, nodes_sub, num_iter=-npiuse)
        
        phisignsumold = phisignsum
        
        phisignsum = _checkangles(r,w,local_max_x)
        if ((npi is None)&((phisignsumold>0)&(phisignsum==0))):
            npiuse = npisample+5
            local_max_x, local_max = local_maxima_sample(errfun, nodes_sub, npiuse)
            phisignsum = _checkangles(r,w,local_max_x)
        
        max_err = local_max.max()
        deviation = 1 - local_max.min() / max_err

        if ((tolstagnation>0)&(max_err<besterr)):
            besterr = max_err
            besterrdev = deviation
            besterrnodes = nodes_pos
        
        errors.append((max_err, deviation, lastusedcorrection, npiuse, phisignsum))

        #### test convergence
        converged = (((deviation <= tolequi)&(phisignsum==0))&(max_err<2))
        if converged:
            success = 0
            break
        if ni == maxiter-1:
            break

        errsmall = ((phisignsum==0)&(max_err<tolerr))
        if errsmall:
            success = 0
            break

        #### interpolation nodes correction
        # if not at last iteration, apply interpolation nodes correction
        if optcorrection is None:
            if ((max_err==2)|(phisignsum!=0)):
                nodes_pos, stepsize = _correction_BRASIL(nodes_sub, local_max, max_step_size, step_factor)
                lastusedcorrection=1
            else:
                optc = 1 if (deviation<Maehlylog) else 0
                if useDunhamformula:
                    nodes_pos = _correction_MaehlyDunham_sym(nodes_pos, local_max_x, local_max, optc)
                else:
                    nodes_pos = _correction_Maehly_sym(nodes_pos, local_max_x, local_max, optc)
                lastusedcorrection = 3.0 + 0.1*optc + 0.01*useDunhamformula
                if np.any(np.diff(nodes_pos)<0):
                    # double check if nodes are in ascending order
                    nodes_pos = np.sort(nodes_pos)
        else:
            lastusedcorrection = optcorrection
            if optcorrection==1:
                nodes_pos, stepsize = _correction_BRASIL(nodes_sub, local_max, max_step_size, step_factor)
            elif optcorrection==2:
                nodes_pos = _correction_Franke(nodes_pos, local_max_x, local_max, hstep)
            else:
                if useDunhamformula:
                    nodes_pos = _correction_MaehlyDunham_sym(nodes_pos, local_max_x, local_max)
                else:
                    nodes_pos = _correction_Maehly_sym(nodes_pos, local_max_x, local_max)
                if np.any(np.diff(nodes_pos)<0):
                    # make sure nodes are in ascending order
                    nodes_pos = np.sort(nodes_pos)
        if max(nodes_pos)>1:
            logging.warning("step %d, opt %f. max interpolation node > 1, %f" % (ni,lastusedcorrection,max(nodes_pos)))
            # fix nodes or terminate algorithm
            success = 1
            break
        
        # test for stagnation and stop loop if stagnation was detected in the previous iteration
        if devstagnation:
            logging.warning("iter %d. deviation stagnated at delta = %.2e before reaching tol" % (ni,deviation))
            break
        if ((ni>kstag)&(tolstagnation>0)): devstagnation = (((sum(devs[-kstag:])/kstag)<deviation)&(besterrdev<tolstagnation))
        if devstagnation: nodes_pos = besterrnodes

    #### output
    if info==0: return r

    niter = len(errors)
    errors = np.array(errors)
    infoout = {'success': success, 'err': errors[-1,0], 'dev': errors[-1,1], 'iterations': niter,
               'ix': nodes_pos, 'eta': local_max_x,
               'errors': errors[:,0], 'deviations': errors[:,1], 'phisignerr': errors[:,4],
               'optc': errors[:,2], 'npi': errors[:,3]}
    return r, infoout

def _checkangles(r,w,x):
    # return number of points for which the phase error has 'wrong' sign
    # sign should be negative at last entries and alternating from there
    phasefctexp = r(1j*x)/_exp(1j*w*x)
    k = len(x) # k = n+1
    local_max_imag = _imag(phasefctexp)
    return np.sum(local_max_imag*(-1)**(k-np.arange(k)) < 0)
    
def _correction_BRASIL(nodes_sub, local_max, max_step_size, step_factor):
    # subroutine motivated by
    # C. Hofreither. An algorithm for best rational approximation based on barycentric rational interpolation.
    # Numerical Algorithms, 88(1):365--388, 2021. doi 10.1007/s11075-020-01042-0
    n = len(nodes_sub)
    intv_lengths = np.diff(nodes_sub)

    mean_err = np.mean(local_max)
    max_dev = abs(local_max - mean_err).max()
    normalized_dev = (local_max - mean_err) / max_dev
    stepsize = min(max_step_size, step_factor * max_dev / (n * mean_err))
    scaling = (1.0 - stepsize)**normalized_dev

    intv_lengths *= scaling
    # rescale so that they add up to b-a again
    intv_lengths /= intv_lengths.sum()
    nodes_pos = np.cumsum(intv_lengths)[:-1]
    return nodes_pos, stepsize
def _correction_Franke(nodes_pos, local_max_x, local_max, hstep):
    dz = np.diff(local_max)*np.diff(local_max_x)/(np.sum(local_max)/len(local_max))
    nodes_pos += hstep*dz
    return nodes_pos
def _correction_Maehly(nodes_pos, local_max_x, local_max, useln=1):
    n=len(nodes_pos)
    zero = 0*nodes_pos[0]
    alleps = np.concatenate((local_max[::-1],local_max))
    allx = np.concatenate((-local_max_x[::-1], local_max_x))
    ally = np.concatenate((-nodes_pos[::-1], [zero], nodes_pos))
    if useln==1:
        b = _log(alleps[1:]/alleps[0])
    else:
        b = 2*(alleps[1:]-alleps[0])/(alleps[1:]+alleps[0])
    M = (allx[0]-allx[1:,None])/((allx[1:,None]-ally)*(allx[0]-ally))
    dz = _solve(M, b)
    return nodes_pos+dz[n+1:]
def _correction_Maehly_sym(xs, local_max_x, local_max, useln=1):
    n=len(xs)
    etas, eta0 = local_max_x[:-1,None], local_max_x[-1]
    ers, er0 = local_max[:-1], local_max[-1]
    if useln==1:
        b = _log(ers/er0)
    else:
        b = 2*(ers-er0)/(ers+er0)
    M = 2*xs*(etas**2-eta0**2)/((xs**2-eta0**2)*(etas**2-xs**2))
    dxs = _solve(M, b)
    return xs + dxs
def _correction_MaehlyDunham(nodes_pos, local_max_x, local_max, useln=1):
    n=len(nodes_pos)
    zero = 0*nodes_pos[0]
    alleps = np.concatenate((local_max[::-1],local_max))
    etas = np.concatenate((-local_max_x[::-1], local_max_x))
    xs = np.concatenate((-nodes_pos[::-1], [zero], nodes_pos))

    epsgeom = _exp(np.mean(_log(alleps)))
    if useln==1:
        b = _log(alleps/epsgeom)
    else:
        b = 2*(alleps - epsgeom)/(alleps + epsgeom)
        
    Ex = xs[:,None]-etas
    Xpx = xs[:,None]-xs+np.eye(2*n+1)
    dzp1 = np.prod(Ex[:,:-1]/Xpx, 1)*Ex[:,-1]

    Xe = etas[:,None]-xs
    Epe = etas[:,None]-etas+np.eye(2*n+2)
    Ms = 1/(xs[:,None]-etas)*(b*np.prod(Xe/Epe[:,:-1],1)/Epe[:,-1])
    dzp2 = np.sum(Ms,1)
    dz = dzp1*dzp2
    return nodes_pos+dz[n+1:]
def _correction_MaehlyDunham_sym(nodes_pos, local_max_x, local_max, useln=1):
    n=len(nodes_pos)
    epsgeom = _exp(np.mean(_log(local_max)))
    if useln==1:
        b = _log(local_max/epsgeom)
    else:
        b = 2*(local_max - epsgeom)/(local_max + epsgeom)
        
    PA = nodes_pos[:,None]**2 - local_max_x[:-1]**2
    PA /= nodes_pos[:,None]**2 - nodes_pos**2 + 2*np.diag(nodes_pos**2)
    dz = np.prod(PA, 1)*(nodes_pos**2 - local_max_x[-1]**2)
    
    PA = 1/(local_max_x[:,None]**2 - local_max_x**2 + np.eye(n+1))
    PA[:,:-1] *= local_max_x[:,None]**2 - nodes_pos**2
    prodv = np.prod(PA,1)
    Ms = (b*prodv) / (nodes_pos[:,None]**2-local_max_x**2)
    dz *= nodes_pos*np.sum(Ms,1)
    return nodes_pos+dz
def _piecewise_mesh(nodes, n):
    # subroutine from https://github.com/c-f-h/baryrat
    # C. Hofreither. An algorithm for best rational approximation based on barycentric rational interpolation.
    # Numerical Algorithms, 88(1):365--388, 2021. doi 10.1007/s11075-020-01042-0
    """Build a mesh over an interval with subintervals described by the array
    ``nodes``. Each subinterval has ``n`` points spaced uniformly between the
    two neighboring nodes.  The final mesh has ``(len(nodes) - 1) * n`` points.
    """
    #z = np.concatenate(([z0], nodes, [z1]))
    M = len(nodes)
    return np.concatenate(tuple(
        np.linspace(nodes[i], nodes[i+1], n, endpoint=(i==M-2))
        for i in range(M - 1)))

def local_maxima_golden(g, nodes, num_iter):
    # subroutine from https://github.com/c-f-h/baryrat
    # C. Hofreither. An algorithm for best rational approximation based on barycentric rational interpolation.
    # Numerical Algorithms, 88(1):365--388, 2021. doi 10.1007/s11075-020-01042-0
    # vectorized version of golden section search
    # removed boundary search at first interval due to symmetry properties
    golden_mean = (3.0 - np.sqrt(5.0)) / 2   # 0.381966...
    L, R = nodes[0:-2], nodes[1:-1]     # skip right-hand side boundary interval (treated below)
    # compute 3 x m array of endpoints and midpoints
    z = np.vstack((L, L + (R-L)*golden_mean, R))
    m = z.shape[1]
    all_m = np.arange(m)
    gB = g(z[1])

    for k in range(num_iter):
        # z[1] = midpoints
        mids = (z[0] + z[2]) / 2

        # compute new nodes according to golden section
        farther_idx = (z[1] <= mids).astype(int) * 2 # either 0 or 2
        X = z[1] + golden_mean * (z[farther_idx, all_m] - z[1])
        gX = g(X)

        for j in range(m):
            x = X[j]
            gx = gX[j]

            b = z[1,j]
            if gx > gB[j]:
                if x > b:
                    z[0,j] = z[1,j]
                else:
                    z[2,j] = z[1,j]
                z[1,j] = x
                gB[j] = gx
            else:
                if x < b:
                    z[0,j] = x
                else:
                    z[2,j] = x

    # prepare output arrays
    Z, gZ = np.empty(m+1, dtype=z.dtype), np.empty(m+1, dtype=gB.dtype)
    Z[:-1] = z[1, :]
    gZ[:-1] = gB
    # treat the boundary intervals specially since usually the maximum is at the boundary
    # (no bracket available!)
    #Z[0], gZ[0] = _boundary_search(g, nodes[0], nodes[1], num_iter=3)
    Z[-1], gZ[-1] = _boundary_search(g, nodes[-2], nodes[-1], num_iter=3)
    return Z, gZ

def _boundary_search(g, a, c, num_iter):
    # subroutine from https://github.com/c-f-h/baryrat
    # C. Hofreither. An algorithm for best rational approximation based on barycentric rational interpolation.
    # Numerical Algorithms, 88(1):365--388, 2021. doi 10.1007/s11075-020-01042-0
    X = [a, c]
    Xvals = [g(a), g(c)]
    max_side = 0 if (Xvals[0] >= Xvals[1]) else 1
    other_side = 1 - max_side

    for k in range(num_iter):
        xm = (X[0] + X[1]) / 2
        gm = g(xm)
        if gm < Xvals[max_side]:
            # no new maximum found; shrink interval and iterate
            X[other_side] = xm
            Xvals[other_side] = gm
        else:
            # found a bracket for the minimum
            return _golden_search(g, X[0], X[1], num_iter=num_iter-k)
    return X[max_side], Xvals[max_side]

def _golden_search(g, a, c, num_iter=20):
    # subroutine from https://github.com/c-f-h/baryrat
    # C. Hofreither. An algorithm for best rational approximation based on barycentric rational interpolation.
    # Numerical Algorithms, 88(1):365--388, 2021. doi 10.1007/s11075-020-01042-0
    golden_mean = 0.5 * (3.0 - np.sqrt(5.0))

    b = (a + c) / 2
    gb = g(b)
    ga, gc = g(a), g(c)
    if not (gb >= ga and gb >= gc):
        # not bracketed - maximum may be at the boundary
        return _boundary_search(g, a, c, num_iter)
    for k in range(num_iter):
        mid = (a + c) / 2
        if b > mid:
            x = b + golden_mean * (a - b)
        else:
            x = b + golden_mean * (c - b)
        gx = g(x)

        if gx > gb:
            # found a larger point, use it as center
            if x > b:
                a = b
            else:
                c = b
            b = x
            gb = gx
        else:
            # point is smaller, use it as boundary
            if x < b:
                a = x
            else:
                c = x
    return b, gb

def local_maxima_sample(g, nodes, N):
    # subroutine from https://github.com/c-f-h/baryrat
    # C. Hofreither. An algorithm for best rational approximation based on barycentric rational interpolation.
    # Numerical Algorithms, 88(1):365--388, 2021. doi 10.1007/s11075-020-01042-0
    Z = _piecewise_mesh(nodes, N).reshape((-1, N))
    vals = g(Z)
    maxk = vals.argmax(axis=1)
    nn = np.arange(Z.shape[0])
    return Z[nn, maxk], vals[nn, maxk]

def brib_err(n, errob, errsandwich=1,
             tolequi=1e-3, tolstagnation=0, kstag=10,
             kiterations=15):
    """
    Input: degree n and error objective errob
    Returns: an approximant with an error and relative deviation which sandwiche the error objective errob as in
            (1-tolequi)*error < errob < error
    """
    goal = np.log10(errob)
    wa = west(n,errob)
    r, info = brib(wa,n,info=1, tolequi=tolequi, tolstagnation=tolstagnation, kstag=kstag)
    erra = (1-0.5*info['dev'])*info['err'] if (errsandwich) else info['err']
    
    rescale = erra/errob
    eo2=errob/rescale
    wnew = west(n,eo2)
    r, info = brib(wnew,n,info=1, tolequi=tolequi, tolstagnation=tolstagnation, kstag=kstag)
    errb = (1-0.5*info['dev'])*info['err'] if (errsandwich) else info['err']
    
    if wa<wnew:
        w1, err1 = wa, erra
        w2, err2 = wnew, errb
    else:
        w1, err1 = wnew, errb 
        w2, err2 = wa, erra  
    
    for iruns in np.arange(kiterations):
        loge1 = np.log10(err1)
        loge2 = np.log10(err2)
        step = (goal-loge1)/(loge2-loge1)
        wnew = w1+step*(w2-w1)
        r, info = brib(wnew,n,info=1, tolequi=tolequi, tolstagnation=tolstagnation, kstag=kstag)
        errl = (1-info['dev'])*info['err']
        erru = info['err']
        errnew = (1-0.5*info['dev'])*info['err'] if () else info['err']

        condsandwich = ((errsandwich) & ((errl<errob)&(errob<erru)))
        condnotsandwich = ((errsandwich!=1) & (abs(errnew-errob)/errnew<tolequi))
        if (condsandwich) or (condnotsandwich):
            success = 0
            return r, success, wnew, info
            break

        if wnew>w2:
            w1, err1 = w2, err2
            w2, err2 = wnew, errnew
        elif wnew<w1:
            w2, err2 = w1, err1
            w1, err1 = wnew, errnew
        else:
            if errob < errnew:
                w2, err2 = wnew, errnew
            else:
                w1, err1 = wnew, errnew
    success = 1
    return r, success, wnew, info
