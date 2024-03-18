import numpy as np
import scipy.linalg

import flamp

class barycentricratfct():
    def __init__(self, y, b, alpha = None):
        self.y, self.beta = y, b
        self.alpha = alpha
    def __call__(self, x, usesym = False):
        y, beta, alpha = self.y, self.beta, self.alpha
        if alpha is None:
            if usesym:
                return _evalr_unisym(x, y, beta)
            else:
                return evalr_unitary(x, y, beta)
        else:
            return evalr_std(x, y, alpha, beta)

    def getpoles(self,sym=False,withgenEig=True):
        y, wj = self.y, self.beta
        m = len(y)
        if (m<=1):
            return []
        if (wj.dtype=='complex128' and withgenEig):
            # with generalized eigenvalue problem, Kle12, NST18
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
        y, beta = self.y, self.beta
        n = len(y)-1
        a0 = np.conj(sum(beta))/sum(beta)
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
        if self.alpha is None:
            return self.y, self.beta
        else:
            return self.y, self.beta, self.alpha
            
def _mpc_to_numpy(a):
    return np.array([float(z.real)+1j*float(z.imag) for z in a])
def _nullspace(A):
    '''
    A .. numpy or flamp matrix
    return a vector in the nullspace of A
    '''
    
    if A.dtype=='O':
        Q, _ = flamp.qr(A.T, mode='full')
        return Q[:, -1].conj()
    else:
        #[_,_,V]=np.linalg.svd(A,full_matrices=True)
        #return V[-1,:]
        Q, _ = scipy.linalg.qr(A.T, mode='full')
        return Q[:, -1].conj()
def _rotvecsLoewner(x):
    if x.dtype=='O':
        # gmpy2 expm1 not implemented for complex arguments
        #expm1 = np.vectorize(gmpy2.expm1, otypes=[object])
        Fmo = -_expm1(-1j*x) # 1 - F.conj()
        Fmo[Fmo == 0.0] = 1j
        Rr = (1-flamp.cos(x))/abs(Fmo)
        Ri = flamp.sin(x)/abs(Fmo)
        Rr[Fmo == 1j] = flamp.to_mp(0.0)
        Ri[Fmo == 1j] = flamp.to_mp(1.0)
        return Rr, Ri
    else:
        Fmo = -_expm1(-1j*x) # 1 - F.conj()
        Fmo[Fmo == 0.0] = 1j
        R = Fmo/abs(Fmo)
        return R.real, R.imag
def _expm1(z):
    if z.dtype=='O':
        return flamp.exp(z)-1
    else:
        return np.expm1(z) # 1 - F.conj()
def _eigvals(A):
    if A.dtype=='O':
        return flamp.eig(A, left=False, right=False)
    else:
        return scipy.linalg.eigvals(A)

        
def _getpoles_geneig(y,wj):
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
def _getpoles_geneig_sym(y,b):
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
    
def _getpoles_stdeig(y,wj):
    """
    compute poles using the approach of Kno08, with standard eigenvalue problem
    """
    ak = wj / (wj.sum())
    M = np.diag(y) - np.outer(ak, y)
    lam = _eigvals(M)
    lam = np.delete(lam, np.argmin(abs(lam)))
    return lam
def _getpoles_stdeig_sym(y,wj):
    """
    symmetrized version of _getpoles_stdeig
    can be used with (flamp datatypes)
    """
    _real = np.vectorize(np.real)
    _imag = np.vectorize(np.imag)
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

def interpolate_std(allnodes, omega):
    """
    allnodes are 2n+1 nodes on the real axis
    interpolates exp(1j*omega*x) for x <- allnodes
    """
    y = allnodes[::2]
    xs = allnodes[1::2]
    F = np.exp(1j*omega*xs)
    f = np.exp(1j*omega*y)
    C = 1./(xs[:,None]-y)
    L = F[:,None]*C-C*f[None,:]
    wjs = _nullspace(L)
    n=len(y)-1
    if n%2==1: wjs*=1j
    alpha = f*wjs
    return barycentricratfct(1j*y,wjs,alpha=alpha)

def interpolate_unitary(allnodes, omega):
    """
    allnodes are 2n+1 nodes on the real axis
    interpolates exp(1j*omega*x) for x <- allnodes
    """
    y=allnodes[::2]
    xs=allnodes[1::2]
    n = len(xs)
    # nodes_pos are all strictly positive nodes
    n = len(y)-1 # total number of nodes is 2n+1 <- always odd!!

    one = xs[0]*0+1
    C = one/(xs[:,None]-y[None,:])

    Rr, Ri = _rotvecsLoewner(omega*xs)
    Kr, Ki = _rotvecsLoewner(omega*y)
    
    A = Ri[:,None]*C*Kr[None,:] - Rr[:,None]*C*Ki[None,:]
    v0 = _nullspace(A)
    b = (-Ki+1j*Kr)*v0
    if n%2==1: b*=1j
    return barycentricratfct(1j*y,b)

def interpolate_unitarysym(nodes_pos, omega):
    """
    nodes_pos are all strictly positive and real nodes, total number of nodes is 2n+1 <- always odd!!
    thus, zero is always in the set of interpolation nodes
    """
    n = len(nodes_pos)
    m = n+1

    # of 2n+1 nodes, n+1 are support nodes
    ys_pos = nodes_pos[(n+1)%2::2]
    # of 2n+1 nodes, n are test nodes
    xs_pos = nodes_pos[n%2::2]
        
    Fmo = -_expm1(-1j*omega*xs_pos)
    Fmo[Fmo == 0.0] = 1j
    Rz = Fmo/abs(Fmo)
    (Rr, Ri) = (Rz.real, Rz.imag)
    
    Fmo = -_expm1(-1j*omega*ys_pos)
    Fmo[Fmo == 0.0] = 1j
    Kz = Fmo/abs(Fmo)
    (Kr, Ki) = (Kz.real, Kz.imag)
    
    C2 = 1./(xs_pos[:,None]-ys_pos[None,:])
    C2m = 1./(xs_pos[:,None]+ys_pos[None,:])
    
    B1 = Ri[:,None]*C2*Kr[None,:] - Rr[:,None]*C2*Ki[None,:]
    B2 = Ri[:,None]*C2m*Kr[None,:] + Rr[:,None]*C2m*Ki[None,:]
    if (m%2 == 1):
        b0 = (Rr/xs_pos)[:,None] # no minus sign needed here since we also remove minus below
        B = np.concatenate((b0, B1 - B2), axis=1)
    else:
        B = B1 + B2
    v0 = _nullspace(B) # nullspace
    if (m%2 == 1):
        # n%2==0
        # Bm has k x k+1 dimensional and always has non-trivial nullspace, no need to look at Bp !!
        b2sub = 1j*Kz*v0[1:]
        b = np.concatenate(([v0[0]], b2sub, b2sub.conj()))
        y = np.concatenate(([0], ys_pos, -ys_pos))
    else:
        # Bp has k x k+1 dimensional and always has non-trivial nullspace, no need to look at Bm !!
        # b2sub = 1j*Kz*v0         #b = 1j*np.concatenate((b2sub, -b2sub.conj()))
        b2sub = -Kz*v0
        b = np.concatenate((b2sub, b2sub.conj()))
        y = np.concatenate((ys_pos, -ys_pos))
    return barycentricratfct(1j*y,b)

def minlin_nonint_std(x, y, w, C=None, B=None, wt=None):
    N = len(x)
    xs = np.concatenate((x,y))
    n, m = len(xs), len(y)
    F = np.exp(1j*w*xs)
    if C is None:
        # Cauchy matrix
        C = np.zeros([n,m])
        C[:N,:] = 1./(x[:,None]-y)
        C[N:,:] = np.eye(m)
    if B is None:
        # extended Loewner matrix
        B = np.concatenate((C, -F[:,None]*C), axis=1)
    if wt is not None:
        A=wt[:,None]*B
    else:
        A=B
    [U,S,V]=np.linalg.svd(A, full_matrices=False)
    wa = V[-1,:m].conj()
    wb = V[-1,m:].conj()
    N = np.dot(C,wa)
    D = np.dot(C,wb)
    return [N/D-F, C, B, barycentricratfct(1j*y, 1j*wb, alpha = 1j*wa)]
    
def minlin_nonint_unitary(x, y, w, C=None, B=None, wt=None):
    N = len(x)
    xs = np.concatenate((x,y))
    n, m = len(xs), len(y)
    F = np.exp(1j*w*xs)
    if C is None:
        # Cauchy matrix
        C = np.zeros([n,m])
        C[:N,:] = 1./(x[:,None]-y)
        C[N:,:] = np.eye(m)
    if B is None:
        Fmo = 1.0 - np.exp(-1j*w*xs[:,None]) # 1 - F.conj()
        Fmo[Fmo == 0.0] = 1j
        Rz = Fmo/abs(Fmo)
        (Rr, Ri) = (Rz.real, Rz.imag)
        B = np.concatenate((Rr*C, -Ri*C), axis=1)
    if wt is not None:
        A=wt[:,None]*B
    else:
        A=B
    [U,S,V]=np.linalg.svd(A, full_matrices=False)
    gam = V[-1,:]
    b = (gam[0:m]-1j*gam[m:])/np.sqrt(2.0)
    D = np.dot(C,b)
    N = D.conj()
    wjs = b
    return [N/D-F, C, B, barycentricratfct(1j*y, 1j*b)]
    

def evalr_std(x, yj, alpha, beta):
    xv = np.asanyarray(x).ravel()
    D = xv[:,None] - yj[None,:]
    # find indices where x is exactly on a node
    (node_xi, node_zi) = np.nonzero(D == 0)
    one = 1.0
    if len(node_xi) == 0:       # no zero divisors
        C = np.divide(one, D)
        numx = C.dot(alpha)
        denomx = C.dot(beta)
        r = numx / denomx
    else:
        # set divisor to 1 to avoid division by zero
        D[node_xi, node_zi] = one
        C = np.divide(one, D)
        numx = C.dot(alpha)
        denomx = C.dot(beta)
        r = numx / denomx
        r[node_xi] = alpha[node_zi]/beta[node_zi]

    if np.isscalar(x):
        return r[0]
    else:
        r.shape = np.shape(x)
        return r

def evalr_unitary(z, zj, beta):
    """
    evaluate r(z) for unitary r=d*/d with weights alpha=-beta*
    """
    m = len(zj)
    zv = np.asanyarray(z).ravel()
    D = zv[:,None] - zj[None,:]
    # find indices where x is exactly on a node
    (node_xi, node_zi) = np.nonzero(D == 0)
    one = beta[0]*0+1.0
    alpha = (-1)**(m-1)*beta.conj()
    if len(node_xi) == 0:       # no zero divisors
        C = np.divide(one, D)
        numx = C.dot(alpha)
        denomx = C.dot(beta)
        r = numx / denomx
    else:
        # set divisor to 1 to avoid division by zero
        D[node_xi, node_zi] = one
        C = np.divide(one, D)
        numx = C.dot(alpha)
        denomx = C.dot(beta)
        r = numx / denomx
        r[node_xi] = alpha[node_zi]/beta[node_zi]

    if np.isscalar(z):
        return r[0]
    else:
        r.shape = np.shape(z)
        return r
def evalr_unitary_oniR(z, zj, wj):
    """
    evaluate r(z) for unitary r=d*/d with weights alpha=-beta*
    """
    zv = np.asanyarray(z).ravel()

    D = zv[:,None] - zj[None,:]
    # find indices where x is exactly on a node
    (node_xi, node_zi) = np.nonzero(D == 0)
    
    one = 1.0    
    with np.errstate(divide='ignore', invalid='ignore'):
        if len(node_xi) == 0:       # no zero divisors
            C = np.divide(one, D)
            denomx = C.dot(wj)
            r = np.conj(denomx) / denomx
        else:
            # set divisor to 1 to avoid division by zero
            D[node_xi, node_zi] = one
            C = np.divide(one, D)
            denomx = C.dot(wj)
            r = denomx.conj() / denomx
            # fix evaluation at support nodes
            r[node_xi] = -np.conj(wj[node_zi])/wj[node_zi]

    if np.isscalar(z):
        return r[0]
    else:
        r.shape = np.shape(z)
        return r
def _evalr_unisym(zs, zj, b):
    yj = zj.imag
    zv = np.asanyarray(zs).ravel()
    m = len(yj)
    ij = np.where(yj>0)
    m2 = len(ij[0])
    b2 = b[ij]
    yjp = yj[ij]
    
    if (m%2==1):
        ijzero = np.where(yj==0)
        bzer = b[ijzero]
        if len(bzer)!=1:
            print("error. did not find zero entry for evalr_unitary_sym with odd m")
    D = zv[:,None]**2 + yjp[None,:]**2
    # find indices where x is exactly on a node
    (node_xi, node_zi) = np.nonzero(D == 0)
    if len(node_xi) > 0:       # remove zero divisors
        D[node_xi, node_zi] = 1.0

    C = np.divide(1.0, D)
    with np.errstate(divide='ignore', invalid='ignore'):
        if (m%2==0):
            denom = 2*zv*C.dot(b2.real) - 2*C.dot(yjp*b2.imag) 
        else:
            denom = 2j*zv*C.dot(b2.imag) + 2j*C.dot(yjp*b2.real) + 1j*bzer.imag/zv
        r=denom.conj()/denom
    if len(node_xi) > 0:
        node_xi_pos = node_xi[zv[node_xi]>0]
        node_zi_pos = node_zi[zv[node_xi]>0]
        node_xi_neg = node_xi[zv[node_xi]<0]
        node_zi_neg = node_zi[zv[node_xi]<0]
        if (m%2==0):
            r[node_xi_pos] = -np.conj(b2[node_zi_pos])/b2[node_zi_pos]
            r[node_xi_neg] = -b2[node_zi_neg]/np.conj(b2[node_zi_neg])
        else:
            r[node_xi_pos] = -np.conj(b2[node_zi_pos])/b2[node_zi_pos]
            r[node_xi_neg] = -b2[node_zi_neg]/np.conj(b2[node_zi_neg])
    if (m%2==1):
        ijz=np.where(zv==0)
        if len(ijz[0])>0:
            r[ijz]=-bzer.conj()/bzer
            
    if np.isscalar(zs):
        return r[0]
    else:
        r.shape = np.shape(zs)
        return r



