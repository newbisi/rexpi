import numpy as np
import scipy.linalg

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

    def getpoles(self,sym=False):
        y, wj = self.y, self.beta
        m = len(y)
        if (m<=1):
            return []
        if sym:
            return _getpoles_sym(y,wj)
        else:
            return _getpoles(y,wj)  
    def getpartialfractioncoef(self,sym=False):
        y, beta = self.y, self.beta
        if (len(y)<=1):
            return []
        sj = self.getpoles(sym)
        a0 = np.conj(sum(beta))/sum(beta)
        C_pol = 1.0 / (sj[:,None] - y[None,:])
        N_pol = C_pol.dot(np.conj(beta))
        Ddiff_pol = (-C_pol**2).dot(beta)
        aj = N_pol / Ddiff_pol
        return -a0, -aj, sj
    def coef(self):
        if self.alpha is None:
            return self.y, self.beta
        else:
            return self.y, self.beta, self.alpha
        
def _getpoles(y,wj):
    m = len(y)
    B = np.eye(m+1)
    B[0,0] = 0
    E = np.block([[0, wj],
                [np.ones([m,1]), np.diag(y)]])
    lam = scipy.linalg.eigvals(E, B)
    lam = lam[np.isfinite(lam)]
    return lam
def _getpoles2(y,wj):
    ak = wj / (wj.sum()).real
    M = np.diag(y) - np.outer(ak, y)
    lam = scipy.linalg.eigvals(M)
    lam = np.delete(lam, np.argmin(abs(lam)))
    return lam
def _getpoles_sym(y,wj):        
    m, n = len(y), len(y)-1
    ij=np.argsort(np.abs(y.imag))
    wj=wj[ij]
    y=y[ij]
    if n%2==0:
        ak = wj / (1j*(wj.sum()).imag)
    else:
        ak = wj / (wj.sum()).real
    M = np.diag(y) - np.outer(ak, y)
    if n%2==0:
        Mx=M[1:,1:]
        n2=n
    else:
        Mx=M
        n2=n+1
    ij=np.zeros(n2,dtype=np.int32)
    ij[::2]=np.arange(1,n2,2)
    ij[1::2]=np.arange(0,n2,2)
    P1=np.eye(n2)+0j
    P2=P1[ij,:]
    P=1j*P1-P2
    #Pinv=np.linalg.inv(P)
    Pinv=-0.5*(1j*P1+P2)
    M2=(Pinv.dot(Mx.dot(P))).real
    lam = scipy.linalg.eigvals(M2)
    if n%2==1:
        lam = np.delete(lam, np.argmin(abs(lam)))
    return lam

def interpolate_std(allnodes, omega):
    # nodes_pos are all strictly positive nodes, total number of nodes is 2n+1 <- always odd!!
    y = allnodes[::2]
    xs = allnodes[1::2]

    C = 1./(xs[:,None]-y)
    F = np.exp(1j*omega*xs[:,None])
    f = np.exp(1j*omega*y[None,:])
    L = F*C-C*f

    [U,S,V]=np.linalg.svd(L,full_matrices=True)
    wjs = (V[-1,:]).conj()
    alpha = f[0]*wjs
    
    r = barycentricratfct(1j*y,1j*wjs,alpha=1j*alpha)
    return r
    
def interpolate_unitary(allnodes, omega):
    y=allnodes[::2]
    xs=allnodes[1::2]
    # nodes_pos are all strictly positive nodes
    n = len(y)-1 # total number of nodes is 2n+1 <- always odd!!
    
    C = 1./(xs[:,None]-y[None,:])
    
    Fmo = -np.expm1(-1j*omega*xs[:,None]) # 1 - F.conj()
    Fmo[Fmo == 0.0] = 1j
    Rz = Fmo/np.abs(Fmo)
    (Rr, Ri) = (Rz.real, Rz.imag)
    
    Fmo = -np.expm1(-1j*omega*y[None,:]) # 1 - f.conj()
    Fmo[Fmo == 0.0] = 1j
    Kz = Fmo/np.abs(Fmo)
    (Kr, Ki) = (Kz.real, Kz.imag)
    
    A = (Ri*C*Kr-Rr*C*Ki)
    [U,S,V]=np.linalg.svd(A,full_matrices=True)
    b = 1j*Kz[0,:]*V[-1,:]   
    
    r = barycentricratfct(1j*y,1j*b)
    return r

def interpolate_unitarysym(nodes_pos, omega):
    # nodes_pos are all strictly positive nodes, total number of nodes is 2n+1 <- always odd!!
    # thus, zero is always in the set of interpolation nodes
    n = len(nodes_pos)
    m = n+1

    # of 2n+1 nodes, n+1 are support nodes
    ys_pos = nodes_pos[(n+1)%2::2]
    # of 2n+1 nodes, n are test nodes
    xs_pos = nodes_pos[n%2::2]

    Fmo = -np.expm1(-1j*omega*xs_pos[:,None])
    Fmo[Fmo == 0.0] = 1j
    Rz = Fmo/np.abs(Fmo)
    (Rr, Ri) = (Rz.real, Rz.imag)
    
    Fmo = -np.expm1(-1j*omega*ys_pos[None,:])
    Fmo[Fmo == 0.0] = 1j
    Kz = Fmo/np.abs(Fmo)
    (Kr, Ki) = (Kz.real, Kz.imag)
    
    C2 = 1./(xs_pos[:,None]-ys_pos[None,:])
    C2m = 1./(xs_pos[:,None]+ys_pos[None,:])
    
    B1 = Ri*C2*Kr - Rr*C2*Ki
    B2 = Ri*C2m*Kr + Rr*C2m*Ki
    if (m%2 == 1):
        b0 = -(2**0.5)*Rr/xs_pos[:,None]
        B = np.concatenate((b0, B1 - B2), axis=1)
    else:
        B = B1 + B2
    [_,SB,V]=np.linalg.svd(B,full_matrices=True)
    if (m%2 == 1):
        # Bm has k x k+1 dimensional and always has non-trivial nullspace, no need to look at Bp !!
        b2sub0 = -(2**0.5)*V[-1,0]
        b2sub = 1j*Kz[0,:]*V[-1,1:]
        b = np.concatenate(([b2sub0], b2sub, b2sub.conj()))
        y = np.concatenate(([0], ys_pos, -ys_pos))
    else:
        # Bp has k x k+1 dimensional and always has non-trivial nullspace, no need to look at Bm !!
        b2sub = 1j*Kz[0,:]*V[-1,:]
        b = np.concatenate((b2sub, -b2sub.conj()))
        y = np.concatenate((ys_pos, -ys_pos))
        
    r = barycentricratfct(1j*y,1j*b)
    return r

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
        Rz = Fmo/np.abs(Fmo)
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

def evalr_unitary(z, zj, wj):
    # evaluate r(z) for unitary r=d*/d with weights alpha=-beta*
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



