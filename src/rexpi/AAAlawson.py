import numpy as np
import scipy.linalg
from .barycentricfcts import *
from .cheb import PositiveChebyshevNodes


def linearizedLawson(w = np.inf, y = None, n = 8, tol = 1e-8,
                              x = None, nx = 2000, nlawson = 20, idl = 1):
    # y or n, mirrored nodes, flat, sorted
    # w or tol, tol only used to specify w if needed, no stopping criteria
    # x or nx, mirrored nodes, flat, sorted, distinct to y
    # idl = 0 classical Lawson
    # idl = 1 unitary Lawson
    # idl = 2 unitary+symmetric Lawson
    # can n be nmax????
    if y is None:
        Cheb_pos = PositiveChebyshevNodes(n)
        Cheb = np.concatenate((-Cheb_pos[::-1],[0],Cheb_pos))
        y = Cheb[::2]
    else:
        y = np.sort(y.flatten())
        if sum(y+y[::-1])>0:
            print("Warning: Lawson support nodes not mirrored around zero")
        n = len(y)-1

    if (w >= np.pi*(n+1)):
        # return constant one function
        print("for w>=(n+1)pi = {} the best approximartion is trivial")
        return (barycentricratfct([0.0],[1.0]) ,[2.0])
        
    if x is None:
        x = np.linspace(-1,1,nx)
    else:
        x = np.sort(x.flatten())
        if sum(x+x[::-1])>0:
            print("Warning: Lawson test nodes not mirrored around zero")
    if len(np.argwhere( (x[:,None]-y) == 0.0 ) > 0):
        print("Warning: some Lawson support and test nodes are identical.")

    return _linearizedLawson(x, y, w, nlawson = nlawson, idl = idl)
        

def _linearizedLawson(x, y, w, nlawson = 20, idl = 1, returnrlist=False):
    # parts of this algorithm are taken from AAA-Lawson http://www.chebfun.org/
    #N = len(x)
    #xs = np.concatenate((x,y))
    #m = len(y)
    #n = N+m
    # is m is the number of support nodes to barycentric rational approximation
    # degree of the approximation is m+1
    # todo: make sure entries of y are not in x yet

    # idl = 0 classical Lawson
    # idl = 1 unitary Lawson
    # idl = 2 unitary+symmetric Lawson

    C, B = None, None
    N, m = len(x), len(y)
    wt = np.ones([N+m])

    errvec = []
    stepno = 0
    rlist = []
    while (stepno < nlawson):
        stepno += 1
        if (idl==0):
            [err,C,B,r] = minlin_nonint_std(x, y, w, C=C, B=B, wt=np.sqrt(wt))
        else:
            [err,C,B,r] = minlin_nonint_unitary(x, y, w, C=C, B=B, wt=np.sqrt(wt))
        errv = np.abs(err)
        maxerr = np.max(errv)
        errvec.append(maxerr)
        if returnrlist:
            rlist.append(r)
        #if maxerr<tol:
        #    break

        wt = wt * errv
        wt = wt/np.max(wt)
        
        if any(wt != wt): # check for nan
            break
    if returnrlist:
        [r, errvec,rlist]
    else:
        return [r, errvec]

def _linearizedLawson_old(x, y, w, nlawson = 20, idl = 1):
    # parts of this algorithm are taken from AAA-Lawson http://www.chebfun.org/
    N = len(x)
    xs = np.concatenate((x,y))
    n = len(xs)
    m = len(y)
    # is m is the number of support nodes to barycentric rational approximation
    # degree of the approximation is m+1
    # todo: make sure entries of y are not in x yet

    # idl = 0 classical Lawson
    # idl = 1 unitary Lawson
    # idl = 2 unitary+symmetric Lawson


    # setup on first r=.. call and then re-use    
    A = None
    # wt = 
    
    # Cauchy matrix
    C = np.zeros([n,m])
    C[:N,:] = 1./(x[:,None]-y)
    C[N:,:] = np.eye(m)

    F = np.exp(1j*w*xs) 
    # Loewner matrix
    if (idl==0):
        A = np.concatenate((C, -F[:,None]*C), axis=1)
        wt = np.ones([n,1])

    elif (idl==1):
        Fmo = 1.0 - np.exp(-1j*w*xs[:,None]) # 1 - F.conj()
        Fmo[Fmo == 0.0] = 1j
        Rz = Fmo/np.abs(Fmo)
        (Rr, Ri) = (Rz.real, Rz.imag)
        A = np.concatenate((Rr*C, -Ri*C), axis=1)
        wt = np.ones([n,1])
        
    elif (idl==2):
        # only positive nodes
        (y2, m2) = (y[y>0], len(y[y>0]))
        (x2, N2) = (x[x>0], len(x[x>0]))
        (xs2, n2) = (xs[xs>0], len(xs[xs>0]))
    
        # Cauchy matrix parts
        C2 = np.zeros([n2,m2])
        C2[:N2,:] = 1./(x2[:,None]-y2)
        C2[N2:,:] = np.eye(m2)
        C2m = np.zeros([n2,m2])
        C2m[:N2,:] = 1./(-x2[:,None]-y2)
        
        Fmo = 1.0 - np.exp(-1j*w*xs2[:,None])
        Fmo[Fmo == 0.0] = 1j
        Rz = Fmo/np.abs(Fmo)
        (Rr, Ri) = (Rz.real, Rz.imag)
    
        if (m%2 == 0):
            B1 = Rr*(C2+C2m)
            B2 = -Ri*(C2-C2m)
        else:
            B1 = Rr*(C2-C2m)
            B2 = -Ri*(C2+C2m)
        A = np.concatenate((B1, B2), axis=1)
        wt = np.ones([n2,1])
    
    errvec = []
    stepno = 0
    while (stepno < nlawson):
        stepno += 1
        [U,S,V]=np.linalg.svd(np.sqrt(wt)*A, full_matrices=False)

        if (idl==0):
            wa = V[-1,:m].conj()
            wb = V[-1,m:].conj()
            N = np.dot(C,wa)
            D = np.dot(C,wb)
            wjs = [wa,wb]
        elif (idl==1):
            gam = V[-1,:]
            b = (gam[0:m]-1j*gam[m:])/np.sqrt(2.0)
            D = np.dot(C,b)
            N = D.conj()
            wjs = b
        elif (idl==2):
            if (m%2 == 0):
                g0 = V[-1,:]
                mh = int(m/2)
                b = np.concatenate((-g0[mh-1::-1] - 1j*g0[m:mh-1:-1], g0[:mh] - 1j*g0[mh:]))/2
                #b = (g0[:mh] - 1j*g0[mh:])/2
            else:
                a0 = np.sqrt(wt[:,0])*Rr[:,0]/xs2
                a0[N2:] = 0
                a1 = np.dot(U.transpose(),a0)
                a2 = np.dot(V.transpose(),a1/S)
                #axnorm = (2*np.linalg.norm(gf)**2+1)**0.5
                mh = int(m/2)
                gf = np.concatenate((a2[mh-1::-1], [-1], a2[:mh], -a2[m:mh-1:-1], [0], a2[mh:]))
                gf = gf/np.linalg.norm(gf)
                b = (gf[0:m]-1j*gf[m:])/np.sqrt(2.0)
                #gf = (1j*a2[:mh] + a2[mh:])/np.sqrt(2.0)/axnorm
                #b = np.concatenate((-b[::-1].conj())
            # evaluation of r at test nodes can be simplified for the symmetric case
            D = np.dot(C,b)
            N = D.conj()
            wjs = b
        errv = np.abs(N/D - F)
        maxerr = np.max(errv)
        errvec.append(maxerr)
        #if maxerr<tol:
        #    break

        if (idl==2):
            wt = wt * errv[xs>0,None]
        else:
            wt = wt * errv[:,None]
        wt = wt/max(wt)
        
        if any(wt != wt): # check for nan
            break
    if (idl==0):
        r = barycentricratfct(y,wb,alpha=wa)
    else:
        r = barycentricratfct(1j*y,1j*wjs)
    return [r, errvec]
