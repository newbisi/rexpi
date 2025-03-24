import numpy as np
from .barycentricfcts import *
from .cheb import PositiveChebyshevNodes

import time
import logging

def brib_err(n,eo):
    w = west(n,eo)
    r, info = brib(w,n)
    return r, info

def brib(w=10.0, n=6, nodes_pos=None, syminterp=True,
         maxiter=None, tolequi=1e-3,
         tolstagnation=0, # if devation is smaller than this tol and stagnates, break iteration
         kstag=10,
         tolerr=0, #  if error is smaller tolerr and phis are alternating
         npi=None, # find max
         optnodesadapt=None, # adapt nodes
         npigolden = -20, npisample=9,
         step_factor=2.2, max_step_size=0.1, hstep=0.1):
    """
    best rational interpolation based (brib) approximation, r(ix) \approx exp(iwx)
    parts of this algorithm are taken from baryrat https://github.com/c-f-h/baryrat
    C. Hofreither. An algorithm for best rational approximation based on barycentric rational interpolation.
    Numerical Algorithms, 88(1):365--388, 2021. doi 10.1007/s11075-020-01042-0

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
        return barycentricratfct([0.0],[1.0]), [success, allpoints ,allerr]
        
    if nodes_pos is None:
        xi = w/(n+1)/np.pi
        nodes_pos = (1-xi)*PositiveChebyshevNodes(n) + xi*np.arange(1,n+1)/(n+1)
        # also test the following set of nodes
        #linnodes=(np.arange(n)+1)*np.pi
        #nodes_pos = np.min((linnodes,PositiveChebyshevNodes(n)),0)
    else:
        nodes_pos = np.sort(nodes_pos[nodes_pos>0].flatten())
        n = len(nodes_pos)
    if maxiter is None: maxiter = 4*n
    if npi is not None: npiuse = npi
    ######
    logging.info("start brib: n=%d, w=%f, xi=%.2e, maxiter=%d, tolequi=%.2e", n, w, w/((n+1)*np.pi), maxiter, tolequi)
    errors = []
    devs = []
    stepsize = np.nan
    #ni = 1
    nodes_history = []
    timeinterpr = 0
    timefindmax = 0
    timecorrection = 0
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
        # compute the new interpolant r
        nodes_history.append(nodes_pos)

        t1 = time.time()
        ### rational interpolation
        if syminterp:
            # ypos, xpos, izero
            r = interpolate_unitarysym(nodes_pos, omega=w)
        else:
            nodes_all = np.concatenate((-nodes_pos[::-1], [zero], nodes_pos))
            r = interpolate_unitary(nodes_all, omega=w)
        timeinterpr += time.time()-t1

        nodes_sub = np.concatenate(([zero], nodes_pos, [one]))
        errfun = lambda x: abs( r(1j*x) - exp_mp(1j*w*x) )

        t1 = time.time()
        ### find nodes with largest error
        if npi is None:
            npiuse = npigolden if ((max_err<2)&(phisignsum==0)) else npisample
        #logging.info("%d"%npiuse)
        if npiuse > 0:
            local_max_x, local_max = local_maxima_sample(errfun, nodes_sub, npiuse)
        else:
            local_max_x, local_max = local_maxima_golden(errfun, nodes_sub, num_iter=-npiuse)
        timefindmax += time.time()-t1
        
        max_err = local_max.max()
        deviation = 1 - local_max.min() / max_err
        local_max_imag = imag_mp( r(1j*local_max_x)/exp_mp(1j*w*local_max_x) )
        phisignsum = np.sum(local_max_imag*(-1)**(n+1-np.arange(n+1)) < 0)

        if ((tolstagnation>0)&(max_err<besterr)):
            besterr = max_err
            besterrdev = deviation
            besterrnodes = nodes_pos
        
        errors.append((max_err, deviation, lastusedcorrection, phisignsum))
        devs.append(deviation)

        ### test convergence
        converged = (((deviation <= tolequi)&(phisignsum==0))&(max_err<2))
        if True:
            #ni%10==0:
            logging.info("step %5d, error %.2e, deviation %.2e, alternating %3d, npiused %+3d, opt %d",ni, max_err, deviation, int(phisignsum), npiuse, lastusedcorrection)
        if converged:
            success = 0
            break
        if ni == maxiter-1:
            break

        errsmall = ((phisignsum==0)&(max_err<tolerr))
        if errsmall:
            success = 0
            break
            
        ### if not at last iteratipon, adapt interpolation nodes
        t1 = time.time()
        if optnodesadapt is None:
            if ((max_err==2)|(phisignsum!=0)):
                nodes_pos, stepsize = _adaptinodes_BRASIL(nodes_sub, local_max, max_step_size, step_factor)
                lastusedcorrection=1
            else:
                optc = 1 if (deviation<0.05) else 0
                if (hstep!=0.1):
                    nodes_pos = _adaptinodes_Maehly_sym(nodes_pos, local_max_x, local_max, optc)
                    lastusedcorrection = 3.01 + optc/2
                else:
                    nodes_pos = _adaptinodes_MaehlyDunham_sym(nodes_pos, local_max_x, local_max, optc)
                    lastusedcorrection = 3 + optc/2
        else:
            lastusedcorrection = optnodesadapt
            if optnodesadapt==1:
                nodes_pos, stepsize = _adaptinodes_BRASIL(nodes_sub, local_max, max_step_size, step_factor)
            elif optnodesadapt==2:
                nodes_pos = _adaptinodes_Franke(nodes_pos, local_max_x, local_max, hstep)
            else:
                nodes_pos = _adaptinodes_Maehly_sym(nodes_pos, local_max_x, local_max, hstep)
        timecorrection += time.time()-t1
        
        if max(nodes_pos)>1: logging.warning("step %d, opt %f. max interpolation node > 1, %f" % (ni,lastusedcorrection,max(nodes_pos)))
        # test for stagnation and stop loop if stagnation was detected in the previous iteration
        if devstagnation:
            logging.warning("iter %d. deviation stagnated at delta = %.2e before reaching tol" % (ni,deviation))
            break
        if ((ni>kstag)&(tolstagnation>0)): devstagnation = (((sum(devs[-kstag:])/kstag)<deviation)&(besterrdev<tolstagnation))
        if devstagnation: nodes_pos = besterrnodes
    
    accuracy = [n, w, max_err] # degree n and accurate on [-w,w] with max_err            
    nodes_last = [nodes_pos, local_max_x] # also return interpolation nodes and equioscillation points 
    timings = [timeinterpr, timefindmax, timecorrection]
    logging.info("done: step %d, error %.2e, deviation %.2e, alternating %d",ni, max_err, deviation, int(phisignsum))

    info = [success, accuracy, nodes_last , errors, nodes_history, timings]
    return r, info

def _adaptinodes_BRASIL(nodes_sub, local_max, max_step_size, step_factor):
    # global interval size adjustment
    n = len(nodes_sub)
    intv_lengths = np.diff(nodes_sub)

    mean_err = np.mean(local_max)
    #sum_err = len(local_max)*mean_err
    max_dev = abs(local_max - mean_err).max()
    normalized_dev = (local_max - mean_err) / max_dev
    stepsize = min(max_step_size, step_factor * max_dev / (n * mean_err))
    scaling = (1.0 - stepsize)**normalized_dev

    intv_lengths *= scaling
    # rescale so that they add up to b-a again
    intv_lengths /= intv_lengths.sum()
    nodes_pos = np.cumsum(intv_lengths)[:-1]
    return nodes_pos, stepsize
def _adaptinodes_Franke(nodes_pos, local_max_x, local_max, hstep):
    dz = np.diff(local_max)*np.diff(local_max_x)/(np.sum(local_max)/len(local_max))
    nodes_pos += hstep*dz
    return nodes_pos
def _adaptinodes_Maehly(nodes_pos, local_max_x, local_max, useln=1):
    n=len(nodes_pos)
    zero = 0*nodes_pos[0]
    alleps = np.concatenate((local_max[::-1],local_max))
    allx = np.concatenate((-local_max_x[::-1], local_max_x))
    ally = np.concatenate((-nodes_pos[::-1], [zero], nodes_pos))
    if useln==1:
        b = log_mp(alleps[1:]/alleps[0])
    else:
        b = 2*(alleps[1:]-alleps[0])/(alleps[1:]+alleps[0])
    M = (allx[0]-allx[1:,None])/((allx[1:,None]-ally)*(allx[0]-ally))
    dz = solve_mp(M, b)
    return nodes_pos+dz[n+1:]
def _adaptinodes_Maehly_sym(xs, local_max_x, local_max, useln=1):
    n=len(xs)
    etas, eta0 = local_max_x[:-1,None], local_max_x[-1]
    ers, er0 = local_max[:-1], local_max[-1]
    if useln==1:
        b = log_mp(ers/er0)
    else:
        b = 2*(ers-er0)/(ers+er0)
    #M = 1/(etas-xs) - 1/(etas**2-xs**2) + 2*xs/(xs**2-eta0**2)
    #M = 2*xs/(etas**2-xs**2) + 2*xs/(xs**2-eta0**2)
    M = 2*xs*(etas**2-eta0**2)/((xs**2-eta0**2)*(etas**2-xs**2))
    dxs = solve_mp(M, b)
    return xs + dxs
def _adaptinodes_MaehlyDunham(nodes_pos, local_max_x, local_max, useln=1):
    n=len(nodes_pos)
    zero = 0*nodes_pos[0]
    alleps = np.concatenate((local_max[::-1],local_max))
    etas = np.concatenate((-local_max_x[::-1], local_max_x))
    xs = np.concatenate((-nodes_pos[::-1], [zero], nodes_pos))

    epsgeom = exp_mp(np.mean(log_mp(alleps)))
    if useln==1:
        b = log_mp(alleps/epsgeom)
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
def _adaptinodes_MaehlyDunham_sym(nodes_pos, local_max_x, local_max, useln=1):
    n=len(nodes_pos)
    epsgeom = exp_mp(np.mean(log_mp(local_max)))
    if useln==1:
        b = log_mp(local_max/epsgeom)
    else:
        b = 2*(local_max - epsgeom)/(local_max + epsgeom)
        
    PA = nodes_pos[:,None]**2 - local_max_x[:-1]**2
    PA /= nodes_pos[:,None]**2 - nodes_pos**2 + 2*np.diag(nodes_pos**2)
    dz = np.prod(PA, 1)*(nodes_pos**2 - local_max_x[-1]**2)
    
    PA = 1/(local_max_x[:,None]**2 - local_max_x**2 + 2*np.eye(n+1))
    PA[:,:-1] *= local_max_x[:,None]**2 - nodes_pos**2
    prodv = np.prod(PA,1)
    Ms = (b*prodv) / (nodes_pos[:,None]**2-local_max_x**2)
    #dzp2 = np.sum(Ms,1)
    dz *= 2*nodes_pos*np.sum(Ms,1)
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
    # removed boundary search at left side due to symmetry properties
    golden_mean = (3.0 - np.sqrt(5.0)) / 2   # 0.381966...
    L, R = nodes[0:-2], nodes[1:-1]     # skip _right_ boundary interval (treated below)
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
    
def local_maxima_sample2(g, nodes, N):
    Z = _piecewise_mesh(nodes, N).reshape((-1, N))
    valsc = g(Z)
    vals = abs(valsc)
    maxk = vals.argmax(axis=1)
    nn = np.arange(Z.shape[0])

    sigp = np.sign(angle_mp(valsc))
    phaseerrs=np.sum(abs((sigp[:,1:]-sigp[:,:-1])/2), axis=1, dtype=np.int32)
    return Z[nn, maxk], vals[nn, maxk], phaseerrs

