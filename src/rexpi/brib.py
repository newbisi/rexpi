import numpy as np
import scipy.linalg
from .barycentricfcts import *
from .cheb import PositiveChebyshevNodes

import gmpy2
import flamp


def _exp(z):
    if isinstance(z, complex):
        return np.exp(z)
    elif isinstance(z, gmpy2.mpc):
        return flamp.exp(z)
    elif z.dtype=='O':
        return flamp.exp(z)
    else:
        return np.exp(z)

def brib(w = 10.0, n=6, nodes_pos = None, syminterp = False,
         maxiter=100, tolequi=1e-3, npi = -30, step_factor = 0.1):
    """
    best rational interpolation based (brib) approximation, r(ix) \approx exp(iwx)
    parts of this algorithm are taken from baryrat https://github.com/c-f-h/baryrat
    C. Hofreither. An algorithm for best rational approximation based on barycentric rational interpolation.
    Numerical Algorithms, 88(1):365--388, 2021. doi 10.1007/s11075-020-01042-0
    
    n .. compute (n,n) unitary rational best approximation
    tol .. stop the iteration 
    y or n, mirrored nodes, flat, sorted
    w or tol, tol only used to specify w if needed, no stopping criteria
    """
    a = -1.0
    a0 = 0.0
    b = 1.0

    ###### set parameters
    if nodes_pos is None:
        nodes_pos = PositiveChebyshevNodes(n)
    else:
        nodes_pos = np.sort(nodes_pos[nodes_pos>0].flatten())
        n = len(nodes_pos)
    
    if (w >= np.pi*(n+1)):
        # return constant one function
        print("for w>=(n+1)pi = %f the best approximartion is trivial" % (n+1)*np.pi)
        allpoints=[[0.0],[0.0]]
        allerr = [2.0]
        return barycentricratfct([0.0],[1.0]), allpoints ,allerr
    ######
    
    f = lambda x : _exp(1j*w*x)
    errors = []
    approxerrors = []
    stepsize = np.nan
    #ni = 1
    max_step_size=0.1
    
    for ni in range(maxiter):
        # compute the new interpolant r
        if syminterp:
            # ypos, xpos, izero
            r = interpolate_unitarysym(nodes_pos, w)
        else:
            allnodes = np.concatenate((-nodes_pos[::-1], [0], nodes_pos))
            r = interpolate_unitary(allnodes, w)
    
        # find nodes with largest error
        all_nodes_pos = np.concatenate(([0.0], nodes_pos, [1.0]))
        errfun = lambda x: abs(f(x) - r(1j*x))

        #print(errfun(nodes_pos))
        if npi > 0:
            local_max_x, local_max = local_maxima_sample(errfun, all_nodes_pos, npi)
        else:
            local_max_x, local_max = local_maxima_golden(errfun, all_nodes_pos, num_iter=-npi)
    
        max_err = local_max.max()
        #deviation_old = max_err / local_max.min() - 1
        deviation = 1 - local_max.min() / max_err


        errors.append((max_err, deviation, stepsize))
        approxerrors.append(max_err)
    
        converged = (deviation <= tolequi)
        if converged:
            # only if converged
            #signed_errors = np.angle(r(1j*local_max_x)/f(local_max_x))
            #signed_errors /= (-1)**np.arange(len(signed_errors)) * np.sign(signed_errors[0]) * max_err
            #equi_err = abs(1.0 - signed_errors).max()
            break
        
        # global interval size adjustment
        intv_lengths = np.diff(all_nodes_pos)

        mean_err = np.mean(local_max)
        max_dev = abs(local_max - mean_err).max()
        normalized_dev = (local_max - mean_err) / max_dev
        stepsize = min(max_step_size, step_factor * max_dev / mean_err)
        scaling = (1.0 - stepsize)**normalized_dev

        intv_lengths *= scaling
        # rescale so that they add up to b-a again
        intv_lengths *= 1 / intv_lengths.sum()
        nodes_pos = np.cumsum(intv_lengths)[:-1] + a0

    # also return interpolation nodes and equioscillation points 
    allpoints = [np.concatenate((-local_max_x[::-1], local_max_x)), np.concatenate((-nodes_pos[::-1], [0], nodes_pos))]
    return r, allpoints, errors

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

def local_maxima_bisect(g, nodes, num_iter=10):
    # subroutine from https://github.com/c-f-h/baryrat
    # C. Hofreither. An algorithm for best rational approximation based on barycentric rational interpolation.
    # Numerical Algorithms, 88(1):365--388, 2021. doi 10.1007/s11075-020-01042-0
    L, R = nodes[1:-2], nodes[2:-1]
    # compute 3 x m array of endpoints and midpoints
    z = np.vstack((L, (L + R) / 2, R))
    values = g(z[1])
    m = z.shape[1]

    for k in range(num_iter):
        # compute quarter points
        q = np.vstack(((z[0] + z[1]) / 2, (z[1] + z[2])/ 2))
        qval = g(q)

        # move triple of points to be centered on the maximum
        for j in range(m):
            maxk = np.argmax([qval[0,j], values[j], qval[1,j]])
            if maxk == 0:
                z[1,j], z[2,j] = q[0,j], z[1,j]
                values[j] = qval[0,j]
            elif maxk == 1:
                z[0,j], z[2,j] = q[0,j], q[1,j]
            else:
                z[0,j], z[1,j] = z[1,j], q[1,j]
                values[j] = qval[1,j]

    # find maximum per column (usually the midpoint)
    #maxidx = values.argmax(axis=0)
    # select abscissae and values at maxima
    #Z, gZ = z[maxidx, np.arange(m)], values[np.arange(m)]
    Z, gZ = np.empty(m+2), np.empty(m+2)
    Z[1:-1] = z[1, :]
    gZ[1:-1] = values
    # treat the boundary intervals specially since usually the maximum is at the boundary
    Z[0], gZ[0] = _boundary_search(g, nodes[0], nodes[1], num_iter=3)
    Z[-1], gZ[-1] = _boundary_search(g, nodes[-2], nodes[-1], num_iter=3)
    return Z, gZ

def local_maxima_golden(g, nodes, num_iter):
    # subroutine from https://github.com/c-f-h/baryrat
    # C. Hofreither. An algorithm for best rational approximation based on barycentric rational interpolation.
    # Numerical Algorithms, 88(1):365--388, 2021. doi 10.1007/s11075-020-01042-0
    # vectorized version of golden section search
    golden_mean = (3.0 - np.sqrt(5.0)) / 2   # 0.381966...
    L, R = nodes[1:-2], nodes[2:-1]     # skip boundary intervals (treated below)
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
    Z, gZ = np.empty(m+2, dtype=z.dtype), np.empty(m+2, dtype=gB.dtype)
    Z[1:-1] = z[1, :]
    gZ[1:-1] = gB
    # treat the boundary intervals specially since usually the maximum is at the boundary
    # (no bracket available!)
    Z[0], gZ[0] = _boundary_search(g, nodes[0], nodes[1], num_iter=3)
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
