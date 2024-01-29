import numpy as np
import scipy.linalg

class pade():
    # the (k,k) Pade approximation to exp(z)
    def __init__(self, k):
        a = np.ones(k+1)
        for j in range(k):
            fact = (k-j)/(2*k-j)/(j+1)
            a[k-j-1] = -fact * a[k-j] # sum of log
        a = a/a[k]
        self.coef = a
        self.k = k

    def __call__(self, x):
        a, k = self.coef, self.k
        xv = np.asanyarray(x).ravel()
        xp = xv**0
        ys = a[k]*xp
        for j in range(k):
            xp = xp*xv
            ys = ys + a[k-j-1]*xp # exp
        denom = ys
        r = np.conj(denom)/denom
        if np.isscalar(x):
            return r[0]
        else:
            r.shape = np.shape(x)
            return r

    def getpoles(self):
        a = self.coef
        poles = np.roots(a)
        return poles
