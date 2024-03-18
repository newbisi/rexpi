import numpy as np
import rexpi

def test_brib():
    n=10
    tol=1e-6
    rdevtol = 1e-3
    w = rexpi.buerrest_getw(n, tol)
    r, _, allerr = rexpi.brib(w, n, tolequi = rdevtol)
    errlast = allerr[-1][0]
    rdev = allerr[-1][1]
    xs = np.linspace(-1,1,5000)
    err = r(1j*xs)-np.exp(1j*w*xs)
    assert (rdev<rdevtol)
    assert (errlast<tol)
    assert (np.max(np.abs(err))<tol)

def test_interpolation_std():
	for m in np.arange(3,50):
		w=2*m
		allnodes = np.linspace(-1, 1, 2*m-1)
		r = rexpi.interpolate_std(allnodes, w)
		for k in range(len(allnodes)):
			xnow = allnodes[k]
			assert (abs(r(1j*xnow)-np.exp(1j*w*xnow))<1e-8)
			assert (abs(abs(r(1j*xnow))-1)<1e-8)
			
def test_interpolation_unitary():
	for m in np.arange(3,50):
		w=2*m
		allnodes = np.linspace(-1, 1, 2*m-1)
		r = rexpi.interpolate_unitary(allnodes, w)
		for k in range(len(allnodes)):
			xnow = allnodes[k]
			assert (abs(r(1j*xnow)-np.exp(1j*w*xnow))<1e-8)
			assert (abs(abs(r(1j*xnow))-1)<1e-14)
			
def test_interpolation_sym():
	for m in np.arange(3,50):
		w=2*m
		allnodes = np.linspace(-1, 1, 2*m-1)
		nodes_pos = allnodes[allnodes>0]
		r = rexpi.interpolate_unitarysym(nodes_pos, w)
		for k in range(len(allnodes)):
			xnow = allnodes[k]
			assert (abs(r(1j*xnow)-np.exp(1j*w*xnow))<1e-8)
			assert (abs(abs(r(1j*xnow))-1)<1e-14)
