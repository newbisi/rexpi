import numpy as np
import rexpi

def test_brib():
    ns = np.array([3,5,10,11,20,30,40,50,81])
    tol=1e-6
    rdevtol = 1e-3
    xtest = np.linspace(-1,1,5000)
    for n in ns:
        w = rexpi.west(n, tol)
        r, info = rexpi.brib(w, n, tolequi = rdevtol, info=1)
        errlast = info['err']
        rdev = info['dev']
        errmax = np.max(np.abs(r(1j*xtest)-np.exp(1j*w*xtest)))
        assert (rdev<rdevtol)
        assert (abs(errlast-tol)<0.1)
        assert (abs(errmax-errlast)<0.1)

def test_interpolation_std():
	for m in np.arange(3,50):
		w=2*m
		allnodes = np.linspace(-1, 1, 2*m-1)
		r = rexpi.interpolate_std(allnodes, w)
		for k in range(len(allnodes)):
			xnow = allnodes[k]
			assert (abs(r(1j*xnow)-np.exp(1j*w*xnow))<1e-6)
			assert (abs(abs(r(1j*xnow))-1)<1e-8)
			
def test_interpolation_unitary():
	for m in np.arange(3,50):
		w=2*m
		allnodes = np.linspace(-1, 1, 2*m-1)
		r = rexpi.interpolate_unitary(allnodes, w)
		for k in range(len(allnodes)):
			xnow = allnodes[k]
			assert (abs(r(1j*xnow)-np.exp(1j*w*xnow))<1e-6)
			assert (abs(abs(r(1j*xnow))-1)<1e-14)
			
def test_interpolation_sym():
	for m in np.arange(3,50):
		w=2*m
		allnodes = np.linspace(-1, 1, 2*m-1)
		nodes_pos = allnodes[allnodes>0]
		r = rexpi.interpolate_unitarysym(nodes_pos, w)
		for k in range(len(allnodes)):
			xnow = allnodes[k]
			assert (abs(r(1j*xnow)-np.exp(1j*w*xnow))<1e-6)
			assert (abs(abs(r(1j*xnow))-1)<1e-14)
