# rexpi [![PyPI version](https://badge.fury.io/py/rexpi.svg)](https://badge.fury.io/py/rexpi)
*by Tobias Jawecki*

A Python package providing algorithms to compute unitary $(n,n)$-rational best approximations $r(\mathrm{i} x) \approx \mathrm{e}^{\mathrm{i}\omega x}$, for $x\in[-1,1]$ and a frequency $\omega>0$. Unitary $(n,n)$-rational functions correspond to rational functions $r=p/q$ where $p$ and $q$ are polynomials of degree $\leq n$ with $|r(\mathrm{i} x)|=1$ for $x\in\mathbb{R}$. Our focus is best approximations in a Chebyshev sense which minimize the uniform error $\max_{x\in[-1,1]}| r(\mathrm{i} x) - \mathrm{e}^{\mathrm{i}\omega x} |$.

Besides new code, this package also contains some routines of the [baryrat](https://github.com/c-f-h/baryrat) package[^Ho20], and variants of the AAA-Lawson[^NST18][^NT20] method which is part of the [Chebfun](http://www.chebfun.org/) package[^DHT14].

**This package is still under development. Future versions might have different API. The current version is not fully documented.**

## Geting started
Install the package `python -m pip install rexpi` and run numerical examples from the `/examples` folder.

## Best approximations and equioscillating phase errors

Following a recent work[^JSxx], for $\omega\in(0,\pi(n+1))$ the unitary best approximation to $\mathrm{e}^{\mathrm{i}\omega x}$ is uniquely characterized by an equioscillating phase error. In particular, unitary rational functions are of the form $r(\mathrm{i} x) = \mathrm{e}^{\mathrm{i}g(x)}$ for $x\in\mathbb{R}$ where $g$ is a phase function, and $g(x) - \omega x$ is the phase error of $r(\mathrm{i} x) \approx \mathrm{e}^{\mathrm{i}\omega x}$.
The phase error equioscillates at $2n+2$ nodes $\eta_1< \eta_2< \ldots <\eta_{2n+2}$ if $g(\eta_j) - \omega \eta_j = (-1)^{j+1} \max_{x\in[-1,1]}| g(x) - \omega x |$ for $j=1,\ldots,2n+2$.

E.g. the phase and approximation error of the unitary rational best approximation for $n=4$ and $\omega=2.65$ computed by the `brib` algorithm.
![errors](https://github.com/newbisi/rexpi/blob/main/docs/errors.png)

## Content

- The best rational interpolation based `brib` algorithm. This is a modified BRASIL algorithm to compute unitary best approximations to $\mathrm{e}^{\mathrm{i}\omega x}$. `r,_,_ = brib(w, n)` computes an $(n,n)$-rational approximation to $\mathrm{e}^{\mathrm{i}\omega x}$ for a given frequency $\omega$ and degree $n$.
```python
import numpy as np
import matplotlib.pyplot as plt
import rexpi
n = 10
w = 10
r, _, _ = rexpi.brib(w,n)

# plot the error
xs = np.linspace(-1,1,5000)
err = r(1j*xs)-np.exp(1j*w*xs)
errmax = np.max(np.abs(err))
plt.plot(xs,np.abs(err),[-1,1],[errmax,errmax],':')
```

- `linearizedLawson`:
A version of the AAA-Lawson method to approximate $\mathrm{e}^{\mathrm{i}\omega x}$ using Chebyshev nodes as pre-assigned support nodes. The approximation computed by the linearizedLawson algorithm is comparable with the approximation computed by the brib algorithm.
```python
n = 10
w = 10
r, _ = rexpi.linearizedLawson(w=w, n=n, nlawson=100,nx=1000)
```

- Error estimation. Best approximations uniquely exist for $\omega\in(0,\pi(n+1))$ and algorithms may fail when choosing $\omega >\pi(n+1)$. In addition, algorithms may fail if $\omega\approx 0$ due to limits of double precision arithmetic. We suggest using  the routine `buerrest` to determine $\omega$ s.t. the approximation error is bounded by a given tolerance, $\mathrm{tol}>10^{-16}$.
```python
n = 10
tol = 1e-6
w = rexpi.buerrest_getw(n, tol)
r, _, _ = rexpi.brib(w,n)
```
- The `brib` and `linearizedLawson` algorithms also utilize ideas presented in[^JS23] to preserve unitarity in computer arithmetic and to reduce computational cost. In particular, when computing approximations based on interpolation or on minimizing a linearized error.
- Applications of unitary best approximations to $\mathrm{e}^{\mathrm{i}\omega x}$ are approximations to the scalar exponential function and approximations to exponentials of skew-Hermitian matrices or the action of such matrix exponentials for numerical time integration, see `examples/ex5_matrixexponential.ipynb`.

## Citing `rexpi`

A full documentation of this package is currently not available. If you use unitary rational best approximations for your scientific publication, please cite *Unitary rational best approximations to the exponential function* by Jawecki, T. and Singh, P..
```
@unpublished{JS23a,
  author = {Jawecki, T. and Singh, P.},
  title = {Unitary rational best approximations to the exponential function},
  eprint = {2312.13809},
  year = 2023,
  note = {preprint at https://arxiv.org/abs/2312.13809}
}
```

[^JSxx]: T. Jawecki and P. Singh. Unitary rational best approximations to the exponential function. *to be published*. preprint available at [https://arxiv.org/abs/2312.13809](https://arxiv.org/abs/2312.13809).

[^JS23]: T. Jawecki and P. Singh. Unitarity of some barycentric rational approximants. *IMA J. Numer. Anal.*, 2023. [doi:10.1093/imanum/drad066](https://doi.org/10.1093/imanum/drad066).

[^Ho20]: C. Hofreither. An algorithm for best rational approximation based on barycentric rational interpolation. *Numer. Algorithms*, 88(1):365–388, 2021. [doi:10.1007/s11075-020-01042-0](https://doi.org/10.1007/s11075-020-01042-0).

[^NST18]: Y. Nakatsukasa, O. Sète, and L.N. Trefethen. The AAA algorithm for rational approximation. *SIAM J. Sci. Comput.*, 40(3):A1494–A1522, 2018. [doi:10.1137/16M1106122](https://doi.org/10.1137/16M1106122).

[^NT20]: Y. Nakatsukasa and L.N. Trefethen. An algorithm for real and complex rational minimax approximation. *SIAM J. Sci. Comput.*, 42(5):A3157–A3179, 2020. [doi:10.1137/19M1281897](https://doi.org/10.1137/19M1281897).

[^DHT14]: T.A. Driscoll, N. Hale, and L.N. Trefethen, editors. Chebfun Guide. Pafnuty Publications, Oxford, 2014. also available online from [https://www.chebfun.org](https://www.chebfun.org).

[^Fra76]: R. Franke. On the convergence of an algorithm for rational Chebyshev approximation. *Rocky Mountain J. Math.*, 6(2), 1976. [doi:10.1216/rmj-1976-6-2-227](https://doi.org/10.1216/rmj-1976-6-2-227).

[^Ma63]: H.J. Maehly. Methods for fitting rational approximations, parts II and III. *J. ACM*, 10(3):257–277, July 1963. [doi:10.1145/321172.321173](https://doi.org/10.1145/321172.321173).
