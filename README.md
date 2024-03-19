# rexpi [![PyPI version](https://badge.fury.io/py/rexpi.svg)](https://badge.fury.io/py/rexpi)
*by Tobias Jawecki*

**This package is still under development. Future versions might have different API. The current version is not fully documented.**

Besides new code, this package also contains some routines of the [baryrat](https://github.com/c-f-h/baryrat) package[^Ho20], and variants of the AAA-Lawson[^NST18][^NT20] method which is part of the [Chebfun](http://www.chebfun.org/) package[^DHT14].

## Geting started
Install the package `python -m pip install rexpi` and run numerical examples from the `/examples` folder.

## Mathematical problem description

We consider rational approximations $$r(\mathrm{i} x) \approx \mathrm{e}^{\mathrm{i}\omega x},\quad\quad\textrm{for}\quad x\in[-1,1]\quad\textrm{and a frequency}\quad\omega>0,$$ where $r$ is a unitary $(n,n)$-rational function, i.e., $r=p/q$ where $p$ and $q$ are polynomials of degree $\leq n$ and $|r(\mathrm{i} x)|=1$ for $x\in\mathbb{R}$. Our focus is best approximations in a Chebyshev sense which minimize the uniform error $$\max_{x\in[-1,1]}| r(\mathrm{i} x) - \mathrm{e}^{\mathrm{i}\omega x} |.$$

Following a recent work[^JSxx], the unitary best approximation to $\mathrm{e}^{\mathrm{i}\omega x}$ is uniquely characterized by an equioscillating phase error. In particular, unitary rational functions are of the form $r(\mathrm{i} x) = \mathrm{e}^{\mathrm{i}g(x)}$ for $x\in\mathbb{R}$ where $g$ is a phase function, and we introduce the phase error $g(x) - \omega x$ in addition to the approximation error $r(\mathrm{i} x) - \mathrm{e}^{\mathrm{i}\omega x}$. The phase and approximation error satisfy the relation $|r(\mathrm{i} x) - \mathrm{e}^{\mathrm{i}\omega x}| = 2|\sin((g(x) - \omega x)/2)|$, and for the case that the approximation error is non-maximal, i.e., $\max_{x\in[-1,1]}| r(\mathrm{i} x) - \mathrm{e}^{\mathrm{i}\omega x} |<2$, the best unitary rational approximation to $\mathrm{e}^{\mathrm{i}\omega x}$ corresponds to the phase function $g$ which minimizes the phase error $\max_{x\in[-1,1]} |g(x) - \omega x|$. Following[^JSxx], the corresponding phase error $g(x)-\omega x$ equioscillates at $2n+2$ nodes $\eta_1< \eta_2< \ldots <\eta_{2n+2}$, namely, $$g(\eta_j) - \omega \eta_j = (-1)^{j+1} \max_{x\in[-1,1]}| g(x) - \omega x |,\quad\quad j=1,\ldots,2n+2.$$ In the present package we introduce numerical algorithms to compute unitary best approximations mostly based on consequences of this equioscillation property, and certainly relying on the result that the unitary best approximation is unique and non-degenerate.

As a consequence of the equioscillation property, the unitary best approximation attains $2n+1$ points of interpolation $x_1 <\ldots < x_{2n+1}$ in the sense of $r(\mathrm{i} x_j) = \mathrm{e}^{\mathrm{i}\omega x_j}$. The existence of such interpolation nodes motivates the `brib` algorithm introduced in the present package - a modified BRASIL algorithm which aims to compute a uniform best approximation by rational interpolation and by iteratively adapting the underlying interpolation nodes. In addition, the present package also provides a variant of the AAA-Lawson method to compute approximations to $\mathrm{e}^{\mathrm{i}\omega x}$. It has been shown recently in[^JS23] that rational interpolants and rational approximations constructed by the AAA-Lawson algorithm are unitary, and thus, these approaches fit well to compute unitary rational best approximations.

#### Numerical illustration of the equioscillating phase error
![phaseerror](https://github.com/newbisi/rexpi/blob/main/docs/phaseerror.png)

#### Numerical illustration of the corresponding approximation error
![approxerror](https://github.com/newbisi/rexpi/blob/main/docs/approxerror.png)

## Content

- The best rational interpolation based `brib` algorithm. This is a modified BRASIL algorithm to compute unitary best approximations to $\mathrm{e}^{\mathrm{i}\omega x}$. `r,_,_ = brib(w, n)` computes an $(n,n)$-rational approximation to $\mathrm{e}^{\mathrm{i}\omega x}$ for $x\in[-1,1]$ in barycentric rational form. For a start, we recommend using `n=10` and `w = buerrest_getw(n, tol)` for the frequency $\omega$ with `tol = 1e-6`.
```python
import numpy as np
import matplotlib.pyplot as plt
import rexpi
n=10
tol=1e-6
w = rexpi.buerrest_getw(n, tol)
r, _, _ = rexpi.brib(w,n)
xs = np.linspace(-1,1,5000)
err = r(1j*xs)-np.exp(1j*w*xs)
errmax = np.max(np.abs(err))
plt.plot(xs,np.abs(err),[-1,1],[errmax,errmax],':')
```

- `linearizedLawson`:
A version of the AAA-Lawson method to approximate $\mathrm{e}^{\mathrm{i}\omega x}$ using Chebyshev nodes as pre-assigned support nodes. The approximation computed by the linearizedLawson algorithm is comparable with the approximation computed by the brib algorithm, however, the latter seems to perform better in various settings. A basic example for the linearizedLawson algorithm:
```python
import numpy as np
import matplotlib.pyplot as plt
import rexpi
n=10
tol=1e-6
w = rexpi.buerrest_getw(n, tol)
r, _ = rexpi.linearizedLawson(w=w, n=n, nlawson=100,nx=1000)
xs = np.linspace(-1,1,5000)
err = r(1j*xs)-np.exp(1j*w*xs)
errmax = np.max(np.abs(err))
plt.semilogy(xs,np.abs(err),[-1,1],[errmax,errmax],':')
```

- Error estimation. This package contains implementations of a priori estimates `buerrest`, `buerrest_getw`, and `buerrest_getn` for the degree $n$ and the frequency $\omega$ s.t. the approximation error is bounded by a given tolerance.
- Stopping criteria for the `brib` algorithm. In case the maximal number of iterations in the birb algorithm is not reached, then this algorithm returns a rational approximation with *relative deviation* $\delta$ smaller than `tolequi=1e-3`. We use a slightly different definition of the relative deviation compared to the BRASIL algorithm. The relative deviation is defined as $$\delta = 1-\frac{\min_{j=1,\ldots,2n+2} \alpha_j}{\max_{j=1,\ldots,2n+2} \alpha_j},$$ where $\alpha_j>0$ refer to the magnitude of the phase error at non-uniform alternating points $\tau_j$ which are local maxima or minima, including boundary points $x=-1$ and $x=1$, with phase error $$g(\tau_j) - \omega \tau_j = (-1)^{j+\iota} \alpha_j,\quad\quad j=1,\ldots,2n+2,\quad \iota\in\{0,1\}.$$ Using the relative deviation as a stopping criteria shows to be mathematical justified. [A variant of lower bounds on the minimax error sometimes referred to de la Vallée-Poussin] Assuming we have $2n+2$ non-uniform alternating points, then the phase error of the unitary best approximation satisfies $$\varepsilon_\star = \min_{g~:~ r(\mathrm{i} x) = \mathrm{e}^{\mathrm{i}g(x)},~ r\in U_n}~ \max_{x\in[-1,1]} |g(x)-\omega x| \geq \min_{j=1,\ldots,2n+2} \alpha_j.$$ Let $\varepsilon>0$ denote the phase error of the computed approximation. Since the phase error $\varepsilon$ is either attained at a local maxima, minima or at the boundary, we have $\varepsilon = \max_{j} \alpha_j$. Thus, $$\varepsilon \geq \varepsilon_\star \geq \varepsilon (1-\delta).$$ Thus, a computed approximation with a small relative deviation $\delta$ has a phase error (and approximation error) close to the minimal achievable phase error (approximation error).
- The brib algorithm is based on rational interpolation and the Lawson method is based on minimizing a linearized error. In addition to ideas in[^JS23] to preserve unitarity of such approximations in computer arithmetic, we introduce algorithms to also preserve symmetry in computer arithmetic, i.e., $r(-\mathrm{i} x) = r(\mathrm{i} x)^{-1}$ for $x\in\mathbb{R}$, a property which holds true for the unitary best approximation. Symmetry and unitarity are preserved when computing weights in barycentric rational form and when computing the poles of the constructed barycentric rational function, i.e., poles are computed as real poles or complex conjugate pairs of poles by design.
- Applications of these algorithms are approximations to the scalar exponential function and approximations to exponentials of skew-Hermitian matrices or the action of such matrix exponentials.

[^JSxx]: T. Jawecki and P. Singh. Unitary rational best approximations to the exponential function. *to be published*. preprint available at [https://arxiv.org/abs/2312.13809](https://arxiv.org/abs/2312.13809).

[^JS23]: T. Jawecki and P. Singh. Unitarity of some barycentric rational approximants. *IMA J. Numer. Anal.*, 2023. [doi:10.1093/imanum/drad066](https://doi.org/10.1093/imanum/drad066).

[^Ho20]: C. Hofreither. An algorithm for best rational approximation based on barycentric rational interpolation. *Numer. Algorithms*, 88(1):365–388, 2021. [doi:10.1007/s11075-020-01042-0](https://doi.org/10.1007/s11075-020-01042-0).

[^NST18]: Y. Nakatsukasa, O. Sète, and L.N. Trefethen. The AAA algorithm for rational approximation. *SIAM J. Sci. Comput.*, 40(3):A1494–A1522, 2018. [doi:10.1137/16M1106122](https://doi.org/10.1137/16M1106122).

[^NT20]: Y. Nakatsukasa and L.N. Trefethen. An algorithm for real and complex rational minimax approximation. *SIAM J. Sci. Comput.*, 42(5):A3157–A3179, 2020. [doi:10.1137/19M1281897](https://doi.org/10.1137/19M1281897).

[^DHT14]: T.A. Driscoll, N. Hale, and L.N. Trefethen, editors. Chebfun Guide. Pafnuty Publications, Oxford, 2014. also available online from [https://www.chebfun.org](https://www.chebfun.org).
