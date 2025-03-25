from functools import partial

import galsim as _galsim
import jax.lax
import jax.numpy as jnp
from quadax import quadgk
from quadax.utils import wrap_func, bounded_while_loop

from jax_galsim.core.utils import implements


@implements(
    _galsim.integ.int1d,
    lax_description=(
        """\
The JAX-GalSim package uses the adaptive Gauss-Kronrod-Patterson
method implemented in the ``quadax`` package. Some import caveats are: "

- This implementation is different than the one in GalSim and lacks some features that
  greatly enhance galsim's accuracy.
- The JAX-GalSim implementation returns NaN on error/non-convergence instead of
  rasing an exception.
"""
    ),
)
@partial(jax.jit, static_argnames=("func", "_wrap_as_callback"))
def int1d(
    func,
    min,
    max,
    rel_err=1.0e-6,
    abs_err=1.0e-12,
    _wrap_as_callback=False,
    _inf_cutoff=1e4,
):
    # the hidden _wrap_as_callback keyword is used for testing against galsim
    # if true, we assume the input function is pure python and wrap it so it
    # can be used with jax
    if _wrap_as_callback:

        @jax.jit
        def _func(x):
            rdt = jax.ShapeDtypeStruct(x.shape, x.dtype)
            return jax.pure_callback(func, rdt, x)
    else:
        _func = func

    _min = jax.lax.cond(
        jnp.abs(min) > _inf_cutoff,
        lambda: jnp.sign(min) * jnp.inf,
        lambda: jnp.float_(min),
    )
    _max = jax.lax.cond(
        jnp.abs(max) > _inf_cutoff,
        lambda: jnp.sign(max) * jnp.inf,
        lambda: jnp.float_(max),
    )

    def _split_inf_integration():
        # Split the integration into two parts
        val1, info1 = quadgk(_func, [_min, 0.0], epsabs=abs_err, epsrel=rel_err)
        val2, info2 = quadgk(_func, [0.0, _max], epsabs=abs_err, epsrel=rel_err)
        status = info1.status | info2.status
        return val1 + val2, status

    def _base_integration():
        val, info = quadgk(_func, [_min, _max], epsabs=abs_err, epsrel=rel_err)
        return val, info.status

    val, status = jax.lax.cond(
        jnp.isinf(_min) & jnp.isinf(_max),
        _split_inf_integration,
        _base_integration,
    )

    return jax.lax.cond(
        status == 0,
        lambda: val,
        lambda: jnp.nan,
    )


import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Callable, Any
from .bessel import get_j0_roots, j0, j1, y0


class AdaptiveOgataRuleOrderZero(eqx.Module):
    """Adaptive Hankel quadrature rule based on Ogata's method.

    Integrates: ∫₀^∞ r f(r) J_0(k r) dr

    Parameters
    ----------
    k : float
        Wavenumber k.
    h0 : float
        Initial spacing in Ogata summation.
    max_iter : int
        Maximum number of refinement steps (halving h).
    N : int
        Number of terms used in each Ogata summation (per side, so 2N+1 samples).
    """
    k: float
    h0: float = 1 / 32.0
    max_iter: int = 50
    N: int = 256
    
    def __init__(self, k: float, h0: float = 1 / 32.0, max_iter: int = 50, N: int = 256):
        self.k = k
        self.h0 = h0
        self.max_iter = max_iter
        self.N = N
        # TODO: Implement integration for non-zero order: Bessel jv (see bessel.py)

    def norm(self, x: jax.Array) -> float:
        return jnp.linalg.norm(x.flatten(), ord=jnp.inf)

    def psi(self, t):
        return t * jnp.tanh(0.5 * jnp.pi * jnp.sinh(t))
    
    def dpsi(self, t):
        return 0.5 * jnp.pi * t * jnp.cosh(t) / jnp.cosh(0.5 * jnp.pi * jnp.sinh(t))**2

    def _ogata_integrate(self, f: Callable, k: float, h: float, args: tuple[Any]) -> float:
        n = jnp.arange(1, self.N + 1)
        xi = get_j0_roots(n) / jnp.pi
        t = xi * h
        x = (jnp.pi / h) * self.psi(t)
        w = y0(jnp.pi*xi) / j1(jnp.pi*xi) * jnp.pi * x * j0(x) * self.dpsi(t)
        fx = f(x / k, *args)
        integrand = w * fx
        return jnp.sum(w * integrand) / k**2

    @eqx.filter_jit
    def integrate(
        self,
        fun: Callable,
        relerr: float = 1e-6,
        abserr: float = 1e-16,
        args: tuple[Any]
    ) -> tuple[float, float, float, float]:

        f = wrap_func(fun)

        k = args[0]  # assume k is passed as the first argument in `args`
        h = self.h0
        if h > 100 * k:
            h = 100 * k

        ans0 = self._ogata_integrate(f, k, h, args)
        h *= 0.5
        ans1 = self._ogata_integrate(f, k, h, args)
        err = jnp.abs(ans1 - ans0)
        iters = 0

        def cond(state):
            _, ans0, ans1, err, h, iters = state
            continue_ = ((err > relerr * jnp.abs(ans1)) &
                         ((err > abserr) | (jnp.abs(ans1) > 2 * jnp.abs(ans0))) |
                         (ans1 == 0.0)) & (iters < self.max_iter)
            return continue_

        def body(state):
            _, ans0, ans1, err, h, iters = state
            h *= 0.5
            ans0 = ans1
            ans1 = self._ogata_integrate(f, k, h, args)
            err = jnp.abs(ans1 - ans0)
            return ans1, ans0, ans1, err, h, iters + 1

        state = (ans1, ans0, ans1, err, h, iters)
        ans1, *_ = bounded_while_loop(cond, body, state, max_iter + 1)

        return ans1


def _psi(t):
    return t * jnp.tanh(0.5 * jnp.pi * jnp.sinh(t))


def _dpsi(t):
    return 0.5 * jnp.pi * t * jnp.cosh(t) / jnp.cosh(0.5 * jnp.pi * jnp.sinh(t))**2

@jax.jit
def _hankel_integrate_zero_order(f, k, h, n_nodes, args):
    n = jnp.arange(1, n_nodes + 1)
    xi = get_j0_roots(n) / jnp.pi
    t = xi * h
    x = (jnp.pi / h) * _psi(t)
    w = y0(jnp.pi*xi) / j1(jnp.pi*xi) * jnp.pi * x * j0(x) * _dpsi(t)
    fx = f(x / k, *args)
    integrand = w * fx
    return jnp.sum(w * integrand) / k**2

@jax.jit
def _ogata_adaptive_integrate_zero_order(fun, k, args, relerr, abserr, h0, n_nodes, max_iter):

    f = wrap_func(fun)

    h = h0
    if h > 100 * k:
        h = 100 * k

    ans0 = _hankel_integrate_zero_order(f, k, h, n_nodes, args)
    h *= 0.5
    ans1 = _hankel_integrate_zero_order(f, k, h, n_nodes, args)
    err = jnp.abs(ans1 - ans0)
    iters = 0

    def cond(state):
        _, ans0, ans1, err, h, iters = state
        continue_ = ((err > relerr * jnp.abs(ans1)) &
                        ((err > abserr) | (jnp.abs(ans1) > 2 * jnp.abs(ans0))) |
                        (ans1 == 0.0))
        return continue_

    def body(state):
        _, ans0, ans1, err, h, iters = state
        h *= 0.5
        ans0 = ans1
        ans1 = _hankel_integrate_zero_order(f, k, h, n_nodes, args)
        err = jnp.abs(ans1 - ans0)
        return ans1, ans0, ans1, err, h, iters + 1

    state = (ans1, ans0, ans1, err, h, iters)
    ans1, *_ = bounded_while_loop(cond, body, state, max_iter + 1)

    return ans1

@jax.jit
def hankel_inf_zero_order(func, k, args, relerr, abserr, h0=1/32.0, max_iter=50, n_nodes=256):
    """Integrate a function from 0 to infinity using the Ogata method.

    Parameters
    ----------
    func : callable
        Function to integrate.
    k : array_like
        Wavenumber.
    args : tuple
        Extra arguments to pass to the function.
    relerr : float
        Relative error tolerance.
    abserr : float
        Absolute error tolerance.
    h0 : float
        Initial spacing in Ogata summation.
    max_iter : int
        Maximum number of refinement steps (halving h).
    n_nodes : int
        Number of terms used in each Ogata summation.

    Returns
    -------
    float
        Integral of the function.
    """
    vec_integ = jax.vmap(lambda x: _ogata_adaptive_integrate_zero_order(func, x, args, relerr, abserr, h0, n_nodes, max_iter))

    return vec_integ(k)


def hankel_trunc_zero_order(func, k, rmax, args, relerr, abserr, h0=1/32.0, max_iter=50, n_nodes=256):
    """Integrate a function from 0 to truncation radius using the GK method.

    Parameters
    ----------
    func : callable
        Function to integrate.
    k : array_like
        Wavenumber.
    trunc : float
        Truncation radius.
    args : tuple
        Extra arguments to pass to the function.
    relerr : float
        Relative error tolerance.
    abserr : float
        Absolute error tolerance.
    h0 : float
        Initial spacing in Ogata summation.
    max_iter : int
        Maximum number of refinement steps (halving h).
    n_nodes : int
        Number of terms used in each Ogata summation.

    Returns
    -------
    float
        Integral of the function.
    """
    def integrand(r, *args):
        return r * func(r, *args) * j0(k * r)

    return int1d(integrand, 0, rmax, rel_err=relerr, abs_err=abserr, _wrap_as_callback=True)