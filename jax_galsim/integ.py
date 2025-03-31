from functools import partial

import galsim as _galsim
import jax
import jax.lax
import jax.numpy as jnp
from quadax import quadgk
from quadax.utils import bounded_while_loop, wrap_func

from jax_galsim.core.utils import implements
from jax_galsim.bessel import get_j0_roots, j0, j1, y0


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


def _psi(t):
    return t * _psi_t(t)


def _psi_t(t):
    return jnp.tanh(0.5 * jnp.pi * jnp.sinh(t))


def _dpsi(t):
    return 0.5 * jnp.pi * t * jnp.cosh(t) / jnp.cosh(0.5 * jnp.pi * jnp.sinh(t)) ** 2 + _psi_t(t)


@partial(jax.jit, static_argnames=("n_nodes",))
def _hankel_integrate_zero_order(f, k, h, n_nodes, args):
    n = jnp.arange(1, n_nodes + 1)
    xi = get_j0_roots(n) / jnp.pi
    t = xi * h
    # x = (jnp.pi / h) * _psi(t)
    x = jnp.pi * _psi_t(t) * xi
    w = y0(jnp.pi * xi) / j1(jnp.pi * xi) * jnp.pi * x * j0(x) * _dpsi(t)
    fx = f(x / k, *args)
    integrand = w * fx

    return jnp.sum(integrand) / k**2


@partial(jax.jit, static_argnames=("max_iter", "n_nodes",))
def _ogata_adaptive_integrate_zero_order(
    fun, k, args, relerr, abserr, h0, n_nodes, max_iter
):

    # f = wrap_func(fun, args)
    f = fun
    h = h0

    # Replacing C++ while loop: `while (h0 > 100*k) h0 *= 0.5;`
    factor = jnp.maximum(0, jnp.ceil(jnp.log2(h0 / (100 * k))))
    h = h0 * 0.5 ** factor

    n_nodes = int(n_nodes)
    ans0 = _hankel_integrate_zero_order(f, k, h, n_nodes, args)
    h *= 0.5
    ans1 = _hankel_integrate_zero_order(f, k, h, n_nodes, args)
    err = jnp.abs(ans1 - ans0)
    iters = 0

    def cond(state):
        _, ans0, ans1, err, h, iters = state
        continue_ = ((err > relerr * jnp.abs(ans1)) & (
            (err > abserr) | (jnp.abs(ans1) > 2 * jnp.abs(ans0))
        ) | (ans1 == 0.0)) & (iters <= max_iter)
        return continue_

    def body(state):
        _, ans0, ans1, err, h, iters = state
        h *= 0.5
        ans0 = ans1
        ans1 = _hankel_integrate_zero_order(f, k, h, n_nodes, args)
        err = jnp.abs(ans1 - ans0)
        return ans1, ans0, ans1, err, h, iters + 1

    state = (ans1, ans0, ans1, err, h, iters)
    # ans1, *_ = bounded_while_loop(cond, body, state, max_iter + 1)
    # state = bounded_while_loop(cond, body, state, max_iter + 1)
    state = jax.lax.while_loop(cond, body, state)

    return state


@partial(jax.jit, static_argnames=("max_iter", "n_nodes"))
def hankel_inf_zero_order(
    func, k, args, relerr, abserr, h0=1 / 32.0, max_iter=50, n_nodes=256
):
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
    vec_integ = jax.vmap(
        lambda x: _ogata_adaptive_integrate_zero_order(
            func, x, args, relerr, abserr, h0, n_nodes, max_iter
        )
    )

    return vec_integ(k)


def hankel_trunc_zero_order(
    func, k, rmax, args, relerr, abserr, h0=1 / 32.0, max_iter=50, n_nodes=256
):
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

    return int1d(
        integrand, 0, rmax, rel_err=relerr, abs_err=abserr, _wrap_as_callback=True
    )
