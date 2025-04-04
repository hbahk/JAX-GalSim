import galsim as _galsim
import jax
import jax.numpy as jnp
from interpax import interp1d
from jax import jit, vmap
from jax.scipy.special import gamma, gammainc
from jax.tree_util import Partial as partial
from jax.tree_util import register_pytree_node_class

from jax_galsim.core.draw import draw_by_kValue, draw_by_xValue
from jax_galsim.core.utils import bisect_for_root, ensure_hashable, implements
from jax_galsim.gsobject import GSObject
from jax_galsim.gsparams import GSParams
from jax_galsim.integ import hankel_inf_zero_order, hankel_trunc_zero_order
from jax_galsim.random import UniformDeviate
from jax_galsim.utilities import lazy_property


class SersicMissingFlux:
    def __init__(self, n, missing_flux):
        self._2n = 2.0 * n
        self._target = missing_flux

    def __call__(self, z):
        f = (1.0 - gammainc(self._2n, z)) * gamma(self._2n)
        return f - self._target


class SersicTruncatedHLR:
    def __init__(self, n, x):
        self._2n = 2.0 * n
        self._x = x

    def __call__(self, b):
        f1 = gammainc(self._2n, b)
        f2 = gammainc(self._2n, self._x * b)
        return (2 * f1 - f2) * gamma(self._2n)


@jit
def calculate_b(n, invn, gamma2n, flux_fraction):
    """
    Compute the parameter b that satisfies the given truncated gamma function condition.
    Uses an initial approximation from Ciotti & Bertin (1999) and refines it using root finding.

    Parameters:
    n (float): Sersic index
    invn (float): 1/n for efficiency
    gamma2n (float): Truncated gamma function value
    flux_fraction (float): Fraction of flux enclosed

    Returns:
    float: Solution for b
    """
    invnsq = invn * invn
    b1 = 2.0 * n - 1.0 / 3.0
    b2 = (
        b1
        + (8.0 / 405.0) * invn
        + (46.0 / 25515.0) * invnsq
        + (131.0 / 1148175.0) * invn * invnsq
    )

    missing_flux = (1.0 - 0.5 * flux_fraction) * gamma2n
    func = SersicMissingFlux(n, missing_flux)
    pfunc = partial(func)

    result = bisect_for_root(pfunc, b1, b2)

    return result


def sersic_integrated_flux(n, r):
    """
    Calculate the integrated flux of a Sersic profile out to a given radius.

    Parameters:
    n (float): Sersic index
    r (float): Radius to integrate to (in units of the scale radius)

    Returns:
    float: Integrated flux
    """
    z = r ** (1.0 / n)
    return gammainc(2.0 * n, z)


@jit
def calculate_truncated_scale(n, invn, b, trunc):
    """
    Calculate the truncated scale radius for a Sersic profile.

    Parameters:
    n (float): Sersic index
    invn (float): 1/n for efficiency
    b (float): Parameter b
    trunc (float): Truncation radius in units of the half-light radius

    Returns:
    float: Truncated scale radius in units of the half-light radius
    """

    x = trunc**invn

    b1 = (jnp.log(0.5) + (2 * n - 1) * jnp.log(x)) / (x - 1)

    b1 = jax.lax.cond(
        b1 < 1.0e-3 * b,
        lambda b1: b / 2,
        lambda b1: b1,
        operand=b1,
    )

    b2 = b
    func = SersicTruncatedHLR(n, x)
    pfunc = partial(func)
    b_result = bisect_for_root(pfunc, b1, b2)

    return 1.0 / b_result**n  # r0 = re / b^n


def sersic_radial_function(r, invn):
    """
    Calculate the Sersic radial profile at a given radius.

    Parameters:
    r (float): Radius at which to evaluate the profile.
    invn (float): 1/n for efficiency.

    Returns:
    float: Value of the Sersic profile at radius r.
    """
    return jnp.exp(-jnp.power(r, invn))


# @partial(jit, static_argnames=("n_ksteps",))
def _build_FT(
    n,
    trunc_factor,
    flux_fraction,
    gamma2n,
    kvalue_accuracy,
    table_spacing,
    integration_relerr,
    integration_abserr,
    maxk_threshold,
    n_ksteps=1000,
):

    def compute_gammaN(p):
        z = trunc_factor ** (1.0 / n)
        return jax.lax.cond(
            (trunc_factor > 0) & (flux_fraction < 1.0),
            lambda: gammainc(p, z) * gamma(p),
            lambda: gamma(p),
        )

    gamma4n = compute_gammaN(4.0 * n)
    gamma6n = compute_gammaN(6.0 * n)
    gamma8n = compute_gammaN(8.0 * n)

    kderiv2 = -gamma4n / (4.0 * gamma2n) / flux_fraction
    kderiv4 = gamma6n / (64.0 * gamma2n) / flux_fraction
    kderiv6 = gamma8n / (2304.0 * gamma2n) / flux_fraction

    kmin = (kvalue_accuracy / kderiv6) ** (1.0 / 6.0)
    ksq_min = kmin * kmin

    hankel_norm = flux_fraction * n * gamma2n
    dlogk_desired = table_spacing * jnp.sqrt(jnp.sqrt(kvalue_accuracy / 10.0))

    # logk = jnp.arange(jnp.log(kmin) - 0.001, jnp.log(500.0), dlogk)
    logk = jnp.linspace(jnp.log(kmin) - 0.001, jnp.log(500.0), n_ksteps)
    dlogk = (jnp.log(500.0) - (jnp.log(kmin) - 0.001)) / n_ksteps
    k = jnp.exp(logk)
    ksq = k**2

    f = partial(sersic_radial_function, invn=1.0 / n)

    f_vals = jax.lax.cond(
        trunc_factor > 0,
        lambda: hankel_trunc_zero_order(
            f,
            k,
            trunc_factor,
            (),
            relerr=integration_relerr,
            abserr=integration_abserr * hankel_norm,
        ),
        lambda: hankel_inf_zero_order(
            f,
            k,
            (),
            relerr=integration_relerr,
            abserr=integration_abserr * hankel_norm,
        ),
    )

    f_vals /= hankel_norm
    f0_vals = f_vals * ksq

    # Fit a/k^2 + b/k^3 to last `n_fit` values
    n_fit = 10
    tail_idx = -n_fit
    inv_k = 1.0 / k[tail_idx:]
    f0 = f0_vals[tail_idx:]

    A = jnp.stack([jnp.ones_like(inv_k), inv_k], axis=1)
    coeffs, *_ = jnp.linalg.lstsq(A, f0, rcond=None)  # [a, b]
    a, b = coeffs

    # Check if we need to use a larger maxk
    thres = maxk_threshold
    found_maxk = jnp.any(f0_vals < thres)

    _approx_k_at_thres = jnp.sqrt(
        (
            a
            - b
            / jnp.sqrt(
                (a - b / jnp.sqrt((a - b / jnp.sqrt(a / thres)) / thres)) / thres
            )
        )
        / thres
    )

    # Predict f0 from high-k approx
    f0_pred = a + b / k
    resid = jnp.abs(f0_vals - f0_pred) / ksq
    within_tol = resid < kvalue_accuracy
    has_converged = jnp.any(within_tol)

    # Find the first k value where high-k approx becomes good
    ksq_max_idx = jnp.argmax(within_tol)
    buffer = 5
    ksq_max_idx_buffered = jnp.minimum(ksq_max_idx + buffer, len(k) - 1)

    ksq_max = jnp.where(has_converged, ksq[ksq_max_idx_buffered], ksq[-1])
    maxk = jnp.sqrt(ksq_max)

    return {
        "kmin": kmin,
        "ksq_min": ksq_min,
        "ksq_max": ksq_max,
        "maxk": maxk,
        "kderiv2": kderiv2,
        "kderiv4": kderiv4,
        "highk_a": a,
        "highk_b": b,
        "found_maxk": found_maxk,
        "_approx_k_at_thres": _approx_k_at_thres,
        "ft_table_logk": logk,
        "ft_table_fvals": f_vals,
        "is_dlogk_good": dlogk < dlogk_desired,
        "dlogk": dlogk,
        "dlogk_desired": dlogk_desired,
    }


@jit
def _calculate_missing_flux_radius(n, gamma2n, b, missing_flux_frac):
    """
    Find the radius enclosing (1 - missing_flux_frac) of the total flux in a Sersic profile.

    Parameters:
    missing_flux_frac (float): Fraction of flux that is missing (not enclosed)

    Returns:
    float: The radius R that encloses (1 - missing_flux_frac) of the total flux
    """
    missing_flux = missing_flux_frac * gamma2n
    z1 = -jnp.log(missing_flux)

    def case_n_half():
        return z1  # Exact for n = 0.5

    def case_general():
        z_init = 4.0 * (n + 1.0)
        twonm1 = 2.0 * n - 1.0

        z2 = (
            z1
            + twonm1 * jnp.log(z_init)
            + twonm1 / z_init
            + (twonm1 * (2.0 * n - 3.0)) / (2.0 * z_init * z_init)
        )

        z2 = jax.lax.cond(
            (z2 > z1) & ((z2 - z1) < 0.01),
            lambda: z1 + 0.01,
            lambda: z2,
        )
        z2 = jax.lax.cond(
            (z2 < z1) & ((z2 - z1) > -0.01),
            lambda: z1 - 0.01,
            lambda: z2,
        )

        z1_new = jax.lax.cond(z1 < 0.0, lambda: b, lambda: z1)

        func = SersicMissingFlux(n, missing_flux).__call__
        z_root = bisect_for_root(partial(func), z1_new, z2)
        return z_root

    z = jax.lax.cond(
        n == 0.5,
        case_n_half,
        case_general,
    )

    return z**n


def parse_radius_options(
    n, gamma2n, trunc, flux_untruncated, half_light_radius, scale_radius, flux
):
    sqrt2 = jnp.sqrt(2.0)

    # Check for mutually exclusive condition
    if half_light_radius is not None and scale_radius is not None:
        raise _galsim.GalSimIncompatibleValuesError(
            "Only one of scale_radius or half_light_radius may be specified for Sersic",
            half_light_radius=half_light_radius,
            scale_radius=scale_radius,
        )

    if half_light_radius is None and scale_radius is None:
        raise _galsim.GalSimIncompatibleValuesError(
            "Either scale_radius or half_light_radius must be specified for Sersic",
            half_light_radius=half_light_radius,
            scale_radius=scale_radius,
        )

    use_hlr = half_light_radius is not None
    hlr = half_light_radius if use_hlr else None

    if use_hlr:

        def hlr_branch():
            def untrunc_branch():
                b = calculate_b(n, 1.0 / n, gamma2n, 1.0)
                r0 = hlr / b**n
                return r0, flux, b

            def trunc_branch():
                too_small = trunc <= sqrt2 * hlr

                def raise_trunc_err():
                    return jnp.nan, jnp.nan, jnp.nan  # to raise an error in case of invalid truncation value

                def good_trunc():
                    b = calculate_b(n, 1.0 / n, gamma2n, 1.0)
                    r0 = hlr * calculate_truncated_scale(n, 1.0 / n, b, trunc / hlr)
                    return r0, flux, b

                return jax.lax.cond(too_small, raise_trunc_err, good_trunc)

            is_untrunc = (trunc == 0.0) or flux_untruncated
            return jax.lax.cond(is_untrunc, untrunc_branch, trunc_branch)

        r0, flux, b = hlr_branch()

    else:  # use scale_radius
        r0 = scale_radius
        b = calculate_b(n, 1.0 / n, gamma2n, 1.0)

    # Now compute flux_fraction and potentially update flux
    def trunc_branch_flux():
        flux_fraction = sersic_integrated_flux(n, trunc / r0)

        def update_flux():
            return flux * flux_fraction

        def keep_flux():
            return flux

        flux_new = jax.lax.cond(flux_untruncated, update_flux, keep_flux)
        return flux_fraction, flux_new

    def untrunc_branch_flux():
        return 1.0, flux

    flux_fraction, flux = jax.lax.cond(
        trunc > 0.0, trunc_branch_flux, untrunc_branch_flux
    )

    # Final half-light radius
    b = calculate_b(n, 1.0 / n, gamma2n, flux_fraction)
    hlr = r0 * b**n

    return r0, hlr, flux, flux_fraction, b


@implements(_galsim.Sersic)
@register_pytree_node_class
class Sersic(GSObject):
    _req_params = {"n": float}
    _opt_params = {"flux": float, "trunc": float, "flux_untruncated": bool}
    _single_params = [{"scale_radius": float, "half_light_radius": float}]

    _is_axisymmetric = True
    _is_analytic_x = True
    _is_analytic_k = True

    # _minimum_n = 0.3  # Lower bounds has hard limit at ~0.29
    # _maximum_n = 6.2  # Upper bounds is just where we have tested that code works well.

    def __init__(
        self,
        n,
        half_light_radius=None,
        scale_radius=None,
        flux=1.0,
        trunc=0.0,
        flux_untruncated=False,
        gsparams=None,
        n_ksteps=1000,
    ):

        _gsparams = GSParams.check(gsparams)
        gamma2n = gamma(2.0 * n)

        def raise_trunc_invalid():
            return jnp.nan  # to raise an error in case of invalid truncation value

        def ok_trunc():
            return trunc

        trunc = jax.lax.cond(trunc < 0, raise_trunc_invalid, ok_trunc)

        # Parse the radius options
        r0, hlr, flux, flux_fraction, b = parse_radius_options(
            n,
            gamma2n,
            trunc,
            flux_untruncated,
            half_light_radius,
            scale_radius,
            flux,
        )

        # Initialize the FT lookup table
        trunc_factor = trunc / r0
        ft_result = _build_FT(
            n,
            trunc_factor,
            flux_fraction,
            gamma2n,
            _gsparams.kvalue_accuracy,
            _gsparams.table_spacing,
            _gsparams.integration_relerr,
            _gsparams.integration_abserr,
            _gsparams.maxk_threshold,
            n_ksteps,
        )

        # Initialize the stepk values
        R = _calculate_missing_flux_radius(n, gamma2n, b, _gsparams.folding_threshold)
        R = jax.lax.cond(
            flux_fraction < 1.0 and trunc_factor < R, lambda: trunc_factor, lambda: R
        )

        # Make sure it is at least 5 hlr
        R = jnp.max(jnp.array([R, _gsparams.stepk_minimum_hlr])) # TODO: check this. shouldn't this be stepk_minimum_hlr * hlr?
        stepk = jnp.pi / R / r0

        super().__init__(
            n=n,
            scale_radius=r0,
            flux=flux,
            trunc=trunc,
            gsparams=gsparams,
        )

        self._ft = ft_result
        self.__stepk = stepk
        self.__flux_fraction = flux_fraction

    def calculateIntegratedFlux(self, r):
        """Return the fraction of the total flux enclosed within a given radius, r"""
        return sersic_integrated_flux(self._n, float(r) / self._r0)

    def calculateHLRFactor(self):
        """Calculate the half-light-radius in units of the scale radius."""
        return self._b**self._n

    @property
    def n(self):
        """The Sersic parameter n."""
        return self._n

    @property
    def _n(self):
        return self.params["n"]

    @property
    def scale_radius(self):
        """The scale radius."""
        return self._r0

    @property
    def _r0(self):
        return self.params["scale_radius"]

    @property
    def trunc(self):
        """The truncation radius (if any)."""
        return self._trunc

    @property
    def _trunc(self):
        return self.params["trunc"]

    @property
    def trunc_factor(self):
        """The truncation factor."""
        return self._trunc / self._r0

    @property
    def half_light_radius(self):
        """The half-light radius."""
        return self._r0 * self.calculateHLRFactor()

    @property
    def gamma2n(self):
        return gamma(2.0 * self._n)

    @property
    def _flux_fraction(self):
        # return self.params["flux_fraction"]
        return self.__flux_fraction

    @property
    def _b(self):
        return calculate_b(self._n, 1.0 / self._n, self.gamma2n, self._flux_fraction)

    def __eq__(self, other):
        return self is other or (
            isinstance(other, Sersic)
            and self.n == other.n
            and self.scale_radius == other.scale_radius
            and self.trunc == other.trunc
            and self.flux == other.flux
            and self.gsparams == other.gsparams
        )

    def __hash__(self):
        return hash(
            (
                "galsim.SBSersic",
                ensure_hashable(self.n),
                ensure_hashable(self.scale_radius),
                ensure_hashable(self.trunc),
                ensure_hashable(self.flux),
                self.gsparams,
            )
        )

    def __repr__(self):
        return (
            "galsim.Sersic(n=%r, scale_radius=%r, trunc=%r, flux=%r, gsparams=%r)"
            % (
                ensure_hashable(self.n),
                ensure_hashable(self.scale_radius),
                ensure_hashable(self.trunc),
                ensure_hashable(self.flux),
                self.gsparams,
            )
        )

    def __str__(self):
        # Note: for the repr, we use the scale_radius, since that should just flow as is through
        # the constructor, so it should be exact.  But most people use half_light_radius
        # for Sersics, so use that in the looser str() function.
        s = "galsim.Sersic(n=%s, half_light_radius=%s" % (
            ensure_hashable(self.n),
            ensure_hashable(self.half_light_radius),
        )
        if self.trunc != 0.0:
            s += ", trunc=%s" % ensure_hashable(self.trunc)
        if self.flux != 1.0:
            s += ", flux=%s" % ensure_hashable(self.flux)
        s += ")"
        return s

    def __getstate__(self):
        d = self.__dict__.copy()
        # d.pop("_sbp", None)
        return d

    def __setstate__(self, d):
        self.__dict__ = d

    @property
    def _maxk(self):
        return self._ft["maxk"]

    @property
    def _stepk(self):
        return self.__stepk

    @property
    def _has_hard_edges(self):
        return self._trunc != 0.0

    def get_xnorm(self):
        return 1.0 / (2.0 * jnp.pi * self._n * self.gamma2n * self._flux_fraction)

    @property
    def _shootnorm(self):
        return self.get_xnorm() * self._flux

    @property
    def _max_sb(self):
        _inv_r0_sq = 1.0 / self._r0**2
        return _inv_r0_sq * self._shootnorm

    def _xValue(self, pos):
        rsq = (pos.x**2 + pos.y**2) / self._r0**2
        _truncated = (self.trunc_factor > 0) and (rsq > self.trunc**2)
        _xvalue = (
            jnp.select(
                [_truncated, ~_truncated],
                [0.0, jnp.exp(-jnp.power(rsq, 0.5 / self._n))],
            )
            * self._max_sb
        )

        return _xvalue

    def _kValue(self, kpos):
        ksq = (kpos.x**2 + kpos.y**2) * self._r0**2

        _kvalue = jnp.select(
            [ksq < self._ft["ksq_min"], ksq >= self._ft["ksq_max"]],
            [
                1.0 + ksq * (self._ft["kderiv2"] + ksq * self._ft["kderiv4"]),
                (self._ft["highk_a"] + self._ft["highk_b"] / jnp.sqrt(ksq)) / ksq,
            ],
            interp1d(
                0.5 * jnp.log(ksq),
                self._ft["ft_table_logk"],
                self._ft["ft_table_fvals"],
            )
            / ksq,
        )

        return _kvalue * self._flux

    def _shoot(self, photons, rng):
        raise NotImplementedError(
            "Sersic profiles are not yet implemented in the shooting API."
        )
        # self._sbp.shoot(photons._pa, rng._rng)

    def _drawReal(self, image, jac=None, offset=(0.0, 0.0), flux_scaling=1.0):
        _jac = jnp.eye(2) if jac is None else jac
        return draw_by_xValue(self, image, _jac, jnp.asarray(offset), flux_scaling)

    def _drawKImage(self, image, jac=None):
        _jac = jnp.eye(2) if jac is None else jac
        return draw_by_kValue(self, image, _jac)

    @implements(_galsim.Sersic.withFlux)
    def withFlux(self, flux):
        return Sersic(
            n=self.n,
            scale_radius=self.scale_radius,
            trunc=self.trunc,
            flux=flux,
            gsparams=self.gsparams,
        )


@implements(_galsim.DeVaucouleurs)
@register_pytree_node_class
class DeVaucouleurs(Sersic):
    _req_params = {}
    _opt_params = {"flux": float, "trunc": float, "flux_untruncated": bool}
    _single_params = [{"scale_radius": float, "half_light_radius": float}]

    def __init__(
        self,
        half_light_radius=None,
        scale_radius=None,
        flux=1.0,
        trunc=0.0,
        flux_untruncated=False,
        gsparams=None,
    ):
        super(DeVaucouleurs, self).__init__(
            n=4,
            half_light_radius=half_light_radius,
            scale_radius=scale_radius,
            flux=flux,
            trunc=trunc,
            flux_untruncated=flux_untruncated,
            gsparams=gsparams,
        )

    def __repr__(self):
        return (
            "galsim.DeVaucouleurs(scale_radius=%r, trunc=%r, flux=%r, gsparams=%r)"
            % (
                ensure_hashable(self.scale_radius),
                ensure_hashable(self.trunc),
                ensure_hashable(self.flux),
                self.gsparams,
            )
        )

    def __str__(self):
        s = "galsim.DeVaucouleurs(half_light_radius=%s" % ensure_hashable(
            self.half_light_radius
        )
        if self.trunc != 0.0:
            s += ", trunc=%s" % ensure_hashable(self.trunc)
        if self.flux != 1.0:
            s += ", flux=%s" % ensure_hashable(self.flux)
        s += ")"
        return s

    @implements(_galsim.DeVaucouleurs.withFlux)
    def withFlux(self, flux):
        return DeVaucouleurs(
            scale_radius=self.scale_radius,
            trunc=self.trunc,
            flux=flux,
            gsparams=self.gsparams,
        )
