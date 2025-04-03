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


@implements(_galsim.Sersic)
@register_pytree_node_class
class Sersic(GSObject):
    _req_params = {"n": float}
    _opt_params = {"flux": float, "trunc": float, "flux_untruncated": bool}
    _single_params = [{"scale_radius": float, "half_light_radius": float}]

    _is_axisymmetric = True
    _is_analytic_x = True
    _is_analytic_k = True

    _minimum_n = 0.3  # Lower bounds has hard limit at ~0.29
    _maximum_n = 6.2  # Upper bounds is just where we have tested that code works well.

    # The conversion from hlr to scale radius is complicated for Sersic, especially since we
    # allow it to be truncated.  So we do these calculations in the C++-layer constructor.
    def __init__(
        self,
        n,
        half_light_radius=None,
        scale_radius=None,
        flux=1.0,
        trunc=0.0,
        flux_untruncated=False,
        gsparams=None,
    ):
        self._n = n
        self._trunc = trunc
        self._ft_table_fvals = None

        if self._n < Sersic._minimum_n:
            raise _galsim.GalSimRangeError(
                "Requested Sersic index is too small",
                self._n,
                Sersic._minimum_n,
                Sersic._maximum_n,
            )
        if self._n > Sersic._maximum_n:
            raise _galsim.GalSimRangeError(
                "Requested Sersic index is too large",
                self._n,
                Sersic._minimum_n,
                Sersic._maximum_n,
            )

        if self._trunc < 0:
            raise _galsim.GalSimRangeError("Sersic trunc must be > 0", self._trunc, 0.0)

        # Parse the radius options
        if half_light_radius is not None:
            if scale_radius is not None:
                raise _galsim.GalSimIncompatibleValuesError(
                    "Only one of scale_radius or half_light_radius may be specified for Spergel",
                    half_light_radius=half_light_radius,
                    scale_radius=scale_radius,
                )
            self._hlr = float(half_light_radius)
            if self._trunc == 0.0 or flux_untruncated:
                self._flux_fraction = 1.0
                self._b = calculate_b(
                    self._n, 1.0 / self._n, self.gamma2n, self._flux_fraction
                )
                self._r0 = self._hlr / self.calculateHLRFactor()
            else:
                if self._trunc <= jnp.sqrt(2.0) * self._hlr:
                    raise _galsim.GalSimRangeError(
                        "Sersic trunc must be > sqrt(2) * half_light_radius",
                        self._trunc,
                        jnp.sqrt(2.0) * self._hlr,
                    )
                self._r0 = self._sersic_truncated_scale(self._n, self._hlr, self._trunc)

        elif scale_radius is not None:
            self._r0 = float(scale_radius)
        else:
            raise _galsim.GalSimIncompatibleValuesError(
                "Either scale_radius or half_light_radius must be specified for Spergel",
                half_light_radius=half_light_radius,
                scale_radius=scale_radius,
            )

        if self._trunc > 0.0:
            self._flux_fraction = self.calculateIntegratedFlux(self._trunc)
            if flux_untruncated:
                # Then update the flux and hlr with the correct values
                flux *= self._flux_fraction
        else:
            self._flux_fraction = 1.0

        super().__init__(n=self._n, scale_radius=self._r0, flux=flux, gsparams=gsparams)

        # Recalculate the half-light radius with finalized _flux_fraction
        self._hlr = self._r0 * self.calculateHLRFactor()

        # Calculate the parameter b
        self._b = calculate_b(self._n, 1.0 / self._n, self.gamma2n, self._flux_fraction)

        # Initialize the FT lookup table
        self._build_FT()

        # Initialize the stepk values
        R = self._calculate_missing_flux_radius(self.gsparams.folding_threshold)
        if self._flux_fraction < 1.0 and self.trunc_factor < R:
            R = self.trunc_factor
        # Make sure it is at least 5 hlr
        R = jnp.max(jnp.array([R, self.gsparams.stepk_minimum_hlr]))
        self.__stepk = jnp.pi / R

    def calculateIntegratedFlux(self, r):
        """Return the fraction of the total flux enclosed within a given radius, r"""
        return sersic_integrated_flux(self._n, float(r) / self._r0)

    # return _galsim.SersicIntegratedFlux(self._n, float(r)/self._r0)

    def calculateHLRFactor(self):
        """Calculate the half-light-radius in units of the scale radius."""
        return self._b**self._n

    # @lazy_property
    # def _sbp(self):
    #     with convert_cpp_errors():
    #         return _galsim.SBSersic(
    #             self._n, self._r0, self._flux, self._trunc, self.gsparams._gsp
    #         )

    @property
    def n(self):
        """The Sersic parameter n."""
        return self._n

    @property
    def scale_radius(self):
        """The scale radius."""
        return self._r0

    @property
    def trunc(self):
        """The truncation radius (if any)."""
        return self._trunc

    @property
    def trunc_factor(self):
        """The truncation factor."""
        return self._trunc / self._r0

    @property
    def half_light_radius(self):
        """The half-light radius."""
        return self._hlr

    @property
    def gamma2n(self):
        return gamma(2.0 * self._n)

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
        return self.__maxk

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
        _inv_r0_sq = 1.0 / (self._r0 * self._r0)
        return _inv_r0_sq * self._shootnorm

    def _xValue(self, pos):
        rsq = (pos.x**2 + pos.y**2) / (self._r0 * self._r0)
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
        ksq = (kpos.x**2 + kpos.y**2) * self._r0 * self._r0

        jax.lax.cond(
            self._ft_table_fvals is None,
            lambda: self._build_FT(),
            lambda: None,
        )

        _kvalue = jnp.select(
            [ksq < self._ksq_min, ksq >= self._ksq_max],
            [
                1.0 + ksq * (self._kderiv2 + ksq * self._kderiv4),
                (self._highk_a + self._highk_b / jnp.sqrt(ksq)) / ksq,
            ],
            interp1d(0.5 * jnp.log(ksq), self._ft_table_logk, self._ft_table_fvals)
            / ksq,
        )

        return _kvalue

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

    def _calculate_missing_flux_radius(self, missing_flux_frac):
        """
        Find the radius enclosing (1 - missing_flux_frac) of the total flux in a Sersic profile.

        Parameters:
        missing_flux_frac (float): Fraction of flux that is missing (not enclosed)

        Returns:
        float: The radius R that encloses (1 - missing_flux_frac) of the total flux
        """
        missing_flux = missing_flux_frac * self.gamma2n
        z1 = -jnp.log(missing_flux)

        def case_n_half():
            return z1  # Exact for n = 0.5

        def case_general():
            z_init = 4.0 * (self._n + 1.0)
            twonm1 = 2.0 * self._n - 1.0

            z2 = (
                z1
                + twonm1 * jnp.log(z_init)
                + twonm1 / z_init
                + (twonm1 * (2.0 * self._n - 3.0)) / (2.0 * z_init * z_init)
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

            z1_new = jax.lax.cond(z1 < 0.0, lambda: self._b, lambda: z1)

            func = SersicMissingFlux(self._n, missing_flux).__call__
            z_root = bisect_for_root(partial(func), z1_new, z2)
            return z_root

        z = jax.lax.cond(
            self._n == 0.5,
            case_n_half,
            case_general,
        )

        return z**self._n

    def _sersic_truncated_scale(self, n, hlr, trunc):
        """Calculate the truncated scale for the Sersic profile."""
        return hlr * calculate_truncated_scale(n, 1.0 / n, self._b, trunc / hlr)

    def _build_FT(self):
        def compute_gammaN(p, n, trunc_factor, flux_fraction):
            z = trunc_factor ** (1.0 / n)
            return jax.lax.cond(
                (trunc_factor > 0) & (flux_fraction < 1.0),
                lambda: gammainc(p, z) * gamma(p),
                lambda: gamma(p),
            )

        gamma4n = compute_gammaN(
            4.0 * self._n, self._n, self.trunc_factor, self._flux_fraction
        )
        gamma6n = compute_gammaN(
            6.0 * self._n, self._n, self.trunc_factor, self._flux_fraction
        )
        gamma8n = compute_gammaN(
            8.0 * self._n, self._n, self.trunc_factor, self._flux_fraction
        )

        kderiv2 = -gamma4n / (4.0 * self.gamma2n) / self._flux_fraction
        kderiv4 = gamma6n / (64.0 * self.gamma2n) / self._flux_fraction
        kderiv6 = gamma8n / (2304.0 * self.gamma2n) / self._flux_fraction

        kmin = (self.gsparams.kvalue_accuracy / kderiv6) ** (1.0 / 6.0)
        ksq_min = kmin * kmin

        hankel_norm = self._flux_fraction * self._n * self.gamma2n
        dlogk = self.gsparams.table_spacing * jnp.sqrt(
            jnp.sqrt(self.gsparams.kvalue_accuracy / 10.0)
        )

        # NOTE: should we use jax.lax.while_loop here? when the cost of evaluating
        # the function is high, it might be worth it...
        logk = jnp.arange(jnp.log(kmin) - 0.001, jnp.log(500.0), dlogk)
        k = jnp.exp(logk)
        ksq = k**2

        # NOTE: should we use jax.lax.while_loop here? when the cost of evaluating
        # the function is high, it might be worth it...
        logk = jnp.arange(jnp.log(kmin) - 0.001, jnp.log(500.0), dlogk)
        k = jnp.exp(logk)
        ksq = k**2

        f = partial(sersic_radial_function, invn=1.0 / self._n)

        if self.trunc_factor > 0:
            f_vals = hankel_trunc_zero_order(
                f,
                k,
                self.trunc_factor,
                (),
                relerr=self.gsparams.integration_relerr,
                abserr=self.gsparams.integration_abserr * hankel_norm,
            )
        else:
            f_vals = hankel_inf_zero_order(
                f,
                k,
                (),
                relerr=self.gsparams.integration_relerr,
                abserr=self.gsparams.integration_abserr * hankel_norm,
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
        thres = self.gsparams.maxk_threshold
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
        within_tol = resid < self.gsparams.kvalue_accuracy
        has_converged = jnp.any(within_tol)

        # Find the first k value where high-k approx becomes good
        ksq_max_idx = jnp.argmax(within_tol)
        buffer = 5
        ksq_max_idx_buffered = jnp.minimum(ksq_max_idx + buffer, len(k) - 1)

        ksq_max = jnp.where(has_converged, ksq[ksq_max_idx_buffered], ksq[-1])
        maxk = jnp.sqrt(ksq_max)

        # Store values for use elsewhere
        self._kmin = kmin
        self._ksq_min = ksq_min
        self._ksq_max = ksq_max
        self.__maxk = maxk
        self._kderiv2 = kderiv2
        self._kderiv4 = kderiv4
        self._highk_a = a
        self._highk_b = b
        self._found_maxk = found_maxk
        self._approx_k_at_thres = _approx_k_at_thres

        # Build the lookup table
        self._ft_table_logk = logk
        self._ft_table_fvals = f_vals


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
