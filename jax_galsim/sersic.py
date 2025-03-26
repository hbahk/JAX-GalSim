import galsim as _galsim
import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.scipy.special import gamma, gammainc
from jax.tree_util import register_pytree_node_class
from jax.tree_util import Partial as partial

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


# TODO: remove scipy dependency!!
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
    if trunc <= jnp.sqrt(2.0):
        raise ValueError(
            "Sersic truncation must be larger than sqrt(2)*half_light_radius."
        )

    x = trunc**invn

    b1 = (jnp.log(0.5) + (2 * n - 1) * jnp.log(x)) / (x - 1)

    if b1 < 1.0e-3 * b:
        b1 = b / 2

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
        self._n = float(n)
        self._flux = float(flux)
        self._trunc = float(trunc)

        self._b = None
        self.__stepk = 0.0
        self.__maxk = 0.0

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
            self._hlr = 0.0
        else:
            raise _galsim.GalSimIncompatibleValuesError(
                "Either scale_radius or half_light_radius must be specified for Spergel",
                half_light_radius=half_light_radius,
                scale_radius=scale_radius,
            )

        super().__init__(scale_radius=self._r0, flux=self._flux, gsparams=gsparams)

        if self._trunc > 0.0:
            self._flux_fraction = self.calculateIntegratedFlux(self._trunc)
            if flux_untruncated:
                # Then update the flux and hlr with the correct values
                self._flux *= self._flux_fraction
                self._hlr = (
                    0.0  # This will be updated by getHalfLightRadius if necessary.
                )
        else:
            self._flux_fraction = 1.0

    def calculateIntegratedFlux(self, r):
        """Return the fraction of the total flux enclosed within a given radius, r"""
        return sersic_integrated_flux(self._n, float(r) / self._r0)

    # return _galsim.SersicIntegratedFlux(self._n, float(r)/self._r0)

    def calculateHLRFactor(self):
        """Calculate the half-light-radius in units of the scale radius."""
        return self.b**self._n

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
        if self._hlr == 0.0:
            self._hlr = self._r0 * self.calculateHLRFactor()
        return self._hlr

    @property
    def gamma2n(self):
        return gamma(2.0 * self._n)

    @property
    def b(self):
        if self._b is None:
            self._b = calculate_b(
                self._n, 1.0 / self._n, self.gamma2n, self._flux_fraction
            )
        return self._b

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
                self.n,
                self.scale_radius,
                self.trunc,
                self.flux,
                self.gsparams,
            )
        )

    def __repr__(self):
        return (
            "galsim.Sersic(n=%r, scale_radius=%r, trunc=%r, flux=%r, gsparams=%r)"
            % (self.n, self.scale_radius, self.trunc, self.flux, self.gsparams)
        )

    def __str__(self):
        # Note: for the repr, we use the scale_radius, since that should just flow as is through
        # the constructor, so it should be exact.  But most people use half_light_radius
        # for Sersics, so use that in the looser str() function.
        s = "galsim.Sersic(n=%s, half_light_radius=%s" % (
            self.n,
            self.half_light_radius,
        )
        if self.trunc != 0.0:
            s += ", trunc=%s" % self.trunc
        if self.flux != 1.0:
            s += ", flux=%s" % self.flux
        s += ")"
        return s

    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop("_sbp", None)
        return d

    def __setstate__(self, d):
        self.__dict__ = d

    @property
    def _maxk(self):
        if self.__maxk == 0.0:
            self.__maxk = self.build_FT()
        return self.__maxk

    @property
    def _stepk(self):
        if self.__stepk == 0.0:
            R = self._calculate_missing_flux_radius(self.gsparams.folding_threshold)
            if self._flux_fraction < 1.0 and self.trunc_factor < R:
                R = self.trunc_factor
            # Make sure it is at least 5 hlr
            R = jnp.max(jnp.array([R, self.gsparams.stepk_minimum_hlr]))
            self.__stepk = jnp.pi / R
        return self.__stepk

    @property
    def _has_hard_edges(self):
        return self._trunc != 0.0

    @property
    def _max_sb(self):
        return self._sbp.maxSB()

    def _xValue(self, pos):
        return self._sbp.xValue(pos._p)

    def _kValue(self, kpos):
        return self._sbp.kValue(kpos._p)

    def _shoot(self, photons, rng):
        self._sbp.shoot(photons._pa, rng._rng)

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

    @jit
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

        if self._n == 0.5:
            z = z1  # Exact formula for n = 0.5
        else:
            z = 4.0 * (self._n + 1.0)  # Initial guess
            twonm1 = 2.0 * self._n - 1.0
            z2 = (
                z1
                + twonm1 * jnp.log(z)
                + twonm1 / z
                + (twonm1 * (2.0 * self._n - 3.0)) / (2.0 * z * z)
            )

            # Ensure gap is not too small
            if z2 > z1 and z2 - z1 < 0.01:
                z2 = z1 + 0.01
            elif z2 < z1 and z2 - z1 > -0.01:
                z2 = z1 - 0.01

            if z1 < 0.0:
                z1 = self.b

            func = SersicMissingFlux(self._n, missing_flux)
            pfunc = partial(func)
            z = bisect_for_root(pfunc, z1, z2)

        return z**self._n

    def _sersic_truncated_scale(self, n, hlr, trunc):
        """Calculate the truncated scale for the Sersic profile."""
        return hlr * calculate_truncated_scale(n, 1.0 / n, self.b, trunc / hlr)

    def _build_FT(self):
        def gammaN(p):
            if self.trunc_factor > 0 and self._flux_fraction < 1.0:
                z = self.trunc_factor ** (1.0 / self._n)
                return gammainc(p, z) * gamma(p)
            else:
                return gamma(p)

        gamma4n = gammaN(4.0 * self._n)
        gamma6n = gammaN(6.0 * self._n)
        gamma8n = gammaN(8.0 * self._n)

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

        f = sersic_radial_function

        f_vals = jax.lax.cond(
            self.trunc_factor > 0,
            lambda x: hankel_trunc_zero_order(
                f,
                x,
                self.trunc_factor,
                1.0 / self._n,
                relerr=self.gsparams.integration_relerr,
                abserr=self.gsparams.integration_abserr * hankel_norm,
            ),
            lambda x: hankel_inf_zero_order(
                f,
                x,
                1.0 / self._n,
                relerr=self.gsparams.integration_relerr,
                abserr=self.gsparams.integration_abserr * hankel_norm,
            ),
            k,
        )

        f_vals /= hankel_norm
        f0_vals = f_vals * ksq

        # TODO: check this!!
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
        self.kmin = kmin
        self.ksq_min = ksq_min
        self.ksq_max = ksq_max
        self._maxk = maxk
        self.kderiv2 = kderiv2
        self.kderiv4 = kderiv4
        self.highk_a = a
        self.highk_b = b
        self._found_maxk = found_maxk
        self._approx_k_at_thres = _approx_k_at_thres


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
            % (self.scale_radius, self.trunc, self.flux, self.gsparams)
        )

    def __str__(self):
        s = "galsim.DeVaucouleurs(half_light_radius=%s" % self.half_light_radius
        if self.trunc != 0.0:
            s += ", trunc=%s" % self.trunc
        if self.flux != 1.0:
            s += ", flux=%s" % self.flux
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
