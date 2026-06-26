"""Saturation transformations for media response modeling.

The module provides an identity transformation (no saturation) and a logistic
saturation transformation. Implementations accept numeric or symbolic
parameters so they can be used directly inside PyMC/PyTensor computation graphs.
"""

from dataclasses import dataclass
from typing import Literal

import pytensor.xtensor.math as ptxmath
from pytensor.xtensor.type import as_xtensor

import pymc.dims as pmd
import pymc as pm


from .utils import (
    ParamLike,
    ArrayLike,
    CheckParameterValue,
)

from .transform import Transform, validate_params

# disable false positive when call ptxmath functions
# pylint: disable=too-many-function-args
# pyright: reportCallIssue=false
# disable false positive when operators are used with symbolic tensors
# pyright: reportOperatorIssue=false
# disable false positive when
# pyright: reportAttributeAccessIssue=false

SaturationType = Literal["None", "Logistic", "Hill"]


_lam_check_op = CheckParameterValue("lam must be positive")


def _check_lam(lam: ParamLike) -> ParamLike:
    """Validate that the logistic saturation parameter ``lam`` is positive.

    Parameters
    ----------
    lam : ParamLike
        Logistic saturation rate parameter, can be numeric or symbolic.

    Returns
    -------
    ParamLike
        The input parameter if valid.

    Raises
    ------
    CheckParameterValue
        If ``lam`` is not positive.
    """
    lam_xt = as_xtensor(lam)
    lam_checked = _lam_check_op(lam_xt.values, (lam_xt.values > 0).all())
    return as_xtensor(lam_checked, dims=lam_xt.type.dims)


# ------------------------------------------------------------------
# Abstract saturation class
# ------------------------------------------------------------------


@dataclass
class Saturation(Transform):
    """Abstract base class for saturation transformations.

    Parameters
    ----------
    mandatory_params : list[str], optional
        Parameters that must be provided when calling the transform.
    """

    @classmethod
    def from_spec(
        cls, kind: SaturationType, params: dict[str, ParamLike]
    ) -> "Saturation":
        """Create a concrete saturation implementation from a type string.

        Parameters
        ----------
        kind : SaturationType
            Saturation type identifier. One of ``"None"``, ``"Logistic"``, or ``"Hill"``.
        params : dict[str, ParamLike]
            Parameters to initialize the saturation transformer.


        Returns
        -------
        Saturation
            Instantiated saturation transformer.

        Raises
        ------
        ValueError
            If ``kind`` is not recognized.
        """
        if kind == "None":
            return IdentitySaturation()
        if kind == "Logistic":
            return LogisticSaturation(mandatory_params=["lam"], params=params)
        if kind == "Hill":
            return HillSaturation(mandatory_params=["n", "k"], params=params)
        raise ValueError(
            f"Unknown saturation kind: {kind},  available options are: {SaturationType.__args__}"
        )

    def sample_saturation_curve(self, mmm, x_grid):
        """Sample the saturation curve over a grid of input values.

        Parameters
        ----------
        mmm : object
            Fitted MMM model instance providing ``idata`` (InferenceData) and
            ``config.random_seed``.
        x_grid : ArrayLike
            1-D array of input values at which to evaluate the saturation
            transformation.

        Returns
        -------
        xarray.DataArray
            Posterior predictive samples of the saturation curve with
            dimension ``"x"`` corresponding to ``x_grid``.
        """
        x_tensor = as_xtensor(x_grid, dims=("x",))

        with pm.Model(coords={"x": x_grid}):
            y = self(x_tensor, params=self.params)
            pmd.Deterministic(
                name="saturation",
                value=y,
                dims=("x",),
            )

            return pm.sample_posterior_predictive(
                mmm.idata,
                var_names=["saturation"],
                random_seed=mmm.config.random_seed,
                extend_inferencedata=False,
                progressbar=False,
            ).posterior_predictive["saturation"]


# ------------------------------------------------------------------
# Identity (no saturation)
# ------------------------------------------------------------------


class IdentitySaturation(Saturation):
    """Pass-through saturation that leaves the signal unchanged."""

    def __call__(self, x: ArrayLike, **kwargs) -> ArrayLike:
        """Return the input unchanged.

        Parameters
        ----------
        x : ArrayLike
            Input media signal.

        Returns
        -------
        ArrayLike
            Input unchanged, as NumPy array or pytensor tensor.
        """
        return as_xtensor(x)


# ------------------------------------------------------------------
# Logistic saturation
# ------------------------------------------------------------------


class LogisticSaturation(Saturation):
    """Logistic saturation with a single rate parameter ``lam``."""

    @validate_params
    def __call__(self, x: ArrayLike, **kwargs) -> ArrayLike:
        """Apply logistic saturation: ``(1 - exp(-lam*x)) / (1 + exp(-lam*x))``.

        Parameters
        ----------
        x : ArrayLike
            Input media signal.

        Returns
        -------
        ArrayLike
            Saturated signal in (0, 1).
        """

        x = as_xtensor(x)
        lam = _check_lam(self.params["lam"])
        exp_lam_x = ptxmath.exp(-lam * x)  # pylint: disable= too-many-function-args

        return (1 - exp_lam_x) / (1 + exp_lam_x)


# ------------------------------------------------------------------
# Hill saturation
# ------------------------------------------------------------------


class HillSaturation(Saturation):
    """Hill saturation with two parameters: ``n`` (shape) and ``k`` (half-saturation point)."""

    @validate_params
    def __call__(self, x: ArrayLike, **kwargs) -> ArrayLike:
        """Apply Hill saturation: ``(x^n) / (k^n + x^n)``.

        Parameters
        ----------
        x : ArrayLike
            Input media signal.

        Returns
        -------
        ArrayLike
            Saturated signal in (0, 1).
        """

        x = as_xtensor(x)
        n = as_xtensor(self.params["n"])
        k = as_xtensor(self.params["k"])

        x_n = x / (x.max() + 1e-8)
        x_p = ptxmath.pow(x_n, n)  # pylint: disable= too-many-function-args
        k_p = ptxmath.pow(k, n)  # pylint: disable= too-many-function-args
        return x_p / (k_p + x_p)
