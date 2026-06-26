"""This module implements adstock transformations for marketing mix modeling,
supporting both geometric and delayed geometric kernels.
It provides a flexible framework for applying adstock effects to media variables,
 with support for symbolic parameters in probabilistic programming contexts.
"""

from dataclasses import dataclass
from typing import Literal

import pytensor.xtensor.math as ptxmath
from pytensor.xtensor.type import as_xtensor

import pytensor.tensor as pt
import pytensor.xtensor as ptx


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
# false positive : ptx.signal.convolve1d
# pyright: reportPrivateImportUsage=false
# pyright: reportReturnType=false
# disable false positive when
# pyright: reportAttributeAccessIssue=false

# pyright: reportOptionalOperand=false


AdstockType = Literal["none", "Geometric", "GeometricDelayed"]


def batched_convolution(
    x,
    w,
    *,
    dim: str,
    kernel_dim: str,
    lags: int | None = None,
) -> ArrayLike:
    """Apply a 1D convolution with support for symbolic inputs and kernels.

    Parameters
    ----------
    x : ArrayLike | pt.TensorVariable
        Input time series.  Can be numeric or a symbolic pytensor tensor.
    w : ArrayLike | pt.TensorVariable
        Convolution kernel. Can be numeric or symbolic.
    dim : str
        Kept for backward API compatibility.
    kernel_dim : str
        Kept for backward API compatibility.
    lags : int | None, optional
        Kernel size. Required when ``w`` is symbolic.

    Returns
    -------
    ArrayLike
        Convolved signal with the same length as the input series.

    Raises
    ------
    ValueError
        If ``lags`` is not provided while ``w`` is symbolic.
    """
    x = as_xtensor(x)
    w = as_xtensor(w)

    if x.values.ndim == 1:
        zeros = as_xtensor(pt.zeros(lags - 1, dtype=x.dtype), dims=(dim,))
    else:
        # Batched input: build padding that matches x's full shape,
        # replacing the signal dimension size with (lags - 1).
        dim_axis = x.type.dims.index(dim)
        pad_shape = [
            lags - 1 if i == dim_axis else pt.shape(x.values)[i]
            for i in range(x.values.ndim)
        ]
        zeros = as_xtensor(pt.zeros(pad_shape, dtype=x.dtype), dims=x.type.dims)

    padded_x = ptx.concat([zeros, x], dim=dim)
    return ptx.signal.convolve1d(padded_x, w, dims=(dim, kernel_dim), mode="valid")


_alpha_check_op = CheckParameterValue(msg="0 < alpha <= 1")


def _check_alpha(alpha: ParamLike) -> ParamLike:
    """Validate the adstock alpha domain for numeric values.

    Parameters
    ----------
    alpha : ParamLike
        Numeric or symbolic adstock coefficient.

    Returns
    -------
    ParamLike
        Original symbolic input or validated numeric array.

    Raises
    ------
    ValueError
        If a numeric alpha is outside [0, 1].
    """
    alpha_xt = as_xtensor(alpha)
    alpha_checked = _alpha_check_op(
        alpha_xt.values, ((alpha_xt.values >= 0) & (alpha_xt.values <= 1)).all()
    )
    return as_xtensor(alpha_checked, dims=alpha_xt.type.dims)


_theta_check_op = CheckParameterValue(msg="0 <= theta < l_max")


def _check_theta(theta: ParamLike, l_max: int) -> ParamLike:
    """Validate the adstock theta domain for numeric values.

    Parameters
    ----------
    theta : ParamLike
        Numeric or symbolic adstock delay parameter.
    l_max : int
        Maximum lag, used to validate numeric theta.
    Returns
    -------
    ParamLike
        Original symbolic input or validated numeric array.
    Raises
    ------
    ValueError
        If a numeric theta is outside [0, l_max).
    """
    theta_xt = as_xtensor(theta)
    theta_checked = _theta_check_op(
        theta_xt.values, ((theta_xt.values >= 0) & (theta_xt.values < l_max)).all()
    )
    return as_xtensor(theta_checked, dims=theta_xt.type.dims)


# ------------------------------------------------------------------
# Abstract adstock class
# ------------------------------------------------------------------
@dataclass
class Adstock(Transform):
    """Abstract base class for adstock transformations.

    Parameters
    ----------
    dim : str
        Name of the time dimension.
    l_max : int, optional
        Maximum number of lags.
    normalize : bool, optional
        Whether to normalize kernel weights to sum to one.
    mandatory_params : list[str], optional
        Parameters that must be provided when calling the transform.
    """

    dim: str = "to_define"
    l_max: int = 12
    normalize: bool = False

    def __post_init__(self):
        """Initialize lag index and validate lag configuration."""
        if self.l_max <= 0:
            raise ValueError("l_max must be a positive integer.")

        self.kernel_dim = f"{self.dim}_kernel"
        self.lags_xtensor = as_xtensor(
            pt.arange(self.l_max, dtype="float64"), dims=(self.kernel_dim,)
        )

    @classmethod
    def from_spec(  # pylint: disable=too-many-arguments, too-many-positional-arguments
        cls, kind, dim, l_max, normalize, params: dict[str, ParamLike]
    ) -> "Adstock":
        """Create a concrete adstock implementation from a specification.

        Parameters
        ----------
        kind : str
            Adstock type identifier.
        dim : str
            Time dimension name.
        l_max : int
            Maximum lag.
        normalize : bool
            Whether to normalize kernel weights.
        params : dict[str, ParamLike]
            Parameters to initialize the adstock transformer.


        Returns
        -------
        Adstock
            Instantiated adstock transformer.

        Raises
        ------
        ValueError
            If an unknown adstock kind is specified.
        """
        if kind == "Geometric":
            return GeometricAdstock(
                dim=dim,
                l_max=l_max,
                normalize=normalize,
                mandatory_params=["alpha"],
                params=params,
            )

        if kind == "GeometricDelayed":
            return GeometricDelayedAdstock(
                dim=dim,
                l_max=l_max,
                normalize=normalize,
                mandatory_params=["alpha", "theta"],
                params=params,
            )

        raise ValueError(
            f"Unknown adstock kind: {kind}, available options are: {AdstockType.__args__}"
        )


# ------------------------------------------------------------------
#  Geometric adstock class
# ------------------------------------------------------------------
class GeometricAdstock(Adstock):
    """Geometric adstock with exponentially decaying lag weights."""

    @validate_params
    def __call__(self, x: ArrayLike, **kwargs) -> ArrayLike:
        """Transform a series with geometric adstock.

        Parameters
        ----------
        x : ArrayLike
            Input time series.

        Returns
        -------
        ArrayLike
            Adstocked series.
        """
        alpha = _check_alpha(self.params["alpha"])
        w = ptxmath.power(alpha, self.lags_xtensor)  # pylint: disable=E1121
        if self.normalize:
            w = w / w.sum(dim=self.kernel_dim)

        return batched_convolution(
            x,
            w,
            dim=self.dim,
            kernel_dim=self.kernel_dim,
            lags=self.l_max,
        )


# ------------------------------------------------------------------
# Delayed adstock class
# ------------------------------------------------------------------
class GeometricDelayedAdstock(Adstock):
    """Delayed geometric adstock with bell-shaped lag emphasis."""

    @validate_params
    def __call__(self, x: ArrayLike, **kwargs) -> ArrayLike:
        """Transform a series with delayed geometric adstock.

        Parameters
        ----------
        x : ArrayLike
            Input time series.

        Returns
        -------
        ArrayLike
            Adstocked series.
        """
        alpha = _check_alpha(self.params["alpha"])
        theta = _check_theta(self.params["theta"], self.l_max)
        x = as_xtensor(x)

        w = ptxmath.power(alpha, (self.lags_xtensor - theta) ** 2)  # pylint: disable=E1121
        if self.normalize:
            w = w / w.sum(dim=self.kernel_dim)

        return batched_convolution(
            x,
            w,
            dim=self.dim,
            kernel_dim=self.kernel_dim,
            lags=self.l_max,
        )
