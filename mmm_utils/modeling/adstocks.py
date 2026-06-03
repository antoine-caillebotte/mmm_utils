"""This module implements adstock transformations for marketing mix modeling,
supporting both geometric and delayed geometric kernels.
It provides a flexible framework for applying adstock effects to media variables,
 with support for symbolic parameters in probabilistic programming contexts.
"""

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Literal


import numpy as np
import pytensor.tensor as pt
import pytensor.xtensor as ptx

from .utils import (
    ParamLike,
    ArrayLike,
    CheckParameterValue,
)

AdstockType = Literal["none", "Geometric", "GeometricDelayed"]


def batched_convolution(
    x,
    w,
    *,
    dim: str,
    kernel_dim: str,
    lags: int | None = None,
) -> np.ndarray | pt.TensorVariable:
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
    np.ndarray | pt.TensorVariable
        Convolved signal with the same length as the input series.

    Raises
    ------
    ValueError
        If ``lags`` is not provided while ``w`` is symbolic.
    """
    x = ptx.as_xtensor(x)
    w = ptx.as_xtensor(w)

    zeros = ptx.as_xtensor(pt.zeros(lags - 1, dtype=x.dtype), dims=(dim,))
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
    alpha_tensor = ptx.as_xtensor(alpha).values
    alpha_checked = _alpha_check_op(
        alpha_tensor, (alpha_tensor >= 0) & (alpha_tensor <= 1).all()
    )
    return ptx.as_xtensor(alpha_checked)


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
    theta_tensor = ptx.as_xtensor(theta).values
    theta_checked = _theta_check_op(
        theta_tensor, (theta_tensor >= 0) & (theta_tensor < l_max).all()
    )
    return ptx.as_xtensor(theta_checked)


# ------------------------------------------------------------------
# Abstract adstock class
# ------------------------------------------------------------------
@dataclass
class Adstock(ABC):
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

    dim: str
    l_max: int = 12
    normalize: bool = False
    mandatory_params: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Initialize lag index and validate lag configuration."""
        if self.l_max <= 0:
            raise ValueError("l_max must be a positive integer.")

        self.kernel_dim = f"{self.dim}_kernel"
        self.lags_xtensor = ptx.as_xtensor(
            pt.arange(self.l_max, dtype="float64"), dims=(self.kernel_dim,)
        )

    def _check_params(self, params: dict[str, ParamLike]):
        """Ensure that all mandatory parameters are present.

        Parameters
        ----------
        params : dict[str, ParamLike]
            Parameters provided to the adstock call.
        """
        for param in self.mandatory_params:
            if param not in params:
                raise ValueError(f"Missing mandatory parameter: {param}")

    @abstractmethod
    def __call__(
        self, x: ArrayLike, params: dict[str, ParamLike], **kwargs
    ) -> np.ndarray | pt.TensorVariable:
        """Apply the adstock transformation.

        Parameters
        ----------
        x : ArrayLike
            Input time series.
        params : dict[str, ParamLike]
            Adstock parameters.

        Returns
        -------
        np.ndarray | pt.TensorVariable
            Transformed series.
        """
        raise NotImplementedError

    @classmethod
    def from_spec(cls, kind, dim, l_max, normalize) -> "Adstock":
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
            )

        if kind == "GeometricDelayed":
            return GeometricDelayedAdstock(
                dim=dim,
                l_max=l_max,
                normalize=normalize,
                mandatory_params=["alpha", "theta"],
            )

        raise ValueError(
            f"Unknown adstock kind: {kind}, available options are: {AdstockType.__args__}"
        )


# ------------------------------------------------------------------
#  Geometric adstock class
# ------------------------------------------------------------------
class GeometricAdstock(Adstock):
    """Geometric adstock with exponentially decaying lag weights."""

    def __call__(
        self, x: ArrayLike, params: dict[str, ParamLike], **kwargs
    ) -> np.ndarray | pt.TensorVariable:
        """Transform a series with geometric adstock.

        Parameters
        ----------
        x : ArrayLike
            Input time series.
        params : dict[str, ParamLike]
            Must include ``alpha``.

        Returns
        -------
        np.ndarray | pt.TensorVariable
            Adstocked series.
        """

        self._check_params(params)

        alpha = _check_alpha(params["alpha"])
        w = ptx.math.power(alpha, self.lags_xtensor)  # pylint: disable=E1121
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

    def __call__(
        self, x: ArrayLike, params: dict[str, ParamLike], **kwargs
    ) -> np.ndarray | pt.TensorVariable:
        """Transform a series with delayed geometric adstock.

        Parameters
        ----------
        x : ArrayLike
            Input time series.
        params : dict[str, ParamLike]
            Must include ``alpha`` and ``theta``.

        Returns
        -------
        np.ndarray | pt.TensorVariable
            Adstocked series.
        """

        self._check_params(params)

        alpha = _check_alpha(params["alpha"])
        theta = _check_theta(params["theta"], self.l_max)
        x = ptx.as_xtensor(x)

        w = ptx.math.power(alpha, (self.lags_xtensor - theta) ** 2)  # pylint: disable=E1121
        if self.normalize:
            w = w / w.sum(dim=self.kernel_dim)

        return batched_convolution(
            x,
            w,
            dim=self.dim,
            kernel_dim=self.kernel_dim,
            lags=self.l_max,
        )
