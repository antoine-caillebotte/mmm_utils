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

from .utils import _to_numpy_1d, _is_symbolic, ParamLike, ArrayLike, _as_scalar_tensor

AdstockType = Literal["none", "Geometric", "GeometricDelayed"]


def batched_convolution(
    x,
    w,
    *,
    dim: str,
    kernel_dim: str,
    lags: int | None = None,
) -> np.ndarray | pt.TensorVariable:
    """Apply a 1D convolution with support for symbolic kernels.

    Parameters
    ----------
    x : ArrayLike
        Input time series.
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
    del dim, kernel_dim  # Kept for API compatibility with previous signature.

    x_arr = _to_numpy_1d(x)
    if lags is None:
        if _is_symbolic(w):
            raise ValueError("lags must be provided when w is symbolic")
        lags = _to_numpy_1d(w).shape[0]

    padded_x = np.concatenate([np.zeros(lags - 1, dtype=x_arr.dtype), x_arr])
    lagged = np.lib.stride_tricks.sliding_window_view(padded_x, lags)[:, ::-1]

    if _is_symbolic(w):
        return pt.dot(pt.as_tensor_variable(lagged), pt.as_tensor_variable(w))

    w_arr = _to_numpy_1d(w)
    return lagged @ w_arr


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
    if _is_symbolic(alpha):
        return alpha
    alpha_array = _to_numpy_1d(alpha)
    if not np.all((alpha_array >= 0.0) & (alpha_array <= 1.0)):
        raise ValueError("0 < alpha <= 1")
    return alpha_array


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
        self.lags = np.arange(self.l_max, dtype=np.float64)

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
        if _is_symbolic(alpha):
            alpha_scalar = _as_scalar_tensor(alpha)
            w = pt.power(alpha_scalar, self.lags)
            if self.normalize:
                w = w / pt.sum(w)
        else:
            alpha_scalar = float(_to_numpy_1d(alpha)[0])
            w = np.power(alpha_scalar, self.lags)
            if self.normalize:
                w = w / np.sum(w)

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
        theta = params["theta"]

        if _is_symbolic(alpha) or _is_symbolic(theta):
            alpha_scalar = _as_scalar_tensor(alpha)
            theta_scalar = _as_scalar_tensor(theta)
            w = pt.power(alpha_scalar, (self.lags - theta_scalar) ** 2)
            if self.normalize:
                w = w / pt.sum(w)
        else:
            alpha_scalar = float(_to_numpy_1d(alpha)[0])
            theta_scalar = float(_to_numpy_1d(theta)[0])
            w = np.power(alpha_scalar, (self.lags - theta_scalar) ** 2)
            if self.normalize:
                w = w / np.sum(w)

        return batched_convolution(
            x,
            w,
            dim=self.dim,
            kernel_dim=self.kernel_dim,
            lags=self.l_max,
        )
