"""this module defines saturation transformations for media response modeling,
including logistic and Hill functions, as well as an identity transformation for no saturation.
Each transformation can handle both numeric and symbolic parameters,
making them suitable for use in PyMC models with PyTensor.
The abstract base class `Saturation"""

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import pytensor.tensor as pt

from .utils import _to_numpy_1d, _is_symbolic, ParamLike, ArrayLike, _as_scalar_tensor

SaturationType = Literal["None", "Logistic", "Hill"]


# ------------------------------------------------------------------
# Abstract saturation class
# ------------------------------------------------------------------


@dataclass
class Saturation(ABC):
    """Abstract base class for saturation transformations.

    Parameters
    ----------
    mandatory_params : list[str], optional
        Parameters that must be provided when calling the transform.
    """

    mandatory_params: list[str] = field(default_factory=list)

    def _check_params(self, params: dict[str, ParamLike]) -> None:
        """Ensure that all mandatory parameters are present.

        Parameters
        ----------
        params : dict[str, ParamLike]
            Parameters provided to the saturation call.

        Raises
        ------
        ValueError
            If a mandatory parameter is missing.
        """
        for param in self.mandatory_params:
            if param not in params:
                raise ValueError(f"Missing mandatory parameter: {param}")

    @abstractmethod
    def __call__(
        self, x: ArrayLike, params: dict[str, ParamLike], **kwargs
    ) -> np.ndarray | pt.TensorVariable:
        """Apply the saturation transformation.

        Parameters
        ----------
        x : ArrayLike
            Input media signal.
        params : dict[str, ParamLike]
            Saturation parameters (numeric or symbolic).

        Returns
        -------
        np.ndarray | pt.TensorVariable
            Saturated signal.
        """
        raise NotImplementedError

    @classmethod
    def from_spec(cls, kind: SaturationType) -> "Saturation":
        """Create a concrete saturation implementation from a type string.

        Parameters
        ----------
        kind : SaturationType
            Saturation type identifier. One of ``"none"``, ``"logistic"``,
            or ``"hill"``.

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
            return LogisticSaturation(mandatory_params=["lam"])
        if kind == "Hill":
            return HillSaturation(mandatory_params=["slope", "half_max"])
        raise ValueError(
            f"Unknown saturation kind: {kind},  available options are: {SaturationType.__args__}"
        )


# ------------------------------------------------------------------
# Identity (no saturation)
# ------------------------------------------------------------------


class IdentitySaturation(Saturation):
    """Pass-through saturation that leaves the signal unchanged."""

    def __call__(
        self, x: ArrayLike, params: dict[str, ParamLike], **kwargs
    ) -> np.ndarray | pt.TensorVariable:
        """Return the input unchanged.

        Parameters
        ----------
        x : ArrayLike
            Input media signal.
        params : dict[str, ParamLike]
            Ignored.

        Returns
        -------
        np.ndarray
            Input converted to a NumPy array.
        """
        return _to_numpy_1d(x)


# ------------------------------------------------------------------
# Logistic saturation
# ------------------------------------------------------------------


class LogisticSaturation(Saturation):
    """Logistic saturation with a single rate parameter ``lam``."""

    def __call__(
        self, x: ArrayLike, params: dict[str, ParamLike], **kwargs
    ) -> np.ndarray | pt.TensorVariable:
        """Apply logistic saturation: ``(1 - exp(-lam*x)) / (1 + exp(-lam*x))``.

        Parameters
        ----------
        x : ArrayLike
            Input media signal.
        params : dict[str, ParamLike]
            Must include ``lam`` (numeric or symbolic).

        Returns
        -------
        np.ndarray | pt.TensorVariable
            Saturated signal in (-1, 1).
        """
        self._check_params(params)
        lam = params["lam"]
        if _is_symbolic(lam):
            lam = _as_scalar_tensor(lam)
        else:
            lam = float(np.ravel(_to_numpy_1d(lam))[0])

        return (1 - np.exp(-lam * x)) / (1 + np.exp(-lam * x))


# ------------------------------------------------------------------
# Hill saturation
# ------------------------------------------------------------------


class HillSaturation(Saturation):
    """Hill saturation with slope and half-max parameters."""

    def __call__(
        self, x: ArrayLike, params: dict[str, ParamLike], **kwargs
    ) -> np.ndarray | pt.TensorVariable:
        """Apply Hill saturation: ``x^slope / (half_max^slope + x^slope)``.

        Parameters
        ----------
        x : ArrayLike
            Input media signal. Values should be non-negative.
        params : dict[str, ParamLike]
            Must include ``slope`` and ``half_max`` (numeric or symbolic).

        Returns
        -------
        np.ndarray | pt.TensorVariable
            Saturated signal in [0, 1).
        """
        self._check_params(params)
        slope = params["slope"]
        half_max = params["half_max"]

        if _is_symbolic(slope) or _is_symbolic(half_max):
            slope_s = _as_scalar_tensor(slope)
            half_max_s = _as_scalar_tensor(half_max)
            x_t = pt.as_tensor_variable(_to_numpy_1d(x))
            x_pow = pt.power(x_t, slope_s)
            return x_pow / (pt.power(half_max_s, slope_s) + x_pow)

        slope_f = float(np.ravel(_to_numpy_1d(slope))[0])
        half_max_f = float(np.ravel(_to_numpy_1d(half_max))[0])
        x_arr = _to_numpy_1d(x)
        x_pow = np.power(x_arr, slope_f)
        return x_pow / (np.power(half_max_f, slope_f) + x_pow)
