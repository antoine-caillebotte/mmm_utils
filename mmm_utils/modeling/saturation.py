"""Saturation transformations for media response modeling.

The module provides an identity transformation (no saturation) and a logistic
saturation transformation. Implementations accept numeric or symbolic
parameters so they can be used directly inside PyMC/PyTensor computation graphs.
"""

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import pytensor.tensor as pt
import pytensor.xtensor.math as ptxmath
from pymc.dims.math import as_xtensor

from .utils import (
    ParamLike,
    ArrayLike,
    CheckParameterValue,
)

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
    lam_tensor = as_xtensor(lam)
    lam_checked = _lam_check_op(lam_tensor, lam_tensor > 0)
    return as_xtensor(lam_checked)


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
            Saturation type identifier. One of ``"None"`` or ``"Logistic"``.

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
        np.ndarray | pt.TensorVariable
            Input unchanged, as NumPy array or pytensor tensor.
        """
        return as_xtensor(x)


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

        x = as_xtensor(x)
        lam = as_xtensor(params["lam"])
        exp_lam_x = ptxmath.exp(-lam * x)  # pylint: disable= too-many-function-args

        return (1 - exp_lam_x) / (1 + exp_lam_x)
