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
import pytensor.xtensor as ptx

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
    lam_tensor = ptx.as_xtensor(lam)
    lam_checked = _lam_check_op(lam_tensor, lam_tensor > 0)
    return ptx.as_xtensor(lam_checked)


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
        return ptx.as_xtensor(x)


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

        x = ptx.as_xtensor(x)
        lam = ptx.as_xtensor(params["lam"])
        mxlma = -lam * x

        return (1 - mxlma.exp()) / (1 + mxlma.exp())
