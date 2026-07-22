"""Utility functions for modeling in MMM."""

import numpy as np
from pytensor.gradient import DisconnectedType
from pytensor.graph.basic import Variable
from pytensor.xtensor.type import XTensorConstant, XTensorType
from pytensor.raise_op import CheckAndRaise
from xarray import DataArray

type ArrayLike = (
    np.ndarray
    | DataArray
    # | list[float]
    # | tuple[float, ...]
    | Variable
    | XTensorConstant[XTensorType]
)
type ParamLike = (
    ArrayLike
    # | TensorVariable
    | Variable
    | XTensorConstant[XTensorType]
)


class ParameterValueError(ValueError):
    """Exception raised when a parameter value is invalid.

    Notes
    -----
    This exception is used by :class:`CheckParameterValue`.
    """


class CheckParameterValue(CheckAndRaise):
    """Implement a parameter-value check in a PyTensor graph.

    Parameters
    ----------
    msg : str, default=""
        Message attached to the raised exception when the check fails.

    Raises
    ------
    ParameterValueError
        Raised when the provided check condition is not ``True``.
    """

    __props__ = ("msg", "exc_type")

    def __init__(self, msg: str = ""):
        super().__init__(ParameterValueError, msg)

    def __str__(self):
        """Return the string representation of the check node.

        Returns
        -------
        str
            String representation containing the configured message.
        """
        return f"Check{{{self.msg}}}"

    def grad(self, inputs, output_grads):
        """Return reverse-mode gradients for this operation.

        Parameters
        ----------
        inputs : list
            Input symbolic variables. The first input is the value input,
            and remaining inputs correspond to check conditions.
        output_grads : list
            Upstream gradients for the outputs.

        Returns
        -------
        list
            Gradient for the first input and disconnected gradients for
            all check-condition inputs.
        """
        return [output_grads[0], *[DisconnectedType()() for _ in inputs[1:]]]

    def pushforward(self, inputs, outputs, tangents):
        """Return forward-mode tangents for this operation.

        Parameters
        ----------
        inputs : list
            Input symbolic variables.
        outputs : list
            Output symbolic variables.
        tangents : list
            Input tangents.

        Returns
        -------
        list
            Forward tangent propagated from the first input.
        """
        return [tangents[0]]


def max_abs_scaler(x: np.ndarray) -> np.ndarray:
    """Scale data by its maximum absolute value.

    Parameters
    ----------
    x : np.ndarray
        Input vector or matrix.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Scaled data and scale factors used per column.
    """
    x = np.asarray(x, dtype=float)
    scale = np.abs(x).max(axis=0)
    if len(scale.shape) == 0:
        scale = np.array([scale])
    scale[scale == 0] = 1.0
    return x / scale, scale
