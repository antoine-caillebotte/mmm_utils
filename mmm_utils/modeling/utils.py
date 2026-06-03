"""Utility functions for modeling in MMM."""

import numpy as np
import pytensor.tensor as pt
from pytensor.graph.basic import Variable
from pytensor.raise_op import CheckAndRaise
from xarray import DataArray

type ArrayLike = np.ndarray | DataArray | list[float] | tuple[float, ...]
type ParamLike = ArrayLike | pt.TensorVariable | float


class ParameterValueError(ValueError):
    """Exception for invalid parameters values"""


class CheckParameterValue(CheckAndRaise):
    """Implements a parameter value check in graph.



    Raises `ParameterValueError` if the check is not True.
    """

    __props__ = ("msg", "exc_type")

    def __init__(self, msg: str = ""):
        super().__init__(ParameterValueError, msg)

    def __str__(self):
        """Return a string representation of the object."""
        return f"Check{{{self.msg}}}"

    def R_op(
        self, inputs: list[Variable], eval_points: Variable | list[Variable]
    ) -> list[Variable]:
        """Return the R-operator for the check, which is zero since
        the check does not depend on the inputs.

        Parameters
        ----------
        inputs : list[Variable]
            List of input variables to the check operation.

        eval_points : Variable | list[Variable]
            Points at which to evaluate the R-operator.

        Returns
        -------
        list[Variable]
            R-operator evaluated at the given points.

        """
        return [pt.zeros_like(inputs[0])]


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
    scale = np.abs(x).max(axis=0)
    if len(scale.shape) == 0:
        scale = np.array([scale])
    scale[scale == 0] = 1.0
    return x / scale, scale


def _to_numpy_1d(x: ArrayLike | float) -> np.ndarray:
    """Convert an input container to a flattened NumPy array.

    Parameters
    ----------
    x : ArrayLike | float
        Input value that can be a scalar, list/tuple, NumPy array, or xarray DataArray.

    Returns
    -------
    np.ndarray
        One-dimensional NumPy representation of the input.
    """
    if isinstance(x, DataArray):
        array = np.asarray(x.to_numpy(), dtype=np.float64)
    else:
        array = np.asarray(x, dtype=np.float64)
    return np.ravel(array)


def _is_symbolic(x: object) -> bool:
    """Check whether an object is a PyTensor symbolic variable.

    Parameters
    ----------
    x : object
        Object to inspect.

    Returns
    -------
    bool
        True if the object is symbolic, False otherwise.
    """
    return isinstance(x, Variable)


def _as_scalar_tensor(x: ParamLike):
    """Extract a scalar value from numeric or symbolic inputs.

    Parameters
    ----------
    x : ParamLike
        Numeric or symbolic parameter value.

    Returns
    -------
    pt.TensorVariable | float
        Scalar symbolic tensor for symbolic inputs, Python float otherwise.
    """
    if _is_symbolic(x):
        return pt.as_tensor_variable(x).reshape((-1,))[0]
    return float(_to_numpy_1d(x)[0])
