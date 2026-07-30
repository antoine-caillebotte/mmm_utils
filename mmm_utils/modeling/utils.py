"""Utility functions for modeling in MMM."""

import numpy as np
from pytensor.graph.basic import Variable
from pytensor.xtensor.type import XTensorConstant, XTensorType
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
