"""
Abstract base class for transformations.

The module provides a base class for transformations that can be applied to
media signals. Implementations accept numeric or symbolic parameters so they
can be used directly inside PyMC/PyTensor computation graphs.
"""

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from functools import wraps

from .utils import (
    ParamLike,
    ArrayLike,
)


def validate_params(func):
    """Validate transformation parameters before function execution.

    This decorator merges default parameters stored on the transform instance
    with parameters passed at call time, validates required parameters using
    :meth:`Transform.check_params`, and then calls the wrapped function.

    Parameters
    ----------
    func : callable
        Transformation callable to wrap.

    Returns
    -------
    callable
        Wrapped callable that validates parameters before execution.
    """

    @wraps(func)
    def wrapper(self, x: ArrayLike, **kwargs):
        """Wrapped transformation call with parameter validation.

        Parameters
        ----------
        self : Transform
            Transformation instance.
        x : ArrayLike
            Input media signal.
        **kwargs
            Parameters provided at call time to override or complement
            ``self.params``.

        Returns
        -------
        ArrayLike
            Output of the wrapped transformation.
        """
        params = {**self.params, **kwargs}
        self.check_params(params)
        return func(self, x, **kwargs)

    return wrapper


# —————————————————————————————————————————————————————————————————————————————
# Abstract Transform class
# —————————————————————————————————————————————————————————————————————————————


@dataclass
class Transform(ABC):
    """Abstract base class for transformations.

    Parameters
    ----------
    mandatory_params : list[str], optional
        Parameters that must be provided when calling the transform.
    """

    mandatory_params: list[str] = field(default_factory=list)
    params: dict[str, ParamLike] = field(default_factory=dict)

    def check_params(self, params: dict[str, ParamLike]) -> None:
        """Ensure that all mandatory parameters are present.

        Parameters
        ----------
        params : dict[str, ParamLike]
            Parameters provided to the transform call.

        Raises
        ------
        ValueError
            If a mandatory parameter is missing.
        """
        for param in self.mandatory_params:
            if param not in params:
                raise ValueError(f"Missing mandatory parameter: {param}")

    @validate_params
    @abstractmethod
    def __call__(self, x: ArrayLike, **kwargs) -> ArrayLike:
        """Apply the transformation.

         Parameters
         ----------
         x : ArrayLike
             Input media signal.

         Returns
         -------
        ArrayLike
             Transformed signal.
        """
        raise NotImplementedError
