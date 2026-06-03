"""This module implements the Optimizer class,
a constrained optimizer for media budget allocation using PyMC and SciPy SLSQP."""

from arviz import InferenceData
import xarray as xr

import numpy as np
from scipy.optimize import minimize

from pytensor.graph.basic import Variable

from .optimizer_utils import (
    replace_variable_by_optimization_variable,
    extract_response_distribution,
    define_constraint_function,
    function_with_grad,
)


def _utiliy_function(samples) -> Variable:
    return -samples.mean()


class Optimizer:
    """Constrained optimizer for media budget allocation.

    This class wraps a model and posterior inference data to build a
    differentiable objective from the response distribution and solve a
    constrained optimization problem with SciPy SLSQP.

    Parameters
    ----------
    model : Any
        Probabilistic model containing a ``channel_data`` variable to be
        replaced by optimization variables.
    idata : arviz.InferenceData
        Posterior samples used to evaluate the response distribution during
        optimization.

    Notes
    -----
    The objective minimizes the negative mean of
    ``total_media_contribution`` (equivalent to maximizing its mean), under
    per-channel bounds and an equality constraint preserving the total budget.
    """

    def __init__(self, model, idata: InferenceData):
        self.model = model.copy()
        self.idata = idata

    def create_optimization_variables(self, budget: xr.DataArray):
        """Build optimization variables and objective from a budget input.

        Parameters
        ----------
        budget : xr.DataArray
            Budget allocation used to replace ``channel_data`` in the copied
            model and define the optimization variable shape.

        Returns
        -------
        tuple[pytensor.graph.basic.Variable, pytensor.graph.basic.Variable]
            A tuple ``(optimizable_target, optimizable_budget)`` where:

            - ``optimizable_target`` is the differentiable scalar objective
              (negative mean of ``total_media_contribution``).
            - ``optimizable_budget`` is the flattened optimization variable
              injected into the model in place of ``channel_data``.
        """
        # 1. Extract the response distribution from the PyMC model and InferenceData
        optimizable_budget, optimizable_model = (
            replace_variable_by_optimization_variable(
                self.model, "channel_data", budget
            )
        )

        # _compile_objective_and_grad
        target_distribution = extract_response_distribution(
            optimizable_model, self.idata, response_variable="total_media_contribution"
        )

        optimizable_target = _utiliy_function(target_distribution)
        return optimizable_target, optimizable_budget

    def get_bound_for_budget(
        self, budget: xr.DataArray, budget_bounds: list[tuple[float, float]]
    ):
        """Expand per-channel bounds to match the flattened budget vector.

        Parameters
        ----------
        budget : xr.DataArray
            Budget allocation with shape ``(time, channel)`` used to infer how
            many times each channel bound must be repeated.
        budget_bounds : list[tuple[float, float]]
            Per-channel ``(lower, upper)`` bounds.

        Returns
        -------
        list[tuple[float, float]]
            Bounds aligned with the flattened optimization vector, where each
            channel bound is repeated across the first budget dimension.
        """

        media_idx = np.stack(
            [np.arange(budget.shape[1])] * budget.shape[0], axis=0
        ).flatten()
        return [budget_bounds[idx] for idx in media_idx]

    def optimize(
        self,
        budget: xr.DataArray,
        budget_bounds: list[tuple[float, float]],
    ):
        """Run constrained budget optimization using SLSQP.

        Parameters
        ----------
        budget : xr.DataArray
            Initial budget allocation with shape ``(time, channel)``.
        budget_bounds : list[tuple[float, float]]
            Per-channel lower and upper bounds. Bounds are repeated across
            the first budget dimension to match the flattened optimization
            vector.

        Returns
        -------
        scipy.optimize.OptimizeResult
            Result object returned by :func:`scipy.optimize.minimize`.
        """

        print("=" * 50 + "\n\t Starting Optimization\n" + "=" * 50)

        optimizable_target, optimizable_budget = self.create_optimization_variables(
            budget
        )

        f = function_with_grad(optimizable_budget, optimizable_target)

        constraint = define_constraint_function(
            optimizable_budget,
            lambda x: x.sum() - budget.sum(),
            constraint_type="eq",
        )

        def track_progress(xk):  # pylint: disable=W0612
            # Evaluate objective and gradient
            obj_val, _ = f(xk)

            # Store iteration info
            iter_info = {
                "x": np.array(xk).sum(),  # Current parameter values
                "fun": float(obj_val),  # Objective function value (scalar)
                # "jac": np.array(grad_val),  # Gradient values
            }

            # Evaluate constraint function
            c_val = constraint["fun"](xk)
            # Evaluate constraint gradient
            # c_jac = constraint["jac"](xk)

            constraint_info = {
                "value": float(c_val),
                # "jac": np.array(c_jac) ,
            }

            print(iter_info | constraint_info)

        res = minimize(
            f,
            x0=np.zeros(shape=(budget.size,), dtype=float),
            jac=True,
            method="SLSQP",
            bounds=self.get_bound_for_budget(budget, budget_bounds),
            constraints=[constraint],
            callback=track_progress,
        )

        return res
