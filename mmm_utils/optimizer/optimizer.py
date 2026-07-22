"""This module implements the Optimizer class,
a constrained optimizer for media budget allocation using PyMC and SciPy SLSQP."""

from warnings import warn

import xarray as xr
import pandas as pd

import numpy as np
from scipy.optimize import minimize

from pytensor.graph.basic import Variable
import pytensor.tensor as pt
from pytensor.xtensor.type import as_xtensor

from .optimizer_utils import (
    replace_variable_by_optimization_variable,
    replace_variable_by_repeated_optimization_variable,
    extract_response_distribution,
    define_constraint_function,
    function_with_grad,
)


def _utiliy_function(samples) -> Variable:
    """Return the scalar objective used by the optimizer.

    The optimization minimizes this function, so the mean response is negated
    to effectively maximize expected media contribution.
    """
    return -samples.mean()


def _validate_optimized_budget(
    budget_optimized, budget_total: float | int, budget_bounds
):
    """Validate optimized allocations against budget and bound constraints."""
    if budget_optimized.sum() - budget_total < 1e-3:
        print("Budget constraint satisfied.")
    else:
        warn("Budget constraint not satisfied.")

    for i in range(budget_optimized.shape[1]):
        if not (
            budget_bounds[i][0] <= budget_optimized[:, i].min()
            and budget_optimized[:, i].max() <= budget_bounds[i][1]
        ):
            warn(f"Budget bounds not satisfied for the {i}th media.")
    print("Budget bounds satisfied for all media.")


class Optimizer:
    """Optimize media budget allocation for a fitted MMM.

    This class wraps a copied MMM model and its posterior samples to build a
    differentiable optimization objective, then solves for budget allocations
    under bound and total-budget constraints with SciPy's SLSQP optimizer.

    The optimizer supports two campaign modes:

    - constant budget per media across the whole campaign period;
    - time-varying budget per media over the campaign period.

    Workflow
    --------
    1. Initialize with a fitted MMM object.
    2. Configure campaign inputs with :meth:`set_campaign`.
    3. Call :meth:`optimize` with bounds and total budget.

    Parameters
    ----------
    mmm : object
        Fitted MMM-like object expected to expose:

        - ``model``: a PyMC model containing ``channel_data`` and
          ``total_media_contribution``;
        - ``idata``: posterior draws as :class:`arviz.InferenceData`;
        - ``config.media_transforms``: mapping used to derive adstock ``l_max``.

    Attributes
    ----------
    model : object
        Copy of the input MMM model used for optimization graph rewriting.
    idata : arviz.InferenceData
        Posterior samples used to evaluate the response distribution.
    l_max : int
        Maximum adstock lag inferred from media transform configuration.
    """

    def __init__(self, mmm):
        self.model = mmm.model.copy()
        self.idata = mmm.idata

        all_lmax = [
            spec.adstock_params.get("l_max", 0)
            for spec in mmm.config.media_transforms.values()
        ]

        self.l_max = np.max(all_lmax)

        self._campaign_period = None
        self._starting_date = None
        self._budget_by_media = None

    def set_campaign(
        self,
        starting_date: pd.Timestamp | None,
        campaign_period: int | None,
        budget_by_media: dict | None = None,
    ):
        """Set campaign inputs used by the optimization routine.

        Parameters
        ----------
        starting_date : pd.Timestamp | None
            Starting date of the campaign period.
        campaign_period : int | None
            Length of the campaign period in time steps (e.g., weeks).
        budget_by_media : dict | None, optional
            Mapping from media channel name to initial/reference budget value.
        """
        if campaign_period is not None:
            self._campaign_period = campaign_period

        if starting_date is not None:
            self._starting_date = starting_date

        if budget_by_media is not None:
            self._budget_by_media = budget_by_media

    def _check_campaign(self):
        """Ensure all campaign inputs were configured before optimization."""
        if not hasattr(self, "_campaign_period"):
            raise ValueError("Campaign period not set. Please call set_campaign().")
        if not hasattr(self, "_starting_date"):
            raise ValueError("Starting date not set. Please call set_campaign().")
        if not hasattr(self, "_budget_by_media"):
            raise ValueError("Budget by media not set. Please call set_campaign().")

    def create_budget_template(self, constant_budget: bool = True):
        """Create a budget template DataArray for optimization variable injection.

        Parameters
        ----------
        constant_budget : bool, optional
            If True, build a single-date template (constant allocation over time).
            If False, build one row per campaign period step.

        Returns
        -------
        xarray.DataArray
            Budget template with dimensions ``date`` and ``media``.
        """
        if constant_budget:
            index = pd.DatetimeIndex([self._starting_date])
        else:
            index = pd.date_range(
                start=self._starting_date, periods=self._campaign_period, freq="W"
            )

        budget = pd.DataFrame(
            self._budget_by_media,
            index=index,
        )

        budget = xr.DataArray(
            budget.values,
            coords={"media": list(budget.columns), "date": budget.index},
            dims=["date", "media"],
        )

        print(f"✅ Budget template created :\n\t{budget}")

        return budget

    def create_optimization_variables(self, constant_budget: bool = True):
        """Build optimization variables and objective from a budget input.

        Parameters
        ----------
        constant_budget : bool, optional
            If True, optimize a single allocation per media channel and repeat
            it over the campaign horizon. If False, optimize one allocation per
            media channel and period.

        Returns
        -------
        tuple[pytensor.graph.basic.Variable, pytensor.graph.basic.Variable]
            A tuple ``(optimizable_target, optimizable_budget)`` where:

            - ``optimizable_target`` is the differentiable scalar objective
              (negative mean of ``total_media_contribution``).
            - ``optimizable_budget`` is the flattened optimization variable
              injected into the model in place of ``channel_data``.
        """

        # 0. Create budget template
        budget = self.create_budget_template(constant_budget=constant_budget)

        campaign_index = pd.date_range(
            start=self._starting_date,
            periods=self._campaign_period,
            freq="W",
        )

        # Keep control_data aligned with the campaign horizon.
        # This is required when product-media interactions depend on controls.
        control_data = np.asarray(self.model["control_data"].eval(), dtype=float)
        n_controls = control_data.shape[1]
        control_names = list(self.model.coords.get("control", range(n_controls)))

        if constant_budget:
            control_values = np.repeat(
                control_data[[-1], :], self._campaign_period, axis=0
            )
        else:
            if control_data.shape[0] >= self._campaign_period:
                control_values = control_data[-self._campaign_period :, :]
            else:
                control_values = np.repeat(
                    control_data[[-1], :], self._campaign_period, axis=0
                )

        control_template = xr.DataArray(
            control_values,
            coords={"date": campaign_index, "control": control_names},
            dims=["date", "control"],
        )

        control_xtensor = as_xtensor(
            pt.as_tensor_variable(control_template.values),
            dims=control_template.dims,
            name="control_data",
        )

        extra_replacements = {"control_data": control_xtensor}

        if constant_budget:
            optimizable_budget, optimizable_model = (
                replace_variable_by_repeated_optimization_variable(
                    self.model,
                    "channel_data",
                    budget,
                    n_repeat=self._campaign_period,
                    extra_replacements=extra_replacements,
                )
            )
        else:
            optimizable_budget, optimizable_model = (
                replace_variable_by_optimization_variable(
                    self.model,
                    "channel_data",
                    budget,
                    extra_replacements=extra_replacements,
                )
            )

        # _compile_objective_and_grad
        target_distribution = extract_response_distribution(
            optimizable_model,
            self.idata.posterior,
            response_variable="total_media_contribution",
        )

        optimizable_target = _utiliy_function(target_distribution)
        return optimizable_target, optimizable_budget

    def get_bound_for_budget(
        self, budget_bounds: list[tuple[float, float]], constant_budget: bool = True
    ):
        """Expand per-channel bounds to match the flattened budget vector.

        Parameters
        ----------
        budget_bounds : list[tuple[float, float]]
            Per-channel ``(lower, upper)`` bounds.
        constant_budget : bool
            Whether the optimization is performed with a constant budget across
            time (True) or with a different budget for each time step (False).
        Returns
        -------
        list[tuple[float, float]]
            Bounds aligned with the flattened optimization vector, where each
            channel bound is repeated across the first budget dimension.
        """
        if constant_budget:
            return budget_bounds

        media_idx = np.stack(
            [np.arange(len(budget_bounds))] * self._campaign_period, axis=0
        ).flatten()
        return [budget_bounds[idx] for idx in media_idx]

    def optimize(
        self,
        budget_bounds: list[tuple[float, float]],
        budget_total: float | int,
        constant_budget: bool = True,
    ):
        """Run constrained budget optimization using SLSQP.

        Parameters
        ----------
        budget_bounds : list[tuple[float, float]]
            Per-channel lower and upper bounds. Bounds are repeated across
            the first budget dimension to match the flattened optimization
            vector.
        budget_total : float | int
            Total budget to be allocated across all channels and time periods.
        constant_budget : bool
            Whether to optimize with a constant budget across time (True) or
            with a different budget for each time step (False).

        Returns
        -------
        tuple[np.ndarray, scipy.optimize.OptimizeResult]
            Optimized budget matrix with shape ``(campaign_period, n_media)``
            and the raw SciPy optimization result.
        """

        self._check_campaign()

        print("=" * 50 + "\n\t Starting Optimization\n" + "=" * 50)

        optimizable_target, optimizable_budget = self.create_optimization_variables(
            constant_budget=constant_budget
        )

        f = function_with_grad(optimizable_budget, optimizable_target)

        if constant_budget:
            constraint = define_constraint_function(
                optimizable_budget,
                lambda x: budget_total - self._campaign_period * x.sum(),
                constraint_type="eq",
            )
        else:
            constraint = define_constraint_function(
                optimizable_budget,
                lambda x: budget_total - x.sum(),
                constraint_type="eq",
            )

        def track_progress(xk):  # pylint: disable=W0612
            obj_val, _ = f(xk)
            print(
                f"⌛Budget {np.array(xk).sum():.4f}, "
                f"Remaining Budget {float(constraint['fun'](xk)):.2e}, "
                f"Objective {float(obj_val):.2e}, "
            )

        final_budget_shape = (self._campaign_period, len(self._budget_by_media))

        # x0 intiatilization
        if constant_budget:
            x0 = np.zeros(shape=len(self._budget_by_media), dtype=float).flatten()
        else:
            x0 = np.zeros(shape=final_budget_shape, dtype=float).flatten()

        print()
        res = minimize(
            f,
            x0=x0,
            jac=True,
            method="SLSQP",
            bounds=self.get_bound_for_budget(budget_bounds, constant_budget),
            constraints=[constraint],
            callback=track_progress,
        )

        if constant_budget:
            budget_optimized = np.tile(res.x, (self._campaign_period, 1))
        else:
            budget_optimized = res.x.reshape(final_budget_shape)

        _validate_optimized_budget(budget_optimized, budget_total, budget_bounds)

        return budget_optimized, res
