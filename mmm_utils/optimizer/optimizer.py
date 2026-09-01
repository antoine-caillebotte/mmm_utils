"""This module implements the Optimizer class,
a constrained optimizer for media budget allocation using PyMC and SciPy SLSQP."""

from warnings import warn

from dataclasses import dataclass

import numpy as np
import arviz as az

from scipy.optimize import minimize
import pymc

from pytensor.graph.basic import Variable


from mmm_utils.modeling.mmm import MMM
from .optimisable_campaign import OptimisableCampaign

from .optimizer_utils import (
    define_constraint_function,
    extract_response_distribution,
    function_with_grad,
)


def _utiliy_function(samples) -> Variable:
    """Return the scalar objective used by the optimizer.

    The optimization minimizes this function, so the mean response is negated
    to effectively maximize expected media contribution.
    """
    return -samples.mean()


def _validate_optimized_budget(budget_optimized, budget_total: float, budget_bounds):
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


@dataclass
class Optimizer:
    """Optimize media budget allocation for a fitted MMM.

    This class wraps a copied MMM model and its posterior samples to build a
    differentiable optimization objective, then solves for budget allocations
    under bound and total-budget constraints with SciPy's SLSQP optimizer.

    The optimizer supports two campaign modes:

    - constant budget per media across the whole campaign period;
    - plain: different allocation per media channel and period;
    - sparse: sparse allocation strategy.

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

    model: pymc.Model
    idata: az.InferenceData
    campaign: OptimisableCampaign

    @staticmethod
    def from_mmm(mmm: MMM, campaign: OptimisableCampaign):
        """Create an Optimizer instance from a fitted MMM and campaign configuration.

        Parameters
        ----------
        mmm : MMM
            Fitted MMM object containing the model and posterior samples.
        campaign : OptimisableCampaign
            Campaign configuration specifying budget allocation strategy and period.

        Returns
        -------
        Optimizer
            An instance of the Optimizer class initialized with the MMM model and campaign.

        Raises
        ------
        Warning
            If the campaign period is shorter than the maximum adstock lag, a warning is issued
        """
        all_lmax = [
            spec.adstock_params.get("l_max", 0)
            for spec in mmm.config.media_transforms.values()
        ]

        l_max = np.max(all_lmax)

        if campaign.period < l_max:
            warn(
                f"Campaign period {campaign.period} is shorter than the maximum adstock"
                f" lag {l_max}. This may lead to suboptimal budget allocation."
            )

        optimizer = Optimizer(
            model=mmm.model.copy(),
            idata=mmm.idata,
            campaign=campaign,
        )
        return optimizer

    def create_optimization_variables(self):
        """Build optimization variables and objective from a budget input.

        Returns
        -------
        tuple[pytensor.graph.basic.Variable, pytensor.graph.basic.Variable]
            A tuple ``(optimizable_target, optimizable_budget)`` where:

            - ``optimizable_target`` is the differentiable scalar objective
              (negative mean of ``total_media_contribution``).
            - ``optimizable_budget`` is the flattened optimization variable
              injected into the model in place of ``channel_data``.
        """

        optimizable_budget, optimizable_model = (
            self.campaign.create_optimization_variables(self.model)
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
            [np.arange(len(budget_bounds))] * self.campaign.period, axis=0
        ).flatten()
        return [budget_bounds[idx] for idx in media_idx]

    def optimize(
        self,
        budget_bounds: list[tuple[float, float]],
        budget_total: float | int,
    ) -> tuple[np.ndarray, object]:
        """Run constrained budget optimization using SLSQP.

        Parameters
        ----------
        budget_bounds : list[tuple[float, float]]
            Per-channel lower and upper bounds. Bounds are repeated across
            the first budget dimension to match the flattened optimization
            vector.
        budget_total : float | int
            Total budget to be allocated across all channels and time periods.

        Returns
        -------
        tuple[np.ndarray, scipy.optimize.OptimizeResult]
            Optimized budget matrix with shape ``(campaign_period, n_media)``
            and the raw SciPy optimization result.
        """

        print("=" * 50 + "\n\t Starting Optimization\n" + "=" * 50)

        optimizable_target, optimizable_budget = (
            self.campaign.create_optimization_variables(self.model)
        )

        f = function_with_grad(optimizable_budget, optimizable_target)

        if self.campaign.mode == "constant":
            constraint = define_constraint_function(
                optimizable_budget,
                lambda x: budget_total - self.campaign.period * x.sum(),
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
                f"\t⌛Budget {np.array(xk).sum():.4f}, "
                f"Remaining Budget {float(constraint['fun'](xk)):.2e}, "
                f"Objective {float(obj_val):.2e}, "
            )

            print()

        res = minimize(
            f,
            x0=self.campaign.create_initial_flattened_budget(),
            jac=True,
            method="SLSQP",
            bounds=self.get_bound_for_budget(
                budget_bounds, self.campaign.mode == "constant"
            ),
            constraints=[constraint],
            callback=track_progress,
        )

        budget_optimized = self.campaign.format_budget_optimized(res.x)

        _validate_optimized_budget(budget_optimized, budget_total, budget_bounds)

        return budget_optimized, res
