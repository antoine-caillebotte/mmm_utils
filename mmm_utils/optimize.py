"""This module provides utilities for optimizing budget allocation"""

import numpy as np
import pandas as pd

from pymc_marketing.mmm.budget_optimizer import optimizer_xarray_builder
from pymc_marketing.mmm.multidimensional import MultiDimensionalBudgetOptimizerWrapper

from mmm_utils.timeline import Timeline


def timeline_from_sample(mmm, sample, media, target: str = "y"):
    """Build a ``Timeline`` instance from MMM posterior samples.

    Parameters
    ----------
    mmm : Any
        Fitted MMM object exposing ``get_scales_as_xarray``.
    sample : xarray.Dataset
        Posterior sample containing media channels and target predictions.
    media : list[str]
        Media channel names used to build the prediction dataframe.
    target : str, default="y"
        Name of the target variable in ``sample``.

    Returns
    -------
    Timeline
        Timeline object containing scaled target predictions and metadata.
    """

    pred_data = sample[media].to_dataframe().reset_index()

    target_scale = float(mmm.get_scales_as_xarray()["target_scale"])
    pred_data["y"] = sample[target].mean(dim="sample").values * target_scale

    timeline = Timeline(
        sample,
        pred_data,
        media=media,
        controls=[],
        target=target,
        target_scale=target_scale,
    )
    return timeline


def get_flexibility(
    data, media: list[str], flexibility: dict[str, float] | float = 0.5
) -> dict[str, float]:
    """Normalize flexibility input to a per-channel dictionary.

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataframe containing media spend columns.
    media : list[str]
        Media channel column names.
    flexibility : dict[str, float] | float, default=0.5
        Either a single flexibility value applied to all channels or a
        channel-to-flexibility mapping.

    Returns
    -------
    dict[str, float]
        Flexibility mapping for each media channel.

    Raises
    ------
    ValueError
        If ``flexibility`` is neither a float nor a dictionary.
    AssertionError
        If dictionary keys do not match channels or values are not floats.
    """

    current_budget = data[media].mean().to_dict()

    if isinstance(flexibility, dict):
        assert set(flexibility.keys()) == set(
            current_budget.keys()
        ), "Flexibility dictionary keys must match media channels"

        assert all(
            isinstance(v, float) for v in flexibility.values()
        ), "Each media channel's flexibility must be a float"

        out = flexibility
    elif isinstance(flexibility, float):
        out = {str(channel): flexibility for channel in current_budget.keys()}

    else:
        raise ValueError("Flexibility must be either a float or a dictionary of floats")

    return out


def get_optimizer(mmm, campaign_period: int):
    """Instantiate a budget optimizer wrapper for a given MMM and campaign period.

    Parameters
    ----------
    mmm : Any
        Fitted MMM object exposing optimizer-compatible APIs.
    campaign_period : int
        Campaign duration in weeks.

    Returns
    -------
    MultiDimensionalBudgetOptimizerWrapper
        Optimizer wrapper instance ready for budget optimization.
    """

    start_date = mmm.X["date"].max() + pd.Timedelta(weeks=1)
    end_date = mmm.X["date"].max() + pd.Timedelta(weeks=campaign_period)

    optimizer = MultiDimensionalBudgetOptimizerWrapper(  # type: ignore[reportAbstractUsage]
        mmm, start_date=start_date, end_date=end_date
    )

    return optimizer


def print_optimization_results(optimize_budget, budget_bounds):
    """Print optimized budget allocation results.

    Parameters
    ----------
    optimize_budget : xarray.DataArray
        Optimized budget allocation per media channel.
    budget_bounds : xarray.DataArray
        Lower and upper bounds for each media channel's budget allocation.
    """

    for channel in optimize_budget.channel:
        lower_bound = float(budget_bounds.sel(channel=channel, bound="lower").item())
        budget = float(optimize_budget.sel(channel=channel).item())
        upper_bound = float(budget_bounds.sel(channel=channel, bound="upper").item())
        budget_filled = np.isclose(budget, upper_bound)
        print(
            f"{channel}:{lower_bound:,.2f}  \t<= \t {budget:,.2f}  \t<= \t {upper_bound:,.2f}"
            + ("  \t(full)" if budget_filled else "")
        )


def get_recommended_budget(
    mmm,
    media: list[str],
    campaign_period: int,
    flexibility: dict[str, float] | float = 0.5,
    verbatim: bool = False,
):
    """Optimize budget allocation over a campaign horizon.

    Parameters
    ----------
    mmm : Any
        Fitted MMM object exposing ``X`` and optimizer-compatible APIs.
    media : list[str]
        Media channels to include in optimization.
    campaign_period : int
        Campaign duration in weeks.
    flexibility : dict[str, float] | float, default=0.5
        Allowed deviation around current average spend, either globally or
        per channel.
    verbatim : bool, default=False
        If ``True``, print optimization bounds and selected allocations.

    Returns
    -------
    xarray.DataArray
        Optimized budget allocation per media channel.

    Raises
    ------
    RuntimeError
        If the numerical optimization does not converge successfully.
    """

    data = mmm.X
    current_budget = data[media].mean().to_dict()
    flexibility = get_flexibility(data, media, flexibility)

    optimizer = get_optimizer(mmm, campaign_period)

    def lower_upper_bound(b, f) -> np.ndarray:  # pylint: disable=missing-function-docstring, missing-return-doc
        return np.array([b * (1 - f), b * (1 + f)])

    budget_bounds = optimizer_xarray_builder(
        np.array(
            [
                lower_upper_bound(b, flexibility[idx])
                for idx, b in current_budget.items()
            ]
        ),
        channel=media,
        bound=["lower", "upper"],
    )

    optimize_budget, res_scipy = optimizer.optimize_budget(  # type: ignore[reportUnknownMemberType]
        budget=sum(current_budget.values()),
        budget_bounds=budget_bounds,
    )

    if not res_scipy.success:
        raise RuntimeError(f"Optimization failed: {res_scipy.message}")

    if verbatim:
        print(f"Recommended budget allocation using flexibility of {flexibility}:")
        print_optimization_results(optimize_budget, budget_bounds)

    return optimize_budget
