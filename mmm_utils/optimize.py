"""This module provides utilities for optimizing budget allocation"""

import pandas as pd

from mmm_utils.optimizer.optimizer import Optimizer
from mmm_utils.data_logger import data_logger


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
    channels = data[media].columns

    if isinstance(flexibility, dict):
        assert set(flexibility.keys()) == set(
            channels
        ), "Flexibility dictionary keys must match media channels"
        assert all(
            isinstance(v, float) for v in flexibility.values()
        ), "Each media channel's flexibility must be a float"
        return flexibility

    if isinstance(flexibility, float):
        return {channel: flexibility for channel in channels}

    raise ValueError("Flexibility must be either a float or a dictionary of floats")


def get_current_budget(mmm):
    """
    Parameters
    ----------
    mmm : Any
        Fitted MMM object exposing ``X`` and optimizer-compatible APIs.

    Returns
    -------
    dict[str, float]
        Current average budget allocation per media channel.
    """

    current_budget = mmm.data.X_media * mmm.data.scale("media")
    return current_budget.mean().to_dict()


def print_optimization_results(budget, current_budget):
    """Print optimization results.

    Parameters
    ----------
        budget : dict[str, float]
        Optimized budget allocation per media channel.
        current_budget : dict[str, float]
        Current average budget allocation per media channel.
    """
    x = pd.DataFrame(
        {
            "media": current_budget.keys(),
            "current_budget": current_budget.values(),
            "plan": budget.values(),
        },
    )

    x["reco"] = x["plan"] / sum(x["current_budget"]) * 100
    x["change"] = x["plan"] / x["current_budget"] * 100 - 100

    data_logger.direct_to_csv("budget_recommendations.csv", dataframe=x)
    print(x.round({"current_budget": 1, "plan": 1, "reco": 1}))


def get_recommended_budget(
    mmm,
    media: list[str],
    campaign_period: int,
    flexibility: dict[str, float] | float = 0.5,
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

    Returns
    -------
    numpy.ndarray
        Optimized budget allocation with shape ``(campaign_period, n_media)``.

    Raises
    ------
    RuntimeError
        If the numerical optimization does not converge successfully.
    """
    current_budget = mmm.data.X_media.mean().to_dict()
    flexibility = get_flexibility(mmm.data.X_media, media, flexibility)

    budget_bounds = [
        (budget * (1 - flexibility[channel]), budget * (1 + flexibility[channel]))
        for channel, budget in current_budget.items()
    ]

    optimizer = Optimizer(mmm)
    optimizer.set_campaign(
        starting_date=mmm.data.date.max() + pd.Timedelta(weeks=1),
        campaign_period=campaign_period,
        budget_by_media=current_budget,
    )

    budget_optimized, res = optimizer.optimize(
        budget_bounds,
        sum(current_budget.values()) * campaign_period,
        constant_budget=True,
    )

    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")

    return budget_optimized
