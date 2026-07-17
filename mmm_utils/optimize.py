"""Budget optimization helpers built on top of :class:`~mmm_utils.optimizer.Optimizer`."""

import numpy as np
import pandas as pd

from .optimizer import Optimizer
from .modeling.mmm import MMM


def get_current_budget(mmm: MMM, media: list[str]) -> dict[str, float]:
    """Compute the average historical spend per media channel.

    Parameters
    ----------
    mmm : MMM
        Fitted MMM instance exposing ``data.X_media`` and ``data.scale``.
    media : list[str]
        Media channel names to summarize.

    Returns
    -------
    dict[str, float]
        Average spend per channel, expressed in the original (unscaled) units.
    """
    media_scales = mmm.data.scale("media")
    idx = [mmm.config.media_names.index(m) for m in media]
    mean_spend = mmm.data.X_media[:, idx].mean(axis=0)

    return {m: float(mean_spend[i] * media_scales[m]) for i, m in enumerate(media)}


def get_flexibility(
    media: list[str], flexibility: dict[str, float] | float = 0.5
) -> dict[str, float]:
    """Normalize a flexibility input into a per-channel dictionary.

    Parameters
    ----------
    media : list[str]
        Media channel names.
    flexibility : dict[str, float] or float, default=0.5
        Either a single flexibility applied to every channel, or a
        channel-to-flexibility mapping, e.g. ``0.2`` allows spend to vary
        by +/-20% around the current budget.

    Returns
    -------
    dict[str, float]
        Flexibility value for each channel in ``media``.

    Raises
    ------
    ValueError
        If ``flexibility`` is neither a float nor a matching dictionary.
    """
    if isinstance(flexibility, dict):
        if set(flexibility) != set(media):
            raise ValueError("Flexibility dictionary keys must match media channels")
        return dict(flexibility)

    if isinstance(flexibility, (int, float)):
        return {m: float(flexibility) for m in media}

    raise ValueError("Flexibility must be either a float or a dictionary of floats")


def get_budget_bounds(
    current_budget: dict[str, float], flexibility: dict[str, float]
) -> dict[str, tuple[float, float]]:
    """Compute lower/upper spend bounds around the current budget.

    Parameters
    ----------
    current_budget : dict[str, float]
        Current average spend per channel.
    flexibility : dict[str, float]
        Allowed relative deviation per channel, keyed the same as
        ``current_budget`` (see :func:`get_flexibility`).

    Returns
    -------
    dict[str, tuple[float, float]]
        Mapping from channel to its ``(lower, upper)`` spend bounds.
    """
    return {
        m: (max(b * (1 - flexibility[m]), 0.0), b * (1 + flexibility[m]))
        for m, b in current_budget.items()
    }


def get_optimizer(
    mmm: MMM, campaign_period: int, budget_by_media: dict[str, float]
) -> Optimizer:
    """Build and configure an :class:`~mmm_utils.optimizer.Optimizer` for a campaign.

    The campaign starts the week following the last observed date.

    Parameters
    ----------
    mmm : MMM
        Fitted MMM instance.
    campaign_period : int
        Campaign duration in weeks.
    budget_by_media : dict[str, float]
        Reference spend per channel, in original (unscaled) units.

    Returns
    -------
    Optimizer
        Optimizer configured with the campaign inputs.
    """
    media_scales = mmm.data.scale("media")
    starting_date = pd.Timestamp(np.max(mmm.data.date)) + pd.Timedelta(weeks=1)

    optimizer = Optimizer(mmm)
    optimizer.set_campaign(
        starting_date=starting_date,
        campaign_period=campaign_period,
        budget_by_media={m: b / media_scales[m] for m, b in budget_by_media.items()},
    )
    return optimizer


def print_optimization_results(
    budget_bounds: dict[str, tuple[float, float]], optimized_budget: dict[str, float]
) -> None:
    """Print each channel's optimized spend against its allowed bounds.

    Parameters
    ----------
    budget_bounds : dict[str, tuple[float, float]]
        Mapping from channel to its ``(lower, upper)`` spend bounds.
    optimized_budget : dict[str, float]
        Optimized spend per channel.
    """
    for channel, budget in optimized_budget.items():
        lower, upper = budget_bounds[channel]
        filled = np.isclose(budget, upper)
        print(
            f"{channel}: {lower:,.2f} \t<= \t{budget:,.2f} \t<= \t{upper:,.2f}"
            + ("  \t(full)" if filled else "")
        )


def get_recommended_budget(  # pylint: disable=too-many-arguments
    mmm: MMM,
    media: list[str],
    campaign_period: int,
    flexibility: dict[str, float] | float = 0.5,
    *,
    constant_budget: bool = True,
    verbatim: bool = False,
) -> dict[str, float]:
    """Optimize the budget allocation over a future campaign horizon.

    Parameters
    ----------
    mmm : MMM
        Fitted MMM instance.
    media : list[str]
        Media channels to include in the optimization.
    campaign_period : int
        Campaign duration in weeks.
    flexibility : dict[str, float] or float, default=0.5
        Allowed relative deviation around the current average spend, either
        globally or per channel.
    constant_budget : bool, default=True
        If True, optimize a single allocation per channel held constant over
        the campaign. If False, optimize one allocation per channel and week.
    verbatim : bool, default=False
        If True, print the optimized allocation against its bounds.

    Returns
    -------
    dict[str, float]
        Recommended spend per channel, in original units. For a
        time-varying campaign (``constant_budget=False``), this is the
        first week's allocation.

    Raises
    ------
    RuntimeError
        If the numerical optimization does not converge successfully.
    """
    media_scales = mmm.data.scale("media")

    current_budget = get_current_budget(mmm, media)
    flexibility = get_flexibility(media, flexibility)
    budget_bounds = get_budget_bounds(current_budget, flexibility)

    optimizer = get_optimizer(mmm, campaign_period, current_budget)

    budget_bounds_scaled = [
        (
            budget_bounds[m][0] / media_scales[m],
            budget_bounds[m][1] / media_scales[m],
        )
        for m in media
    ]
    budget_total = (
        sum(current_budget[m] / media_scales[m] for m in media) * campaign_period
    )

    optimized_budget, res_scipy = optimizer.optimize(
        budget_bounds_scaled,
        budget_total,
        constant_budget=constant_budget,
    )

    if not res_scipy.success:
        raise RuntimeError(f"Optimization failed: {res_scipy.message}")

    recommended_budget = {
        m: optimized_budget[:, i] * media_scales[m] for i, m in enumerate(media)
    }

    if verbatim:
        print(f"Recommended budget allocation using flexibility of {flexibility}:")
        print_optimization_results(budget_bounds, recommended_budget)

    return recommended_budget
