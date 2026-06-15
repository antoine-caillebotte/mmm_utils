"""Plotting utilities for media mix modeling."""

from collections.abc import Iterable

import arviz as az
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from .timeline import Timeline

tab20colors = plt.get_cmap("tab20").colors


def plot_posterior_predictive_y(mmm, plot_seasonality: bool = True):
    """Plot posterior predictive distribution of ``y`` with observed data.

    Parameters
    ----------
    mmm : MediaMixModel
        Fitted media mix model containing inference data with posterior predictive samples.

    plot_seasonality : bool, default=True
        If ``True``, overlay the posterior seasonality and intercept contributions.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib figure containing the plot.
    ax : matplotlib.axes.Axes
        Axes containing the posterior predictive distribution and observed data.
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    target_scale = mmm.scale("y")
    posterior_predictive_y = mmm.idata.posterior_predictive.y * target_scale
    posterior_season = (
        mmm.idata.posterior.yearly_seasonality_contribution
        + mmm.idata.posterior.intercept_contribution
    ) * target_scale

    date = mmm.idata.posterior_predictive.date

    for i, hdi_prob in enumerate([0.94, 0.5]):
        az.plot_hdi(
            x=date,
            y=posterior_predictive_y.unstack().transpose(..., "date"),
            smooth=False,
            color="C0",
            hdi_prob=hdi_prob,
            fill_kwargs={"alpha": 0.3 + i * 0.1, "label": f"{hdi_prob:.0%} HDI"},
            ax=ax,
        )

    _ = sns.lineplot(
        x=date,
        y=posterior_predictive_y.mean(dim=["chain", "draw"]),
        color="blue",
        label="Predicted",
        ax=ax,
    )
    if plot_seasonality:
        _ = sns.lineplot(
            x=date,
            y=posterior_season.mean(dim=["chain", "draw"]),
            color="green",
            label="Seasonality + Intercept",
            ax=ax,
        )
    sns.lineplot(
        x=date,
        y=mmm.idata.observed_data.y * target_scale,
        color="black",
        label="Observed",
        ax=ax,
    )

    return fig, ax


def plot_controls_variable(data, controls):
    """Plot control variables over time.

    Parameters
    ----------
    data : pandas.DataFrame
        Input data containing a ``date`` column and control variable columns.
    controls : list[str]
        Control variable names to plot. These should be columns in ``data``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib figure containing the plot.
    ax : numpy.ndarray of matplotlib.axes.Axes
        Axes array, one subplot per control variable.
    """
    fig, ax = plt.subplots(
        nrows=len(controls),
        ncols=1,
        figsize=(10, 7),
        sharex=True,
        sharey=False,
        layout="constrained",
    )

    for i, m in enumerate(controls):
        ax[i].step(
            data["date"],
            data[m],
            color=f"C{i}",
            where="post",
        )
        ax[i].set(ylabel="")
        ax[i].set_title(m, fontsize=11)

    ax[1].set(xlabel="date")

    _ = fig.suptitle("Controls Data", fontsize=18, fontweight="bold")

    return fig, ax


def plot_media_costs(data, media):
    """Plot media costs over time for the specified media channels.

    Parameters
    ----------
    data : pandas.DataFrame
        Input data containing a ``date`` column and media cost columns.
    media : list[str]
        Media channel names to plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib figure containing the plot.
    ax : numpy.ndarray of matplotlib.axes.Axes
        Axes array, one subplot per media channel.
    """

    fig, ax = plt.subplots(
        nrows=len(media),
        ncols=1,
        figsize=(10, 7),
        sharex=True,
        sharey=False,
        layout="constrained",
    )

    for i, m in enumerate(media):
        sns.lineplot(
            x="date",
            y=m,
            data=data,
            color=f"C{i}",
            ax=ax[i],
        )
        ax[i].set(ylabel="")
        ax[i].set_title(m, fontsize=11)

    ax[1].set(xlabel="date")

    _ = fig.suptitle("Media Costs Data", fontsize=18, fontweight="bold")

    return fig, ax


def plot_spend(timeline, channels, grid, colors: list[tuple] = tab20colors):
    """Plot media spend over time for the specified channels.

    Parameters
    ----------
    timeline : Timeline
        Timeline object containing ``spend_df``.
    channels : list[str]
        Channel names to plot. These should be columns in ``spend_df``.
    grid : bool
        If ``True``, plot each channel in a separate subplot; otherwise,
        plot all channels on the same axes.
    colors : list[tuple], default=tab20colors
        Colors used for the line plots. Should have at least as many colors as channels.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib figure containing the plot.
        ax : matplotlib.axes.Axes or numpy.ndarray of matplotlib.axes.Axes
        Axes containing the line plot(s). If ``grid`` is ``True``,
        this will be an array of subplots; otherwise, it will be a single Axes.
    """
    x = timeline.spend_df

    assert "date" in x.columns, "x must contain a 'date' column."
    assert all(c in x.columns for c in channels), "All channels must be columns in x."

    if grid:
        fig, ax = plt.subplots(
            len(channels), 1, figsize=(8, 1 * len(channels)), sharex=True
        )
    else:
        fig, ax = plt.subplots(figsize=(8, 3))

    for i, c in enumerate(channels):
        if grid:
            sns.lineplot(data=x, x="date", y=c, color=colors[i], label=c, ax=ax[i])
        else:
            sns.lineplot(data=x, x="date", y=c, color=colors[i], label=c, ax=ax)

    if grid:
        for i in range(len(channels)):
            ax[i].set_ylabel("Spend")
            ax[i].legend(loc="upper left")
            ax[i].tick_params(axis="x", rotation=45)

    else:
        ax.set_ylabel("Spend")
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
        ax.tick_params(axis="x", rotation=45)

    return fig, ax


def plot_cross_correlation(data, media, controls, target: str = "y", maxlags: int = 20):
    """Plot cross-correlation between media/controls and target variable.

    Parameters
    ----------
    data : pandas.DataFrame
        Input data containing media, controls, and target variable.
    media : list[str]
        Media channel names to include in the plot.
    controls : list[str]
        Control variable names to include in the plot.
    target : str, default="y"
        Target variable name.
    maxlags : int, default=20
        Maximum number of lags to display in the cross-correlation plot.


    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib figure containing the cross-correlation plots.
    axes : numpy.ndarray of matplotlib.axes.Axes
        Axes array containing the cross-correlation and trend plots for each variable.
    """
    nplot = len(media) + len(controls)
    fig, axes = plt.subplots(nplot, 2, figsize=(8, 2.5 * nplot))
    fig.suptitle("Cross-correlation", fontsize=16, y=1.02)

    for i, col in enumerate(media + controls):
        # Cross-correlation plot : sum x[n+k] * y[n]
        axes[i, 0].xcorr(
            data[col] - data[col].mean(),
            data[target] - data[target].mean(),
            usevlines=True,
            maxlags=maxlags,
            normed=True,
            lw=2,
        )
        axes[i, 0].grid(True)
        axes[i, 0].set_title(f"{col} vs {target}")

        sns.lineplot(x="date", y=col, data=data, ax=axes[i, 1])
        axes[i, 1].tick_params(axis="x", labelrotation=45)

    plt.tight_layout()

    return fig, axes


def corr_plot(data, media, controls):
    """Build a styled correlation matrix for media, controls, and target ``y``.

    Parameters
    ----------
    data : pandas.DataFrame
        Input data.
    media : list[str]
        Media channel names to include.
    controls : list[str]
        Control variable names to include.

    Returns
    -------
    pandas.io.formats.style.Styler
        Styled correlation matrix with formatting and a color gradient.
    """

    dataframe_styled = data[media + controls + ["y"]].corr().style

    fig = (
        dataframe_styled.format(precision=2)
        .background_gradient(cmap="coolwarm")
        .set_table_styles(
            [
                {
                    "selector": "td",
                    "props": [
                        ("width", "50px"),
                        ("height", "50px"),
                        ("text-align", "center"),
                    ],
                }
            ]
        )
    )

    return fig


def plot_contributions(  # pylint: disable=too-many-arguments,too-many-positional-arguments, too-many-locals
    timeline: Timeline,
    channels: list[str],
    decomposition: bool = True,
    plot_y: bool = True,
    remove_baseline: bool = False,
    ascending: bool = True,
    colors: list[tuple] = tab20colors,
):
    """Plot channel contributions to predicted values over time.

    Parameters
    ----------
    timeline : Timeline
        Timeline object containing ``outcome_df``.
    channels : list[str]
        Channel names to display.
    decomposition : bool, default=True
        If ``True``, decompose contributions to relative shares scaled by ``y``.
        If ``False``, plot absolute contributions.
    plot_y : bool, default=True
        If ``True``, overlay observed ``y`` values.
    remove_baseline : bool, default=False
        If ``True``, do not plot the baseline contribution.
    ascending : bool, default=True
        Sort channels by total contribution in ascending order if ``True``;
        descending otherwise.
    colors : list[tuple], default=tab20colors
        Colors used for stacked areas.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib figure containing the plot.
    ax : matplotlib.axes.Axes
        Axes containing the stacked contribution plot.
    """
    x = timeline.outcome_df

    assert "date" in x.columns, "x must contain a 'date' column."
    assert all(c in x.columns for c in channels), "All channels must be columns in x."

    contributions_order = (
        x.drop(["date", timeline.target], axis=1)
        .sum()
        .sort_values(ascending=ascending)
        .index.tolist()
    )

    if decomposition:
        x[contributions_order] = (
            x[contributions_order]
            / x[contributions_order].sum(axis=1).to_numpy()[:, None]
        ) * x[timeline.target].to_numpy()[:, None]

    fig, ax = plt.subplots(figsize=(10, 4))

    base_mean = x["Baseline"].to_numpy()

    def fill_between(y, offset, icolor, label):
        ax.fill_between(
            x["date"], y, (y + offset), alpha=0.7, color=colors[icolor], label=label
        )

    if not remove_baseline:
        fill_between(0, base_mean, 14, "Baseline")
    else:
        base_mean = np.zeros_like(base_mean)

    last_fill = base_mean.copy()
    for i, c in enumerate([c for c in contributions_order if c in channels]):
        c_mean = x[c].to_numpy()
        fill_between(last_fill, c_mean, i, c)
        last_fill += c_mean

    # Plot observed & predict
    if plot_y and not remove_baseline:
        sns.lineplot(
            data=x,
            x="date",
            y=timeline.target,
            color="black",
            label="Observed",
            ax=ax,
        )

    ylim = (np.nanmin(base_mean), np.nanmax(last_fill))
    ylim *= np.array([0.7, 1.1])

    # _ = ax.set_ylim(ylim.tolist())
    _ = ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    _ = ax.set(xlabel="Date", ylabel="Y")

    return fig, ax


def plot_summary_contributions(timeline, controls=None, baseline_override=None):  # pylint: disable=too-many-locals
    """Plot baseline versus media contribution summary as a stacked bar.

    Parameters
    ----------
    timeline :  Timeline or iterable of Timeline
        One timeline object or an iterable of timeline objects.
    controls : list of str, optional
        List of control variables to exclude from the contribution calculation.
    baseline_override : list of str, optional
        List of variables to exclude from the baseline when calculating contributions.
         If None, only the "Baseline" column will be excluded.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib figure containing the summary plot.
    ax : matplotlib.axes.Axes
        Axes containing the stacked bar chart.
    """
    if controls is None:
        controls = []

    timelines, is_single = _normalize_timelines(timeline)
    fig, axes = plt.subplots(1, len(timelines), figsize=(3 * len(timelines), 5))
    if len(timelines) == 1:
        axes = np.array([axes])

    for i, name in enumerate(timelines.keys()):
        timeline_contributions = timelines[name].outcome_df

        baseline_contrib = timeline_contributions["Baseline"].sum()
        if baseline_override is not None:
            for var in baseline_override:
                baseline_contrib += timeline_contributions[var].sum()
        else:
            baseline_override = []

        media_contrib = (
            timeline_contributions.drop(
                columns=["date", "Baseline", timelines[name].target]
                + baseline_override
                + controls
            )
            .sum(axis=0)
            .sum()
        )

        controls_contrib = timeline_contributions[controls].sum().sum()

        total_contrib = baseline_contrib + media_contrib + controls_contrib
        baseline_contrib = 100 * baseline_contrib / total_contrib
        media_contrib = 100 * media_contrib / total_contrib
        controls_contrib = 100 * controls_contrib / total_contrib

        axes[i].bar(0, baseline_contrib, width=0.1, label="Baseline")
        if controls_contrib > 0:
            axes[i].bar(
                0,
                controls_contrib,
                width=0.1,
                label="Controls",
                bottom=baseline_contrib,
            )
        axes[i].bar(
            0,
            media_contrib,
            width=0.1,
            label="Media",
            bottom=baseline_contrib + controls_contrib,
        )

        for p in axes[i].patches:
            height = p.get_height()
            if height > 0:
                axes[i].annotate(
                    f"{height:.0f}%",
                    (p.get_x() + p.get_width() / 2.0, p.get_y() + height / 2.0),
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="white",
                    weight="bold",
                )

        _ = axes[i].legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2)
        _ = axes[i].set_title(name, fontsize=9, weight="bold")
        axes[i].spines["top"].set_visible(False)
        axes[i].spines["right"].set_visible(False)
        axes[i].set_xticks([])

    fig.tight_layout()
    return (fig, axes[0]) if is_single else (fig, axes)


def _normalize_timelines(timelines_input):
    """Normalize input to a list of timelines and determine if it's a single timeline.

    Parameters
    ----------
    timelines_input : Timeline or iterable of Timeline
        One timeline object or an iterable of timeline objects.

    Returns
    -------
    list of Timeline
        List of timeline objects.
    bool
        True if the input was a single timeline, False otherwise.
    """
    if hasattr(timelines_input, "outcome_df") and hasattr(timelines_input, "target"):
        return {"Timeline 1": timelines_input}, True

    if not isinstance(timelines_input, Iterable):
        raise TypeError("timeline must be a Timeline-like object or an iterable.")

    if not isinstance(timelines_input, dict):
        timelines_list = {
            f"Timeline {i + 1}": tl for i, tl in enumerate(timelines_input)
        }
    else:
        timelines_list = timelines_input

    if len(timelines_list) == 0:
        raise ValueError("At least one timeline is required.")

    return timelines_list, len(timelines_list) == 1


def plot_summary_contributions_per_media(
    timeline, controls=None, baseline_override=None
):  # pylint: disable=too-many-locals, too-many-statements
    """Plot percentage contribution by media channel.

    Parameters
    ----------
    timeline : Timeline or iterable of Timeline
        One timeline object or an iterable of timeline objects containing
        ``outcome_df``.
    controls : list of str
        List of control variables to exclude from the contribution calculation.
    baseline_override : list of str, optional
        List of variables to exclude from the baseline when calculating contributions.
         If None, only the "Baseline" column will be excluded.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib figure containing the bar chart.
    ax : matplotlib.axes.Axes
        Axes containing channel contribution percentages.
    """

    def _extract_contribution_decomposition(tl):
        timeline_contributions = tl.outcome_df
        contribution_totals = timeline_contributions.drop(
            columns=["date", "Baseline", tl.target] + baseline_override
        ).sum(axis=0)
        contribution_totals.sort_values(ascending=False, inplace=True)
        return contribution_totals / contribution_totals.sum() * 100

    def _single_timeline_colors(contribution_decomposition, controls_list):
        colors = []
        for col in contribution_decomposition.index:
            if col in controls_list:
                colors.append("green" if contribution_decomposition[col] > 0 else "red")
            elif col == "yearly_seasonality":
                colors.append("orange")
            else:
                colors.append("blue")
        return colors

    def _annotate_bars(ax, bars):
        hmax = max(b.get_height() for b in bars)
        height_offset = 5 * hmax / 200.0

        for b in bars:
            font_size = float(np.clip(4 + b.get_width() * 12, 4, 12))
            height = b.get_height()

            ax.annotate(
                f"{height:.1f}%",
                (b.get_x() + b.get_width() / 2.0, height + height_offset),
                ha="center",
                va="bottom" if height > 0 else "top",
                fontsize=font_size,
                color="black",
            )

    if baseline_override is None:
        baseline_override = []
    if controls is None:
        controls = []

    timelines, is_single = _normalize_timelines(timeline)
    decompositions = [
        _extract_contribution_decomposition(tl) for tl in timelines.values()
    ]

    if is_single:
        contribution_decomposition = decompositions[0]
        fig, ax = plt.subplots(figsize=(7, 5))

        colors = _single_timeline_colors(contribution_decomposition, controls)
        contribution_decomposition.plot.bar(ax=ax, color=colors)

        _annotate_bars(ax, ax.patches)

    else:
        labels = list(timelines.keys())

        contribution_matrix = dict(zip(labels, decompositions))

        contribution_df = pd.DataFrame(contribution_matrix).fillna(0.0)

        order = (
            contribution_df.abs()
            .mean(axis=1)
            .sort_values(ascending=False)
            .index.tolist()
        )
        contribution_df = contribution_df.loc[order]

        n_categories = len(contribution_df.index)
        n_timelines = len(labels)
        fig_width = max(7, 0.9 * n_categories + 1.3 * n_timelines)
        fig, ax = plt.subplots(figsize=(fig_width, 5))

        x = np.arange(n_categories)
        bar_width = min(0.8 / n_timelines, 0.35)
        offsets = (np.arange(n_timelines) - (n_timelines - 1) / 2.0) * bar_width
        base_palette = sns.color_palette("tab10", n_colors=n_timelines)

        for i, label in enumerate(labels):
            values = contribution_df[label].to_numpy()
            ax.bar(
                x + offsets[i],
                values,
                width=bar_width,
                color=base_palette[i],
                label=label,
            )
        _annotate_bars(ax, ax.patches)

        ax.set_xticklabels(contribution_df.index, rotation=45, ha="right")
        ax.set_xticks(x)

    ax.set_ylabel("Contribution (%)")
    ax.legend(title="Timelines", bbox_to_anchor=(1.01, 1), loc="upper left")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    return fig, ax


def plot_summary_spend_per_media(timeline):  # pylint: disable=too-many-locals
    """Plot percentage spend by media channel.

    Parameters
    ----------
    timeline : Timeline or iterable of Timeline
        One timeline object or an iterable of timeline objects containing
        ``outcome_df``.


    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib figure containing the bar chart.
    ax : matplotlib.axes.Axes
        Axes containing channel spend percentages.
    """

    def _extract_spend_decomposition(tl):
        spend_totals = tl.spend_df.drop(columns=["date", tl.target]).sum(axis=0)
        spend_totals.sort_values(ascending=False, inplace=True)
        return spend_totals / spend_totals.sum() * 100

    def _annotate_bars(ax, bars):
        for b in bars:
            height = b.get_height()
            ax.annotate(
                f"{height:.1f}%",
                (b.get_x() + b.get_width() / 2.0, height * 1.05),
                ha="center",
                va="bottom" if height > 0 else "top",
                fontsize=11,
                color="black",
            )

    timelines, _ = _normalize_timelines(timeline)
    decompositions = [_extract_spend_decomposition(tl) for tl in timelines]

    labels = [f"Timeline {index + 1} " for index in range(len(timelines))]

    spend_matrix = dict(zip(labels, decompositions))
    spend_df = pd.DataFrame(spend_matrix).fillna(0.0)

    order = spend_df.abs().mean(axis=1).sort_values(ascending=False).index.tolist()
    spend_df = spend_df.loc[order]

    n_categories = len(spend_df.index)
    n_timelines = len(labels)
    fig_width = max(7, 0.9 * n_categories + 1.3 * n_timelines)
    fig, ax = plt.subplots(figsize=(fig_width, 5))

    x = np.arange(n_categories)
    bar_width = min(0.8 / n_timelines, 0.35)
    offsets = (np.arange(n_timelines) - (n_timelines - 1) / 2.0) * bar_width
    base_palette = sns.color_palette("tab10", n_colors=n_timelines)

    for i, label in enumerate(labels):
        values = spend_df[label].to_numpy()
        _ = ax.bar(
            x + offsets[i],
            values,
            width=bar_width,
            color=base_palette[i],
            label=label,
        )
    _annotate_bars(ax, ax.patches)

    min_val = spend_df.min().min()
    max_val = spend_df.max().max()
    ax.set_ylim(min_val * 1.2, max_val * 1.2)

    ax.set_xticks(x)
    ax.set_xticklabels(spend_df.index, rotation=90)
    ax.set_ylabel("Contribution (%)")
    ax.legend(title="Timelines", bbox_to_anchor=(1.01, 1), loc="upper left")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    return fig, ax


def plot_saturation_curves(  # pylint: disable=too-many-locals
    curves, mmm, data, media
):
    """Plot saturation curves for each media channel.

    Parameters
    ----------
    curves : dict[str, xarray.DataArray]
        Dictionary mapping media channel names to their corresponding saturation curve data arrays.
    mmm : MediaMixModel
        Fitted media mix model containing saturation specifications and inference data.
    data : pandas.DataFrame
        Input data containing media cost columns and observed contributions.
    media : list[str]
        Media channel names to plot. These should be columns in ``data`` and
        correspond to media channels in the model.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib figure containing the saturation curve plots.
    axes : dict[str, matplotlib.axes.Axes]
        Dictionary mapping media channel names to their corresponding Axes objects.
    """
    fig, ax = plt.subplots(len(media) // 3 + 1, 3, figsize=(16, 5))
    media_scales = mmm._scales["media"]  # pylint: disable=protected-access
    target_scale = mmm._scales["y"][0]  # pylint: disable=protected-access

    contrib = mmm.idata.posterior.media_contribution.sel(media=media).mean(
        dim=["chain", "draw"]
    )

    for i, m in enumerate(media):
        curve = curves[m]

        xx = (curve.coords["x"]).values * media_scales[i]
        beta = (
            mmm.idata.posterior["beta_media"]
            .sel(media=m)
            .mean(dim=["chain", "draw"])
            .values
        )
        yy = beta * curve.mean(dim=["chain", "draw"]).values * target_scale

        axi = ax[i // 3, i % 3]

        axi.plot(
            xx,
            yy,
            label=m,
        )

        axi.plot(
            data[m],
            contrib.sel(media=m) * target_scale,
            "o",
            alpha=0.5,
            label=m,
        )

        # present_contrib = yy[np.argmin(np.abs(xx - total_spend[m]))]
        # axi.plot(
        #     total_spend[m],
        #     present_contrib,
        #     "X",
        #     markersize=10,
        #     color="red",
        # )
        # axi.axvline(total_spend[m], color="black", lw=0.5, ls="--")
        # axi.axhline(present_contrib, color="black", lw=0.5, ls="--")

        axi.set_title(m)

    for axi in ax.ravel():
        if not axi.has_data():
            axi.remove()

    plt.tight_layout()

    return fig, ax
