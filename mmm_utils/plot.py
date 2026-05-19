"""Plotting utilities for media mix modeling."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from .timeline import Timeline

tab20colors = plt.get_cmap("tab20").colors


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


def plot_cross_correlation(data, media, controls, target: str = "y"):
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
            maxlags=20,
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

    fig, ax = plt.subplots(figsize=(10, 5))

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
            y="y",
            color="black",
            label="Observed",
            ax=ax,
        )

    ylim = np.array([0.7, 1.1]) * (np.min(base_mean), np.max(last_fill))
    _ = ax.set_ylim(ylim.tolist())
    _ = ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    _ = ax.set(xlabel="Date", ylabel="Y")

    return fig, ax


def plot_summary_contributions(timeline):
    """Plot baseline versus media contribution summary as a stacked bar.

    Parameters
    ----------
    timeline : Timeline
        Timeline object containing ``outcome_df``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib figure containing the summary plot.
    ax : matplotlib.axes.Axes
        Axes containing the stacked bar chart.
    """

    timeline_contributions = timeline.outcome_df
    baseline_contrib = timeline_contributions["Baseline"].sum()
    media_contrib = (
        timeline_contributions.drop(columns=["date", "Baseline", timeline.target])
        .sum(axis=0)
        .sum()
    )
    total_contrib = baseline_contrib + media_contrib
    baseline_contrib = 100 * baseline_contrib / total_contrib
    media_contrib = 100 * media_contrib / total_contrib

    fig, ax = plt.subplots(figsize=(4, 5))
    ax.bar(0, baseline_contrib, width=0.2, label="Baseline")
    ax.bar(0, media_contrib, width=0.2, label="Media", bottom=baseline_contrib)

    for p in plt.gca().patches:
        height = p.get_height()
        if height > 0:
            plt.gca().annotate(
                f"{height:.0f}%",
                (p.get_x() + p.get_width() / 2.0, p.get_y() + height / 2.0),
                ha="center",
                va="center",
                fontsize=10,
                color="white",
                weight="bold",
            )

    _ = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2)
    return fig, ax


def plot_summary_contributions_per_media(timeline):
    """Plot percentage contribution by media channel.

    Parameters
    ----------
    timeline : Timeline
        Timeline object containing ``outcome_df``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib figure containing the bar chart.
    ax : matplotlib.axes.Axes
        Axes containing channel contribution percentages.
    """
    timeline_contributions = timeline.outcome_df
    contribution_totals = timeline_contributions.drop(
        columns=["date", "Baseline", timeline.target]
    ).sum(axis=0)
    contribution_totals.sort_values(ascending=False, inplace=True)
    contribution_decomposition = contribution_totals / sum(contribution_totals) * 100

    fig, ax = plt.subplots(figsize=(7, 5))

    contribution_decomposition.plot.bar(ax=ax)

    ax.set_ylim(0, contribution_decomposition.max() * 1.2)
    for patch in ax.patches:
        height = patch.get_height()
        if height > 0:
            ax.annotate(
                f"{height:.1f}%",
                (patch.get_x() + patch.get_width() * 0.6, patch.get_height() + 4),
                ha="center",
                va="top",
                fontsize=11,
                color="black",
            )

    plt.tight_layout()
    return fig, ax
