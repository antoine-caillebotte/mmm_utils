"""Post-modeling utilities for media mix modeling."""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def get_dist_parameters(mmm, var, imedia):
    """
    Retrieve the distribution parameters for a given variable and media index.

    Parameters
    ----------
    mmm : object
        The media mix model object containing model configuration.
    var : str
        The variable name for which to retrieve parameters.
    imedia : int
        The index of the media to retrieve parameters for.

    Returns
    -------
    dict
        A dictionary containing the distribution parameters
        for the specified variable and media index.
    """

    parameters = mmm.model_config[var].parameters
    dist = mmm.model_config[var].distribution
    print(f"parameters of {dist}: {parameters}")

    p = {k: np.array(parameters[k]) for k in parameters}
    for k in p:
        if isinstance(p[k], np.ndarray) and p[k].ndim == 1:
            p[k] = p[k][imedia]
    # p["variance"] = p.pop("sigma")
    # p["loc"] = 0
    # print(p)
    # if var == "beta":

    if dist == "HalfNormal":
        p["loc"] = 0

    return p


def plot_prior_vs_posterior(
    mmm, var, media, *, ncol=3, figsize=(12, 8), seperately=False
):  # pylint: disable=too-many-arguments
    """
    Plot prior and posterior distributions for a given variable and media.

    Parameters
    ----------
    mmm : object
        The media mix model object containing prior and posterior data.
    var : str
        The variable name to plot.
    media : list of str
        A list of media names to include in the plot.
    ncol : int, optional
        Number of columns for subplots. Defaults to 3.
    figsize : tuple of int, optional
        Figure size for the plot. Defaults to (12, 8).
    seperately : bool, optional
        Whether to plot each media separately. Defaults to False.

    Returns
    -------
    tuple
        A tuple containing the figure and axes of the plot.
    """

    def _make_plot(ax, i):
        sample = mmm.prior[var].values[:, :, i].flatten()
        sns.kdeplot(
            sample,
            ax=ax,
            label=f"{media[i]} - Prior (sampled)",
            color=f"C{i}",
            fill=True,
            cut=0 if sample.min() >= 0 else 3,
        )

        sample = mmm.posterior[var].values[:, :, i].flatten()
        sns.kdeplot(
            sample,
            ax=ax,
            label=f"{media[i]} - Posterior",
            color=f"C{i}",
            fill=False,
            cut=0 if sample.min() >= 0 else 3,
        )

        x_grid = np.linspace(0, sample.max(), 100)

        dist = mmm.model_config[var].pymc_distribution
        dist_name = mmm.model_config[var].distribution
        p = get_dist_parameters(mmm, var, i)

        pdf_values = np.exp(dist.logp(x_grid, **p).eval())

        kde_max = max(
            (
                np.nanmax(line.get_ydata())
                for line in ax.lines
                if len(line.get_ydata()) > 0
            ),
            default=np.nan,
        )

        if np.isfinite(kde_max):
            pdf_values = np.where(pdf_values <= 1.3 * kde_max, pdf_values, np.nan)

        ax.plot(
            x_grid,
            pdf_values,
            color=f"C{i}",
            linestyle="--",
            linewidth=2,
            label=f"{media[i]} - {dist_name} PDF",
        )

        ax.legend()

    if seperately:
        nrow = int(np.ceil(len(media) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=figsize)

        if ncol == 1:
            axes = axes[:, None]
        if nrow == 1:
            axes = axes[None, :]

        for i, _ in enumerate(media):
            _make_plot(axes[i // ncol, i % ncol], i)

        for ax in axes.flatten()[len(media) :]:
            ax.set_visible(False)

    else:
        fig, axes = plt.subplots(figsize=figsize)
        for i, _ in enumerate(media):
            _make_plot(axes, i)

        axes.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=len(media))

    return fig, axes
