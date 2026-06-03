"""Post-modeling utilities for media mix modeling."""

import pandas as pd

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xarray as xr


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


def rope_probability_test(  # pylint: disable=too-many-arguments, too-many-locals
    posterior: xr.DataArray,
    var: list[str] | None = None,
    rope: tuple[float, float] = (-0.1, 0.1),
    *,
    rope_by_parameter: dict[str, tuple[float, float]] | None = None,
    decision_threshold: float = 0.95,
    verbatim: bool = True,
) -> pd.DataFrame | pd.Series:
    """Compute ROPE probabilities for every posterior parameter.

    https://easystats.github.io/bayestestR/articles/region_of_practical_equivalence.html

    Parameters
    ----------
    posterior : xr.DataArray
        Posterior dataset containing the sampled parameters.
    var : list[str] | None, optional
        List of variable names to include in the ROPE analysis. If None, all variables are included.
    rope : tuple[float, float], optional
        Default ROPE interval applied to all parameters.
    rope_by_parameter : dict[str, tuple[float, float]] | None, optional
        Optional per-parameter ROPE overrides. Keys can target either a base
        variable name (for example "beta_media") or an indexed parameter name
        (for example "beta_media[TV]").
    decision_threshold : float, optional
        Probability threshold used for the decision label.
    verbatim : bool, optional
        If True, return the full DataFrame with ROPE probabilities and decisions.
        If False, return only the parameters where the decision is "undetermined".

    Returns
    -------
    pd.DataFrame | pd.Series
        If verbatim is True, one row per scalar posterior parameter with ROPE
        probabilities. If verbatim is False, a boolean Series where True means
        p_in_rope >= decision_threshold.
    """

    rope_by_parameter = rope_by_parameter or {}
    rows: list[dict[str, float | str]] = []

    def _resolve_rope(parameter_name: str, base_name: str) -> tuple[float, float]:
        if parameter_name in rope_by_parameter:
            return rope_by_parameter[parameter_name]
        if base_name in rope_by_parameter:
            return rope_by_parameter[base_name]
        return rope

    def _compute_row(
        parameter_name: str,
        base_name: str,
        samples: np.ndarray,
    ) -> None:
        rope_low, rope_high = _resolve_rope(parameter_name, base_name)
        rope_low *= np.std(samples)
        rope_high *= np.std(samples)

        p_in_rope = float(np.mean((samples >= rope_low) & (samples <= rope_high)))
        p_below_rope = float(np.mean(samples < rope_low))
        p_above_rope = float(np.mean(samples > rope_high))

        if p_in_rope >= decision_threshold:
            decision = "practically_equivalent"
        elif p_above_rope >= decision_threshold:
            decision = "practically_greater"
        elif p_below_rope >= decision_threshold:
            decision = "practically_lower"
        else:
            decision = "undetermined"

        alpha = 1 - decision_threshold

        def _format_with_alpha(probability: float) -> str:
            rounded_probability = round(probability, 2)
            n_alpha = (
                int(max(0, round((1.0 - rounded_probability) / alpha)))
                if alpha > 0
                else 0
            )
            if n_alpha > 4:
                n_alpha = 0
            else:
                n_alpha = 4 - n_alpha

            return f"{rounded_probability:.2f} ({'*' * n_alpha})"

        rows.append(
            {
                "parameter": parameter_name,
                "rope_low": float(rope_low),
                "rope_high": float(rope_high),
                "lower": _format_with_alpha(p_below_rope),
                "in": _format_with_alpha(p_in_rope),
                "greater": _format_with_alpha(p_above_rope),
                "decision": decision,
            }
        )

    for var_name, var_da in posterior.data_vars.items():
        if var is not None and var_name not in var:
            continue

        stacked = var_da.stack(sample=("chain", "draw"))
        value_dims = [dim for dim in stacked.dims if dim != "sample"]

        if not value_dims:
            samples = np.asarray(stacked.values, dtype=np.float64).ravel()
            _compute_row(var_name, var_name, samples)
            continue

        dim_sizes = [stacked.sizes[dim] for dim in value_dims]
        dim_coords = [stacked.coords[dim].values for dim in value_dims]

        for idx in np.ndindex(*dim_sizes):
            selector = dict(zip(value_dims, idx))
            sub = stacked.isel(**selector)
            samples = np.asarray(sub.values, dtype=np.float64).ravel()

            coord_label = ",".join(str(dim_coords[k][i]) for k, i in enumerate(idx))
            parameter_name = f"{var_name}[{coord_label}]"
            _compute_row(parameter_name, var_name, samples)

    rope_df = pd.DataFrame(rows).set_index("parameter").sort_index()

    if verbatim:
        return rope_df

    return rope_df[rope_df["decision"] == "undetermined"]
