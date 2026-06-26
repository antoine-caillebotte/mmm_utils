"""Post-modeling utilities for media mix modeling."""

import pandas as pd

import arviz as az
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xarray as xr
from pytensor.xtensor.type import as_xtensor

from mmm_utils.modeling.mmm import MMM
from mmm_utils.data_logger import data_logger


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


def summarize_high_mcse_mean(
    summary_df: pd.DataFrame,
    relative_threshold: float = 0.1,
) -> pd.DataFrame:
    """
    Identify parameters with potentially large `mcse_mean`.

    Parameters
    ----------
    summary_df : pd.DataFrame
        ArviZ summary output containing at least:
        `parameter`, `mcse_mean`, and `sd`.
    relative_threshold : float, default=0.1
        Threshold on `mcse_mean / sd`.
        < 1 % Excellent — more than enough iterations
        1-5 % Acceptable for most uses
        5-10 % Limit — consider more iterations
        > 10 % Insufficient

    Returns
    -------
    pd.DataFrame
        Filtered dataframe with problematic parameters, sorted by
        relative and absolute MCSE magnitude.

    Raises
    ------
    ValueError
        If the input dataframe is missing required columns.
    """
    required_columns = {"parameter", "mcse_mean", "sd"}
    missing_columns = required_columns - set(summary_df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    result_df = summary_df.copy()
    result_df["mcse_over_sd"] = result_df["mcse_mean"] / result_df["sd"].replace(
        0, np.nan
    )

    result_df["decision"] = pd.cut(
        result_df["mcse_over_sd"],
        bins=[-np.inf, 0.01, 0.05, 0.1, np.inf],
        labels=[
            "Excellent (<1%)",
            "Acceptable (1-5%)",
            "Limit (5-10%)",
            "Insufficient (>10%)",
        ],
    )
    result_df = result_df[["parameter", "mcse_mean", "mcse_over_sd", "decision"]]

    data_logger.clear()
    data_logger.record(dataframe=result_df)
    data_logger.flush_to_csv("mcse_summary.csv")

    mask = result_df["mcse_over_sd"] > relative_threshold

    return result_df.loc[mask].sort_values(
        by=["mcse_over_sd", "mcse_mean"], ascending=False
    )


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
            decision = "~"
        elif p_above_rope >= decision_threshold:
            decision = ">"
        elif p_below_rope >= decision_threshold:
            decision = "<"
        else:
            decision = "?"

        alpha = 1 - decision_threshold

        def _format_with_alpha(probability: float) -> str:
            rounded_probability = round(probability, 2)
            n_alpha = int(max(0, (1.0 - probability) / alpha))
            if n_alpha > 4:
                n_alpha = 0
            else:
                n_alpha = 4 - n_alpha

            if n_alpha == 0:
                return f"{rounded_probability:.2f}"

            return f"{rounded_probability:.2f} ({'*' * n_alpha})"

        rows.append(
            {
                "parameter": parameter_name,
                "rope_low": f"{rope_low:.2e}",
                "rope_high": f"{rope_high:.2e}",
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

    data_logger.clear()
    data_logger.record(rope_df)
    data_logger.flush_to_csv("rope_probabilities.csv")

    if verbatim:
        return rope_df

    return rope_df[rope_df["decision"] == "?"]


def plot_adstock_effects(data, mmm: MMM, media: list[str]):  # pylint: disable=too-many-locals
    """Plot adstock-transformed media effects over time.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset containing a ``date`` column and media spend columns.
    mmm : MMM
        Fitted media mix model containing adstock definitions and posterior data.
    media : list[str]
        Media channel names to include in the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib figure containing the adstock effects plot.
    ax : matplotlib.axes.Axes
        Axes containing one line per media channel adstock effect.
    """
    time = data["date"]
    fig, ax = plt.subplots(figsize=(10, 6))

    data_logger.clear()
    data_logger.record(date=time)

    media_scales = mmm.data.scale("media")

    for m in media:
        adstock = mmm.adstocks[m]

        adstock_param = {}
        for k, p in adstock["params"].items():
            pname = str(p)
            if pname in mmm.idata.posterior:
                adstock_param[k] = float(
                    mmm.idata.posterior[pname].mean(dim=["chain", "draw"]).values
                )
            else:
                adstock_param[k] = p

        spend_np = data[m] / media_scales[m]
        spend_np = as_xtensor(spend_np, dims=["date"])
        y = adstock["function"](spend_np, adstock_param).eval()
        y_np = np.asarray(y, dtype=np.float64).ravel() * media_scales[m]
        ax.plot(time, y_np, label=f"{m} ({adstock_param['alpha']:.2f})")

        data_logger.record(
            **{f"{m}_adstock_effect": y_np, f"{m}_alpha": adstock_param["alpha"]}
        )

    ax.set_xlabel("Lag")
    ax.set_ylabel("Adstock Effect")
    ax.set_title("Adstock Effects by Media")
    ax.legend()

    data_logger.flush_to_csv("adstock_effects.csv")

    return fig, ax


def plot_residuals(mmm, ax=None):
    """Plot residuals of the fitted media mix model.

    Parameters
    ----------
    mmm : MediaMixModel
        Fitted media mix model containing inference data with posterior predictive samples.

    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure and axes will be created.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib figure containing the plot.
    ax : matplotlib.axes.Axes
        Axes containing the residuals plot.
    """
    if ax is None:
        data_logger.clear()

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    target_scale = mmm.data.scale("y")
    posterior_predictive_y = mmm.idata.posterior_predictive.y * target_scale
    observed_y = mmm.idata.observed_data.y * target_scale

    residuals = observed_y - posterior_predictive_y.mean(dim=["chain", "draw"])
    date = mmm.idata.posterior_predictive.date

    data_logger.record(
        date=np.asarray(date.values),
        observed_y=np.asarray(observed_y.values),
        predicted_y_mean=np.asarray(
            posterior_predictive_y.mean(dim=["chain", "draw"]).values
        ),
        residuals=np.asarray(residuals.values),
    )

    ax.bar(
        date,
        residuals,
        width=pd.Timedelta(days=5),
        color=["#16A34A" if r >= 0 else "#DC2626" for r in residuals],
        alpha=0.7,
    )

    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("Residuals")

    if ax is None:
        data_logger.flush_to_csv("residuals_plot.csv")

    return ax.figure, ax


def plot_seasonality(mmm, ax=None):
    """Plot seasonality contribution of the fitted media mix model.

    Parameters
    ----------
    mmm : MediaMixModel
        Fitted media mix model containing inference data with posterior predictive samples.

    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure and axes will be created.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib figure containing the plot.
    ax : matplotlib.axes.Axes
        Axes containing the seasonality contribution plot.
    """
    if ax is None:
        data_logger.clear()

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    target_scale = mmm.data.scale("y")
    posterior_intercept = (
        mmm.idata.posterior.controls["intercept"] * target_scale
        if "intercept" in mmm.idata.posterior.controls
        else 0.0
    )
    posterior_season = (
        mmm.idata.posterior.yearly_seasonality_contribution
    ) * target_scale

    date = mmm.idata.posterior_predictive.date
    posterior_season_mean = posterior_season.mean(dim=["chain", "draw"])
    posterior_intercept_mean = posterior_intercept.mean(dim=["chain", "draw"])

    data_logger.record(
        date=np.asarray(date.values),
        seasonality_intercept=np.asarray(
            posterior_season_mean.values + posterior_intercept_mean.values
        ),
    )

    sns.lineplot(
        x=date,
        y=posterior_season_mean,
        color="green",
        label="Seasonality + Intercept",
        ax=ax,
    )

    if ax is None:
        data_logger.flush_to_csv("seasonality_plot.csv")

    return ax.figure, ax


def plot_posterior_predictive_y(
    mmm, add_seasonality: bool = True, add_residuals: bool = False
):
    """Plot posterior predictive distribution of ``y`` with observed data.

    Parameters
    ----------
    mmm : MediaMixModel
        Fitted media mix model containing inference data with posterior predictive samples.

    add_seasonality : bool, default=True
        If ``True``, overlay the posterior seasonality and intercept contributions.

    add_residuals : bool, default=False
        If ``True``, overlay the residuals of the model.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib figure containing the plot.
    ax : matplotlib.axes.Axes
        Axes containing the posterior predictive distribution and observed data.
    """

    data_logger.clear()

    def _make_plot(ax):
        target_scale = mmm.data.scale("y")
        posterior_predictive_y = mmm.idata.posterior_predictive.y * target_scale
        date = mmm.idata.posterior_predictive.date
        predicted_mean = posterior_predictive_y.mean(dim=["chain", "draw"])
        observed_y = mmm.idata.observed_data.y * target_scale

        data_logger.record(
            date=np.asarray(date.values),
            predicted_y_mean=np.asarray(predicted_mean.values),
            observed_y=np.asarray(observed_y.values),
        )

        for i, hdi_prob in enumerate([0.94, 0.5]):
            hdi = az.hdi(
                posterior_predictive_y.unstack().transpose(..., "date"),
                prob=hdi_prob,
            )
            lower = np.asarray(hdi.sel(ci_bound="lower"))
            upper = np.asarray(hdi.sel(ci_bound="upper"))
            data_logger.record(
                **{
                    f"hdi_{int(hdi_prob * 100)}_lower": lower,
                    f"hdi_{int(hdi_prob * 100)}_upper": upper,
                }
            )
            ax.fill_between(
                x=date,
                y1=lower,
                y2=upper,
                # smooth=False,
                color="C0",
                alpha=0.3 + i * 0.1,
                label=f"{hdi_prob:.0%} HDI",
            )

            ax.plot(date, predicted_mean, color="C0")

        _ = sns.lineplot(
            x=date,
            y=predicted_mean,
            color="blue",
            label="Predicted",
            ax=ax,
        )
        sns.lineplot(
            x=date,
            y=observed_y,
            color="black",
            label="Observed",
            ax=ax,
        )
        if add_seasonality:
            _, ax = plot_seasonality(mmm, ax=ax)

        return fig, ax

    if add_residuals:
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(10, 6),
            sharex=True,
            height_ratios=[3, 1],
            layout="constrained",
        )
        _, axes[1] = plot_residuals(mmm, ax=axes[1])
        _, axes[0] = _make_plot(axes[0])
        data_logger.flush_to_csv("posterior_predictive_y_plot.csv")
        return fig, axes

    fig, ax = plt.subplots(figsize=(10, 4))
    _, ax = _make_plot(ax)
    data_logger.flush_to_csv("posterior_predictive_y_plot.csv")
    return fig, ax


def adstock_to_half_life(mmm, media: list[str]) -> pd.DataFrame:
    """Convert adstock alpha parameters to half-life and end-of-effect metrics.

    Parameters
    ----------
    mmm : MediaMixModel
        Fitted media mix model containing adstock priors/parameters and posterior draws.
    media : list[str]
        Media channel names to evaluate.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per media channel and columns:
        ``media``, ``adstock_alpha``, ``half_life``, and ``end_adstock``.
    """
    adstock_alpha = {
        m: f"adstock_alpha[{m}]"
        for m in media
        if m in mmm.config.media_transforms
        and "alpha" in mmm.config.media_transforms[m].adstock_priors
    }
    if len(adstock_alpha) == 0:
        return pd.DataFrame({"media": [], "adstock_alpha": [], "half_life": []})

    alphas = (
        mmm.idata.posterior.to_dataset()[list(adstock_alpha.values())]
        .mean(dim=["chain", "draw"])
        .to_array()
    )

    fixed_alphas = {
        m: mmm.config.media_transforms[m].adstock_params["alpha"]
        for m in media
        if m in mmm.config.media_transforms
        and "alpha" in mmm.config.media_transforms[m].adstock_params
    }
    df_media = list(adstock_alpha.keys()) + list(fixed_alphas.keys())
    alphas = np.concatenate(
        [
            alphas.to_numpy(),
            np.array([fixed_alphas[m] for m in fixed_alphas]),
        ]
    )

    half_lives = -np.log(2) / np.log(alphas)
    end_adstock = -np.log(100) / np.log(alphas)
    x = pd.DataFrame(
        {
            "media": df_media,
            "adstock_alpha": alphas,
            "half_life": half_lives,
            "end_adstock": end_adstock,
        }
    )

    data_logger.direct_to_csv("adstock_to_half_life.csv", dataframe=x)
    return x
