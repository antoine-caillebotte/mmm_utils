"""This module provides utilities for handling and
visualizing priors in PyMC-based MMM models, including:
"""

from dataclasses import dataclass, field

import re

from typing import Literal
import numpy as np
import pymc as pm
import seaborn as sns
import matplotlib.pyplot as plt


PriorType = Literal[
    "HalfNormal", "Normal", "Gamma", "Beta", "LogNormal", "Laplace", "LaPlace"
]


@dataclass(slots=True)
class PriorSpec:
    """Spécification de prior PyMC."""

    kind: PriorType
    params: dict[str, float] = field(default_factory=dict)


def _make_prior(name: str, spec: PriorSpec, dims: str | tuple[str, ...] | None = None):
    """Build a PyMC prior variable from a PriorSpec.

    Parameters
    ----------
    name : str
        Name of the random variable in the model.
    spec : PriorSpec
        Distribution kind and parameter dictionary.
    dims : str | tuple[str, ...] | None, optional
        Optional named dimensions passed to PyMC.

    Returns
    -------
    pm.TensorVariable
        PyMC random variable matching the requested prior.

    Raises
    ------
    ValueError
        If ``spec.kind`` is not supported.
    """
    if spec.kind == "HalfNormal":
        return pm.HalfNormal(name, sigma=spec.params.get("sigma", 1.0), dims=dims)
    if spec.kind == "Normal":
        return pm.Normal(
            name,
            mu=spec.params.get("mu", 0.0),
            sigma=spec.params.get("sigma", 1.0),
            dims=dims,
        )
    if spec.kind == "Gamma":
        return pm.Gamma(
            name,
            alpha=spec.params.get("alpha", 2.0),
            beta=spec.params.get("beta", 1.0),
            dims=dims,
        )
    if spec.kind == "Beta":
        return pm.Beta(
            name,
            alpha=spec.params.get("alpha", 2.0),
            beta=spec.params.get("beta", 2.0),
            dims=dims,
        )
    if spec.kind == "LogNormal":
        return pm.LogNormal(
            name,
            mu=spec.params.get("mu", 0.0),
            sigma=spec.params.get("sigma", 1.0),
            dims=dims,
        )

    if spec.kind in ("Laplace", "LaPlace"):
        return pm.Laplace(
            name,
            mu=spec.params.get("mu", 0.0),
            b=spec.params.get("b", 1.0),
            dims=dims,
        )
    raise ValueError(f"Unknown prior type: {spec.kind}")


def _prior_pdf(prior_spec: PriorSpec, x_grid: np.ndarray, name_idx: int) -> np.ndarray:
    """Evaluate the prior PDF defined by a PriorSpec on a grid.

    Parameters
    ----------
    prior_spec : PriorSpec
        Prior specification.
    x_grid : np.ndarray
        Grid values where the PDF is evaluated.

    Returns
    -------
    np.ndarray
        Prior PDF values.
    """

    def _return_param(name, default_value):
        param = prior_spec.params.get(name, default_value)
        if isinstance(param, np.ndarray) and param.size > 1:
            return param[name_idx]
        return param

    if prior_spec.kind == "HalfNormal":
        dist = pm.HalfNormal.dist(sigma=_return_param("sigma", 1.0))
    elif prior_spec.kind == "Normal":
        dist = pm.Normal.dist(
            mu=_return_param("mu", 0.0),
            sigma=_return_param("sigma", 1.0),
        )
    elif prior_spec.kind == "Gamma":
        dist = pm.Gamma.dist(
            alpha=_return_param("alpha", 2.0),
            beta=_return_param("beta", 1.0),
        )
    elif prior_spec.kind == "Beta":
        dist = pm.Beta.dist(
            alpha=_return_param("alpha", 2.0),
            beta=_return_param("beta", 2.0),
        )
    elif prior_spec.kind == "LogNormal":
        dist = pm.LogNormal.dist(
            mu=_return_param("mu", 0.0),
            sigma=_return_param("sigma", 1.0),
        )
    elif prior_spec.kind in ("Laplace", "LaPlace"):
        dist = pm.Laplace.dist(
            mu=_return_param("mu", 0.0),
            b=_return_param("b", 1.0),
        )
    else:
        raise ValueError(f"Unsupported prior type for plotting: {prior_spec.kind}")

    return np.exp(pm.logp(dist, x_grid).eval())


def _get_group_var_values(group, var: str, media: str) -> np.ndarray:
    """Extract flattened draws for one indexed component from an idata group."""
    if var in group.data_vars:
        da = group[var]
    elif f"{var}[{media}]" in group.data_vars:
        da = group[f"{var}[{media}]"]
    else:
        raise ValueError(
            f"Variable '{var}' or '{var}[{media}]' not found in idata group."
            " Available variables: {list(group.data_vars)}"
        )

    value_dims = [d for d in da.dims if d not in ("chain", "draw")]

    if value_dims:
        assert (
            media in da.coords[value_dims[0]]
        ), f"Media '{media}' not found in variable '{var}' coordinates."
        da = da.sel({value_dims[0]: media})

    return np.asarray(da.values, dtype=np.float64).reshape(-1)


def _resolve_prior_spec_for_var(mmm, var: str, media_name: str) -> PriorSpec:
    """Resolve PriorSpec from MMM config for a variable/component."""
    cfg = mmm.config

    direct_map = {
        "intercept": cfg.prior_intercept,
        "beta_media": cfg.prior_media,
        "beta_control": cfg.prior_control,
        "sigma": cfg.prior_sigma,
        "beta_season": cfg.prior_season,
        "beta_trend": cfg.prior_trend,
    }
    if var in direct_map:
        return direct_map[var]

    # var = adstock_theta, adstock_alpha & media_name = TV, SEA, ...
    if media_name in cfg.media_transforms.keys():
        param_name = re.sub(r"adstock_(\w+)", r"\1", var)
        if param_name in cfg.media_transforms[media_name].adstock_priors.keys():
            spec = cfg.media_transforms[media_name].adstock_priors.get(param_name)
            return spec

    # var = saturation_lam & media_name = TV, SEA], ...
    if var == "saturation_lam" and media_name in cfg.media_transforms.keys():
        spec = cfg.media_transforms[media_name].saturation_priors.get("lam")
        return spec

    raise ValueError(f"Cannot resolve prior specification for variable '{var}'")


def plot_prior_vs_posterior(
    mmm, var: str, media: list[str], *, ncol: int = 3, figsize=(12, 8), seperately=False
):  # pylint: disable=too-many-arguments
    """Plot prior and posterior distributions for a given variable and media.

    Parameters
    ----------
    mmm : object
        MMM object containing inference data and config priors.
    var : str
        Variable name to plot.
    media : list[str]
        Media names used for component labels.
    ncol : int, optional
        Number of columns for subplots when ``seperately`` is True.
    figsize : tuple[int, int], optional
        Figure size.
    seperately : bool, optional
        Whether to create one subplot per media component.

    Returns
    -------
    tuple
        Matplotlib figure and axes.

    Raises
    ------
    RuntimeError
        If posterior or prior draws are missing in the idata.
    ValueError
        If the prior specification for the variable cannot be resolved.
    """
    if mmm.idata is None or "posterior" not in mmm.idata:
        raise RuntimeError("Posterior draws are required to plot prior vs posterior.")
    if "prior" not in mmm.idata:
        raise RuntimeError(
            "Prior draws are required in idata to plot prior vs posterior."
        )

    prior_group = mmm.idata.prior
    posterior_group = mmm.idata.posterior

    def _make_plot(ax, i, name):
        prior_sample = _get_group_var_values(prior_group, var, name)
        posterior_sample = _get_group_var_values(posterior_group, var, name)

        sns.kdeplot(
            prior_sample,
            ax=ax,
            label=f"{name} - Prior (sampled)",
            color=f"C{i}",
            fill=True,
            cut=0 if np.min(prior_sample) >= 0 else 3,
        )

        sns.kdeplot(
            posterior_sample,
            ax=ax,
            label=f"{name} - Posterior",
            color=f"C{i}",
            fill=False,
            cut=0 if np.min(posterior_sample) >= 0 else 3,
        )

        x_min = min(float(np.min(prior_sample)), float(np.min(posterior_sample)))
        x_max = max(float(np.max(prior_sample)), float(np.max(posterior_sample)))
        x_grid = np.linspace(x_min, x_max, 200)

        prior_spec = _resolve_prior_spec_for_var(mmm, var, name)

        if var in prior_group.data_vars:
            da_prior = prior_group[var]
            dims = [d for d in da_prior.dims if d not in ("chain", "draw")]
            coord_values = np.asarray(da_prior.coords[dims[0]].values)
            matches = np.where(coord_values == name)[0]
            name_idx = int(matches[0])
        else:
            name_idx = 0

        pdf_values = _prior_pdf(prior_spec, x_grid, name_idx)

        ax.plot(
            x_grid,
            np.where(pdf_values < 25, pdf_values, np.nan),
            color=f"C{i}",
            linestyle="--",
            linewidth=2,
            label=f"{name} - {prior_spec.kind} PDF",
        )
        ax.legend()

    if seperately:
        nrow = int(np.ceil(len(media) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=figsize)

        if ncol == 1:
            axes = axes[:, None]
        if nrow == 1:
            axes = axes[None, :]

        for i, m in enumerate(media):
            _make_plot(axes[i // ncol, i % ncol], i, m)

        for ax in axes.flatten()[len(media) :]:
            ax.set_visible(False)
    else:
        fig, axes = plt.subplots(figsize=figsize)
        for i, m in enumerate(media):
            _make_plot(axes, i, m)
        axes.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=len(media))

    fig.suptitle(var, fontsize=16, fontweight="bold")
    plt.tight_layout()
    return fig, axes
