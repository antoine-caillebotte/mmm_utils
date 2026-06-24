"""This module implements the MMM class,
a PyMC-based Bayesian Media Mix Modeling framework
with adstock and saturation transformations."""

import warnings
from dataclasses import dataclass, field

# from typing import Callable
import numpy as np
import pandas as pd
import arviz as az
import pymc as pm
import pymc.dims as pmd
import pytensor.xtensor as ptx


from .utils import ArrayLike, max_abs_scaler
from .seasonality import fourier_features
from .prior import PriorSpec, _make_prior
from .adstocks import Adstock, AdstockType
from .saturation import Saturation, SaturationType


@dataclass(slots=True)
class MediaTransformSpec:
    """Adstock and saturation configuration for one media channel."""

    adstock: AdstockType = "Geometric"
    adstock_params: dict[str, float] = field(
        default_factory=lambda: {"alpha": 0.5, "l_max": 12, "normalize": False}
    )
    adstock_priors: dict[str, PriorSpec] = field(default_factory=dict)

    saturation: SaturationType = "Logistic"
    saturation_params: dict[str, float] = field(default_factory=lambda: {"lam": 0.5})
    saturation_priors: dict[str, PriorSpec] = field(default_factory=dict)

    def __post_init__(self):
        overlap = set(self.adstock_params) & set(self.adstock_priors)
        if overlap:
            raise ValueError(
                "adstock_params and adstock_priors must not share keys: "
                f"{sorted(overlap)}"
            )

        overlap = set(self.saturation_params) & set(self.saturation_priors)
        if overlap:
            raise ValueError(
                "saturation_params and saturation_priors must not share keys: "
                f"{sorted(overlap)}"
            )


@dataclass(slots=True)
class MMMConfig:  # pylint: disable=too-many-instance-attributes
    """Main configuration object for the MMM model."""

    date_name: str = "date"
    media_names: list[str] = field(default_factory=list)
    control_names: list[str] = field(default_factory=list)
    include_intercept: bool = True
    seasonality_order: int = 0
    media_transforms: dict[str, MediaTransformSpec] = field(default_factory=dict)
    random_seed: int = 42
    umbrella_driver: str | None = None

    prior_umbrella: dict[str, PriorSpec] = field(default_factory=dict)
    prior_product_media: dict[str, PriorSpec] = field(default_factory=dict)
    prior_intercept: PriorSpec = field(
        default_factory=lambda: PriorSpec("Normal", {"mu": 0.0, "sigma": 2.0})
    )
    prior_media: PriorSpec = field(
        default_factory=lambda: PriorSpec("HalfNormal", {"sigma": 1.0})
    )
    prior_control: PriorSpec = field(
        default_factory=lambda: PriorSpec("Normal", {"mu": 0.0, "sigma": 1.0})
    )
    prior_sigma: PriorSpec = field(
        default_factory=lambda: PriorSpec("HalfNormal", {"sigma": 1.0})
    )
    prior_season: PriorSpec = field(
        default_factory=lambda: PriorSpec("Laplace", {"mu": 0.0, "b": 0.5})
    )

    def __post_init__(self):
        if self.seasonality_order < 0:
            raise ValueError("seasonality_order must be non-negative")
        if not set(self.media_transforms) <= set(self.media_names):
            not_in_media = set(self.media_transforms) - set(self.media_names)
            raise ValueError(
                "media_transforms keys must be a subset of media_names. "
                f"Got {not_in_media} not in {set(self.media_names)}"
            )

        if self.include_intercept and self.prior_intercept is None:
            raise ValueError(
                "prior_intercept must be specified if include_intercept is True"
            )

    @property
    def var_names(self) -> list[str]:
        """List all variable names in the model, including media and control parameters.

        Returns
        -------
        list[str]
            List of variable names in the model.
        """
        return [
            "intercept",
            "beta_media",
            "beta_control",
            *[
                f"adstock_alpha[{m}]"
                for m, t in self.media_transforms.items()
                if "alpha" in t.adstock_priors
            ],
            *[
                f"saturation_lam[{m}]"
                for m, t in self.media_transforms.items()
                if "lam" in t.saturation_priors
            ],
            *[
                f"saturation_k[{m}]"
                for m, t in self.media_transforms.items()
                if "k" in t.saturation_priors
            ],
            *[
                f"saturation_n[{m}]"
                for m, t in self.media_transforms.items()
                if "n" in t.saturation_priors
            ],
            "beta_season",
            "sigma",
            *[
                f"umbrella[{m}]"
                for m in self.media_transforms
                if m in self.prior_umbrella
            ],
        ]

    def var_names(self) -> list[str]:
        """List all variable names in the model, including media and control parameters.

        Returns
        -------
        list[str]
            List of variable names in the model.
        """
        return [
            "intercept",
            "beta_media",
            "beta_control",
            # "beta_trend",
            *[
                f"adstock_alpha[{m}]"
                for m, t in self.media_transforms.items()
                if "alpha" in t.adstock_priors
            ],
            *[
                f"saturation_lam[{m}]"
                for m, t in self.media_transforms.items()
                if "lam" in t.saturation_priors
            ],
            *[
                f"saturation_k[{m}]"
                for m, t in self.media_transforms.items()
                if "k" in t.saturation_priors
            ],
            *[
                f"saturation_n[{m}]"
                for m, t in self.media_transforms.items()
                if "n" in t.saturation_priors
            ],
            "beta_season",
            "sigma",
            *[
                f"umbrella[{m}]"
                for m in self.media_transforms
                if m in self.prior_umbrella
            ],
            *[f"product_media[{m}]" for m in self.prior_product_media],
        ]


class MMM:  # pylint: disable=too-many-instance-attributes
    """PyMC-based Bayesian MMM framework using the NumPyro NUTS backend."""

    def __init__(self, config: MMMConfig) -> None:
        self.config = config
        self.model: pm.Model | None = None
        self.idata: az.InferenceData | None = None
        self._X_media: ArrayLike | None = None  # pylint: disable=invalid-name
        self._X_control: ArrayLike | None = None  # pylint: disable=invalid-name
        self._y: ArrayLike | None = None
        self._season: ArrayLike | None = None
        self._scales: dict[str, np.ndarray | float] = {}

        self.adstocks = {}
        self.saturations = {}

        self.priors = {}

    def add_prior(self, name: str, prior_spec: PriorSpec):
        """Add a prior to the model if it doesn't already exist.

        Parameters
        ----------
        name : str
            The name of the prior.
        prior_spec : PriorSpec
            The specification of the prior.

        Returns
        -------
        pm.Distribution
            The prior distribution.
        """
        if name in self.priors:
            return self.priors[name]

        self.priors[name] = _make_prior(name, prior_spec)
        return self.priors[name]

    def scale(self, key: str):
        """Get the scaling factor for a given key.

        Parameters
        ----------
        key : str
            The key for which to retrieve the scaling factor.

        Returns
        -------
        float | np.ndarray
            The scaling factor for the specified key.
            If the scaling factor is a single value, it returns a float;
            otherwise, it returns a numpy array.
        """

        s = self._scales[key]
        return s

    def _apply_boosts(self, cols, x_c):
        """Apply umbrella and product media boosts to media channels.

        Parameters
        ----------
        cols : list[pt.TensorVariable]
            List of symbolic media columns after adstock and saturation transformations.
        control_contribution : pt.TensorVariable
            Symbolic control contribution matrix with shape (n_obs, n_controls).

        Returns
        -------
        list[pt.TensorVariable]
            List of symbolic media columns after applying boosts.
        """

        # tv x media interaction
        tv_hill = -1
        tv_idx = -1
        if self.config.umbrella_driver is not None:
            tv_idx = self.config.media_names.index(self.config.umbrella_driver)
            tv_hill = cols[tv_idx]

        # control x media interaction
        boost_controls = 0.0
        for name, pspec in self.config.prior_product_media.items():
            ctrl_idx = self.config.control_names.index(name)
            para_product = self.add_prior(f"product_media[{name}]", pspec)
            boost_controls = boost_controls + para_product * x_c.isel(control=ctrl_idx)

        for j, m in enumerate(self.config.media_names):
            boost = boost_controls
            if tv_idx not in (-1, j):
                pspec = self.config.prior_umbrella.get(m, None)
                if pspec is not None:
                    umbrella = self.add_prior(f"umbrella[{m}]", pspec)

                    boost = boost + umbrella * tv_hill

            if boost != 0.0:
                cols[j] = cols[j] * (1 + boost)

        return cols

    def _build_media_contribution(self, x_m, control_contribution):
        """Transform media channels with adstock operators.

        Parameters
        ----------
        x_m : np.ndarray | pt.TensorVariable
            Raw (or symbolic) media matrix with shape ``(n_obs, n_media)``.
            When a ``pm.Data`` tensor is passed the convolution graph stays
            symbolic so that ``pymc.do`` can later substitute it.

        Returns
        -------
        pt.TensorVariable
            Symbolic transformed media matrix.
        """
        cols = []
        x_m = ptx.as_xtensor(x_m.values, dims=("date", "media"))

        for j, name in enumerate(self.config.media_names):
            spec = self.config.media_transforms.get(name, MediaTransformSpec())
            col = x_m.isel(media=j)

            # === sample adstock stochastic params and apply adstock ===
            adstock_params: dict = dict(spec.adstock_params)
            for pname, pspec in spec.adstock_priors.items():
                adstock_params[pname] = self.add_prior(
                    f"adstock_{pname}[{name}]", pspec
                )

            ad = Adstock.from_spec(
                kind=spec.adstock,
                dim="date",
                l_max=spec.adstock_params.get("l_max"),
                normalize=spec.adstock_params.get("normalize"),
            )
            col = ad(col, params=adstock_params)

            # === apply saturation ===
            saturation_params: dict = dict(spec.saturation_params)
            for pname, pspec in spec.saturation_priors.items():
                saturation_params[pname] = self.add_prior(
                    f"saturation_{pname}[{name}]", pspec
                )
            sat = Saturation.from_spec(spec.saturation, params=saturation_params)
            col = sat(col)

            # === collect transformed column ===
            col = col.expand_dims(dim="media")
            cols.append(col)

            self.adstocks[name] = {"function": ad, "params": adstock_params}
            self.saturations[name] = sat

        cols = self._apply_boosts(cols, control_contribution)

        return ptx.concat(cols, dim="media").transpose("date", "media")

    def _process_data(self, X, y, rescale: bool = True):  # pylint: disable=invalid-name
        """Extract and scale model inputs.

        Parameters
        ----------
        X : pd.DataFrame
            Input design table.
        y : array-like
            Target series.

        Returns
        -------
        np.ndarray
            Date coordinate array.
        """

        X_media = X[self.config.media_names].to_numpy(dtype=np.float64)  # pylint: disable=invalid-name
        X_control = X[self.config.control_names].to_numpy(dtype=np.float64)  # pylint: disable=invalid-name

        if rescale:
            self._X_media, self._scales["media"] = max_abs_scaler(
                np.asarray(X_media, dtype=np.float64)
            )
            self._scales["media"] = dict(
                zip(self.config.media_names, self._scales["media"])
            )

            self._X_control, self._scales["control"] = max_abs_scaler(
                np.asarray(X_control, dtype=np.float64)
            )
            self._scales["control"] = dict(
                zip(self.config.control_names, self._scales["control"])
            )

            self._y, self._scales["y"] = max_abs_scaler(np.asarray(y, dtype=np.float64))
            # y_log = np.log1p(np.asarray(y, dtype=np.float64))
            # self._y = (y_log - y_log.mean()) / y_log.std()
            # self._scales["y"] = {"mean": y_log.mean(), "std": y_log.std()}
            self._scales["y"] = float(self._scales["y"][0])

        else:
            self._X_media = np.asarray(X_media, dtype=np.float64)
            self._X_control = np.asarray(X_control, dtype=np.float64)
            self._y = np.asarray(y, dtype=np.float64)
            self._scales = {"media": 1, "control": 1, "y": 1}

        date = X[self.config.date_name].to_numpy()

        return date

    def build(
        self,
        X: ArrayLike,
        y: ArrayLike,
        rescale: bool = True,
    ):  # pylint: disable=too-many-locals, invalid-name
        """Build the probabilistic MMM model.

        Parameters
        ----------
        X : ArrayLike
            Input feature table containing date, media, and controls.
        y : ArrayLike
            Observed target series.
        rescale : bool, optional
            Whether to rescale media, controls, and target by their max absolute value.
        """
        date = self._process_data(X, y, rescale=rescale)
        n = self._y.shape[0]

        # === Build seasonality features ===
        seas = fourier_features(n, order=self.config.seasonality_order)
        seas_name = sum(
            [
                [f"sin[{i + 1}]", f"cos[{i + 1}]"]
                for i in range(self.config.seasonality_order)
            ],
            [],
        )
        self._season = seas

        coords = {
            "date": date,
            "media": self.config.media_names,
            "control": self.config.control_names,
            "season": seas_name,
        }

        # for (
        #     prior_name,
        #     priors_specs,
        # ) in self.config.get_list_media_with_priors().items():
        #     coords[f"media_{prior_name}"] = list(priors_specs.keys())

        with pm.Model(coords=coords) as self.model:
            # Register all data nodes as pm.Data so they can be swapped via
            # pymc.do() for counterfactual / optimisation scenarios.
            x_m = pmd.Data("channel_data", self._X_media, dims=("date", "media"))
            x_c = pmd.Data("control_data", self._X_control, dims=("date", "control"))
            x_s = pmd.Data("season_data", self._season, dims=("date", "season"))
            y_o = pmd.Data("y_obs", self._y, dims="date")

            # === CONTROL ===
            beta_control = _make_prior(
                "beta_control", self.config.prior_control, dims="control"
            )
            control_contribution = pmd.Deterministic(
                "control_contribution",
                value=x_c * beta_control,
                dims=["date", "control"],
            )
            mu = control_contribution.sum(dim="control")

            # trend_idx = self.config.control_names.index("trend")
            # trend_contribution = control_contribution.isel(control=trend_idx)
            # pspec = self.config.prior_product_media["cospirit"]
            # mu = (
            #     mu
            #     - trend_contribution
            #     + trend_contribution * self.add_prior(f"product_media[cospirit]", pspec)
            # )

            # === MEDIA ===
            beta_media = _make_prior(
                "beta_media", self.config.prior_media, dims="media"
            )
            # stochastic media transform if any media has adstock/saturation params with priors
            x_m_transformed = self._build_media_contribution(x_m, x_c)

            media_contribution = pm.Deterministic(
                "media_contribution",
                var=x_m_transformed * beta_media,  # [None, :],
                dims=["date", "media"],
            )
            mu = mu + media_contribution.sum(dim="media")

            # === INTERCEPT ===
            intercept = (
                _make_prior("intercept", self.config.prior_intercept)
                if self.config.include_intercept
                else 0.0
            )
            intercept_contribution = pm.Deterministic(
                "intercept_contribution", var=intercept
            )
            mu = mu + intercept_contribution

            # === SEASONALITY ===
            beta_season = _make_prior(
                "beta_season", self.config.prior_season, dims="season"
            )
            yearly_seasonality = pmd.Deterministic(
                "yearly_seasonality_contribution",
                value=ptx.math.dot(x_s, beta_season),
                dims="date",
            )
            mu = mu + yearly_seasonality

            # === LIKELIHOOD ===
            # if self.config.likelihood == "gaussian":
            sigma = _make_prior("sigma", self.config.prior_sigma)
            pmd.Normal("y", mu=mu, sigma=sigma, observed=y_o, dims="date")
            # elif self.config.likelihood == "student_t":
            #     sigma = _make_prior("sigma", self.config.prior_sigma)
            #     nu = pm.Exponential("nu", 1 / 30)
            #     pm.StudentT("y", mu=mu, sigma=sigma, nu=nu, observed=y_o, dims="date")
            # else:
            #     sigma = _make_prior("sigma", self.config.prior_sigma)
            #     pm.LogNormal(
            #         "y",
            #         mu=mu,
            #         sigma=sigma,
            #         observed=np.clip(y_o, 1e-12, np.inf),
            #         dims="date",
            #     )

            _ = pmd.Deterministic(
                "total_media_contribution",
                value=media_contribution.sum(dim="media"),
                dims="date",
            )

    def fit(  # pylint: disable=too-many-arguments
        self,
        *,
        draws: int = 1000,
        tune: int = 1000,
        chains: int = 2,
        cores: int = 2,
        target_accept: float = 0.9,
    ) -> None:
        """Run posterior sampling for the built model.

        Parameters
        ----------
        draws : int, optional
            Number of posterior draws.
        tune : int, optional
            Number of tuning steps.
        chains : int, optional
            Number of MCMC chains.
        cores : int, optional
            Number of CPU cores to use.
        target_accept : float, optional
            Target acceptance probability for NUTS.
        """

        with self.model:
            self.idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                target_accept=target_accept,
                random_seed=self.config.random_seed,
                nuts_sampler="numpyro",
                return_inferencedata=True,
                idata_kwargs={"log_likelihood": True},
            )

            n_diverging = self.idata["sample_stats"]["diverging"].sum().item()
            if n_diverging > 0:
                warnings.warn(
                    f"Divergences detected in sampling: {n_diverging}!"
                    " Consider increasing target_accept or reparameterizing."
                )

    def sample_posterior_predictive(self, samples: int = 1000) -> None:
        """Sample posterior predictive distribution.

        Parameters
        ----------
        samples : int, optional
            Kept for API compatibility.

        Raises
        ------
        RuntimeError
             If called before fitting the model.
        """
        if self.model is None or self.idata is None:
            raise RuntimeError("Call fit() before sample_posterior_predictive().")
        with self.model:
            pm.sample_posterior_predictive(
                trace=self.idata,
                # var_names=["y"],
                random_seed=self.config.random_seed,
                extend_inferencedata=True,
            )

            prior = pm.sample_prior_predictive(
                random_seed=self.config.random_seed,
                return_inferencedata=True,
                draws=samples,
            )

        self.idata.update(prior)

    def sample_saturation_curves(self, x_max=2.0):
        """Sample saturation curves for each media channel.

        Parameters
        ----------
        x_max : float, optional
            Maximum value of the input range for the saturation curves, by default 2.0.

        Returns
        -------
        dict
            Dictionary mapping each media channel name to its sampled saturation curve.
        """
        x = np.linspace(0, x_max, 200)
        curves = {}
        for _, m in enumerate(self.config.media_names):
            curves[m] = self.saturations[m].sample_saturation_curve(self, x)

        return curves

    def compute_contributions(self) -> pd.DataFrame:
        """Compute posterior mean contributions by component.

        Returns
        -------
        pd.DataFrame
            Contribution table with media and optional seasonal/intercept terms.

        Raises
        ------
        RuntimeError
            If called before fitting the model.
        """
        if self.idata is None or self._X_media is None:
            raise RuntimeError("Call fit() before compute_contributions().")
        post = self.idata.posterior
        b_media = post["beta_media"].mean(("chain", "draw")).values
        contrib_media = self._X_media * b_media[None, :]
        out = pd.DataFrame(
            contrib_media, columns=[f"contrib_{m}" for m in self.config.media_names]
        )
        if "intercept" in post:
            out["intercept"] = float(post["intercept"].mean().values)
        if (
            self._season is not None
            and self._season.shape[1] > 0
            and "beta_season" in post
        ):
            b_s = post["beta_season"].mean(("chain", "draw")).values
            out["seasonality"] = self._season @ b_s
        out["total_media"] = out[
            [c for c in out.columns if c.startswith("contrib_")]
        ].sum(axis=1)
        return out
