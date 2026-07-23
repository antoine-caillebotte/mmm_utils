"""This module implements the MMM class,
a PyMC-based Bayesian Media Mix Modeling framework
with adstock and saturation transformations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import tempfile
import warnings
import zipfile

import cloudpickle
import numpy as np
import pandas as pd
import arviz as az


import pymc as pm
import pymc.dims as pmd
import pytensor.xtensor as ptx


from .utils import ArrayLike, max_abs_scaler
from .seasonality import fourier_features
from .prior import _make_prior
from .transform_handler import TransformHandler
from .model_definition.mmm_config import MMMConfig


@dataclass
class MMMDataHandler:
    """Data handler for the MMM model."""

    date: ArrayLike | None = None
    X_media: ArrayLike | None = None  # pylint: disable=invalid-name
    X_control: ArrayLike | None = None  # pylint: disable=invalid-name
    X_interactions: ArrayLike | None = None  # pylint: disable=invalid-name
    y: ArrayLike | None = None
    season: ArrayLike | None = None

    _scales: dict[str, float | dict[str, float]] = field(default_factory=dict)

    def build_pm_data(self, model: pm.Model):
        """Build PyMC data containers from processed inputs.

        Parameters
        ----------
        model : Model
            The PyMC model instance to which data containers are attached.

        Returns
        -------
        tuple
            A tuple ``(x_m, x_c, x_s, y_o)`` containing PyMC data containers
            for media, control, seasonality, and observed target data.
        """

        x_m = pmd.Data(
            "channel_data",
            self.X_media,
            dims=("date", "media"),
            model=model,
        )
        x_c = pmd.Data(
            "control_data",
            self.X_control,
            dims=("date", "control"),
            model=model,
        )
        x_s = pmd.Data(
            "season_data",
            self.season,
            dims=("date", "season"),
            model=model,
        )
        y_o = pmd.Data("y_obs", self.y, dims="date", model=model)

        return x_m, x_c, x_s, y_o

    def build_seasonality(self, order: int) -> list[str]:
        """Build seasonality features using Fourier series.

        Parameters
        ----------
        order : int
            The order of the Fourier series.

        Returns
        -------
        list[str]
            The names of the seasonality features.
        """
        n = self.y.shape[0]

        # === Build seasonality features ===
        seas = fourier_features(n, order=order)
        seas_name = sum(
            [[f"sin[{i + 1}]", f"cos[{i + 1}]"] for i in range(order)],
            [],
        )
        self.season = seas

        return seas_name

    def process_data(self, X, y, config: MMMConfig, rescale: bool = True):  # pylint: disable=invalid-name
        """Extract and scale model inputs.

        Parameters
        ----------
        X : pd.DataFrame
            Input design table.
        y : pd.Series
            Observed target series.
        config : MMMConfig
            The MMM configuration object.
        rescale : bool, optional
            Whether to rescale media, controls, and target by their max absolute value.
        """

        x_media = X[config.media_names]
        x_control = X[config.control_names]

        if rescale:
            self.X_media, self._scales["media"] = max_abs_scaler(x_media)
            self._scales["media"] = dict(zip(config.media_names, self._scales["media"]))

            self.X_control, self._scales["control"] = max_abs_scaler(x_control)
            self._scales["control"] = dict(
                zip(config.control_names, self._scales["control"])
            )

            self.y, self._scales["y"] = max_abs_scaler(y)
            self.y, self._scales["y"] = max_abs_scaler(y)
            self._scales["y"] = float(self._scales["y"][0])

        else:
            self.X_media = np.asarray(x_media, dtype=np.float64)
            self.X_control = np.asarray(x_control, dtype=np.float64)
            self.y = np.asarray(y, dtype=np.float64)
            self._scales = {
                "media": {m: 1 for m in config.media_names},
                "control": {c: 1 for c in config.control_names},
                "y": 1,
            }

        self.date = X[config.date_name].to_numpy()

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


class MMM:  # pylint: disable=too-many-instance-attributes
    """PyMC-based Bayesian MMM framework using the NumPyro NUTS backend."""

    def __init__(self, config: MMMConfig) -> None:
        self.config = config
        self.model: pm.Model | None = None
        self.idata: az.InferenceData | None = None
        self.data = MMMDataHandler()
        self.adstocks = {}
        self.saturations = {}

    def _apply_media_transforms(self, x_m):
        """Transform media channels with adstock and saturation operators.

        Delegates to :class:`~.transform.TransformHandler` which implements
        a vectorized fast path (uniform specs) and a per-channel fallback
        (heterogeneous specs).  Must be called inside an active ``pm.Model``
        context.

        Parameters
        ----------
        x_m : XTensorVariable
            Raw media data with dims ``("date", "media")``.

        Returns
        -------
        XTensorVariable
            Symbolically transformed media matrix, dims ``("date", "media")``.
        """
        handler = TransformHandler(
            media_names=self.config.media_names,
            media_transforms=self.config.media_transforms,
        )
        x_transformed = handler.apply(x_m)
        self.adstocks = handler.adstocks
        self.saturations = handler.saturations
        return x_transformed

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
        self.data.process_data(X, y, config=self.config, rescale=rescale)
        seas_name = self.data.build_seasonality(order=self.config.seasonality_order)

        coords = {
            "date": self.data.date,
            "media": self.config.media_names,
            "control": self.config.control_names,
            "season": seas_name,
        } | self.config.beta_priors.interaction.get_coords()

        with pm.Model(coords=coords) as self.model:
            # Register all data nodes as pm.Data so they can be swapped via
            # pymc.do() for counterfactual / optimisation scenarios.
            x_m, x_c, x_s, y_o = self.data.build_pm_data(self.model)
            self.config.beta_priors.build_pymc_priors()

            # === MEDIA TRANSFORMATION ===
            # Stochastic adstock/saturation params are created as vectorized
            # PyMC variables inside _apply_media_transforms using transform_coords.
            x_m_transformed = self._apply_media_transforms(x_m)
            x_m_transformed = pmd.Deterministic(
                "media_transformed", value=x_m_transformed, dims=("date", "media")
            )

            # === BETA (with interaction adjustments) ===
            beta_adjusted = self.config.beta_priors.get_beta_adjusted(
                x_m_transformed, x_c
            )

            # === MEDIA ===
            media_contribution = pmd.Deterministic(
                "media_contribution",
                value=x_m_transformed * beta_adjusted["media"],
                dims=["date", "media"],
            )
            total_media_contribution = pmd.Deterministic(
                "total_media_contribution",
                value=media_contribution.sum(dim="media"),
                dims="date",
            )

            mu = total_media_contribution

            # === CONTROL ===
            control_contribution = pmd.Deterministic(
                "control_contribution",
                value=x_c * beta_adjusted["control"],
                dims=["date", "control"],
            )

            mu = mu + control_contribution.sum(dim="control")

            # === SEASONALITY ===
            yearly_seasonality = pmd.Deterministic(
                "yearly_seasonality_contribution",
                value=ptx.math.dot(x_s, beta_adjusted["season"]),
                dims="date",
            )

            mu = mu + yearly_seasonality

            # === LIKELIHOOD ===
            sigma = _make_prior("sigma", self.config.prior_sigma)
            pmd.Normal("y", mu=mu, sigma=sigma, observed=y_o, dims="date")

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

        Raises
        ------
        RuntimeError
            If called before building the model with :meth:`build`.
        """
        if self.model is None:
            raise RuntimeError("Call build() before fit().")

        with self.model:
            self.idata = pm.sample(
                draws=draws,
                var_names=self.config.var_names,
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

    def save(self, path: str | Path) -> None:
        """Persist the full MMM object to a single file.

        The posterior ``idata`` is stored via Arviz's netCDF format (the
        robust, long-term storage format for :class:`arviz.InferenceData`),
        while everything else (config, PyMC model graph, processed data,
        adstocks, saturations) is serialized with ``cloudpickle``, which —
        unlike the stdlib ``pickle`` — can also handle the closures and
        PyTensor graph objects held by the model. Both parts are bundled
        into one zip archive so a fitted model can be reloaded intact.

        Parameters
        ----------
        path : str | Path
            Destination file path (created or overwritten).
        """
        state = {k: v for k, v in self.__dict__.items() if k != "idata"}

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            state_path = tmp_dir / "state.pkl"
            with open(state_path, "wb") as f:
                cloudpickle.dump(state, f)

            with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.write(state_path, "state.pkl")
                if self.idata is not None:
                    idata_path = tmp_dir / "idata.nc"
                    self.idata.to_netcdf(idata_path)
                    zf.write(idata_path, "idata.nc")

    @classmethod
    def load(cls, path: str | Path) -> "MMM":
        """Reload an MMM object previously saved with :meth:`save`.

        Parameters
        ----------
        path : str | Path
            Path to the file written by :meth:`save`.

        Returns
        -------
        MMM
            The restored object, including the fitted model, data, and
            posterior samples (if the model had been fit).
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            with zipfile.ZipFile(path) as zf:
                zf.extractall(tmp_dir)

            with open(tmp_dir / "state.pkl", "rb") as f:
                state = cloudpickle.load(f)

            obj = cls(state["config"])
            obj.__dict__.update(state)

            idata_path = tmp_dir / "idata.nc"
            obj.idata = az.from_netcdf(idata_path) if idata_path.exists() else None

        return obj

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
        tuple
            A tuple containing:
            - curves: dict[str, XArray] - Saturation curves for each media channel.
            - saturation: list[dict] - Saturation values for each media channel.
        """
        x = np.linspace(0, x_max, 200)
        curves = {}
        for _, m in enumerate(self.config.media_names):
            curves[m] = self.saturations[m].sample_saturation_curve(self, x)

        media_scales = self.data.scale("media")
        saturation = []
        for m in self.config.media_names:
            curve = curves[m]
            xx = (curve.coords["x"]).values * media_scales[m]
            beta = (
                self.idata.posterior["beta_media"]
                .sel(media=m)
                .mean(dim=["chain", "draw"])
                .values
            )
            yy = beta * curve.mean(dim=["chain", "draw"]).values * self.data.scale("y")

            saturation.append(
                {
                    "name": m,
                    "values": {str(int(xx[k])): float(yy[k]) for k in range(len(xx))},
                }
            )

        return curves, saturation

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
        if self.idata is None or self.data.X_media is None or self.model is None:
            raise RuntimeError("Call fit() before compute_contributions().")

        with self.model:
            self.idata.posterior = pm.compute_deterministics(
                self.idata.posterior,
                merge_dataset=True,
                var_names=list(self.config.expressions_to_compute),
            )

        return self.idata
