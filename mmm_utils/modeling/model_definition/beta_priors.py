"""
Beta priors module for MMM models.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytensor.xtensor as ptx

from .formulae import Interaction
from ..prior import PriorSpec, _make_prior


@dataclass
class BetaPriors:  # pylint: disable=too-many-instance-attributes
    """All prior specifications for an MMM model.

    Holds both the fixed structural priors (intercept, media coefficients,
    controls, noise, seasonality) and the interaction-term priors.  Each
    unique interaction parameter produced by the ``Interaction`` configuration
    (e.g. ``"beta_Digital,SEA:TV"``) must have exactly one entry in
    ``priors``.  ``BetaPriors`` validates this one-to-one coverage at
    construction time and exposes helpers for look-up and PyMC variable
    creation.

    Parameters
    ----------
    interaction : Interaction
        The interaction configuration this set of priors belongs to.
        Determines the *expected* interaction parameter names via
        :meth:`~.formulae.Interaction.get_unique_parameter_names`.
    priors : dict[str, PriorSpec], optional
        Mapping ``{parameter_name: prior_spec}`` covering every unique
        interaction parameter produced by ``interaction``.  May be empty
        only when the interaction itself has no terms.
    prior_intercept : PriorSpec, optional
        Prior for the model intercept.
        Default: ``Normal(mu=0, sigma=2)``.
    prior_media : PriorSpec, optional
        Shared prior for all media baseline coefficients.
        Default: ``HalfNormal(sigma=1)``.
    prior_control : PriorSpec, optional
        Shared prior for all control-variable coefficients.
        Default: ``Normal(mu=0, sigma=1)``.
    prior_sigma : PriorSpec, optional
        Prior for the observation noise standard deviation.
        Default: ``HalfNormal(sigma=1)``.
    prior_season : PriorSpec, optional
        Prior for seasonality coefficients.
        Default: ``Laplace(mu=0, b=0.5)``.

    Attributes
    ----------
    is_well_specified : bool
        ``True`` when every expected interaction parameter has a prior and
        no extra keys are present.  Set automatically during construction.

    Raises
    ------
    ValueError
        If any expected interaction parameter name is missing from
        ``priors``, or if ``priors`` contains keys that do not correspond
        to any known interaction parameter.

    Examples
    --------
    >>> from mmm_utils.modeling.model_definition.formulae import Interaction
    >>> from mmm_utils.modeling.prior import PriorSpec
    >>> ia = Interaction(
    ...     formulas={"TV": "1", "SEA": "1 + TV", "Digital": "1 + TV"},
    ...     is_shared_with=[("TV", "Digital", "SEA")],
    ... )
    >>> bp = BetaPriors(
    ...     interaction=ia,
    ...     priors={"beta_Digital,SEA:TV": PriorSpec("HalfNormal", {"sigma": 1.0})},
    ... )
    >>> bp.is_well_specified
    True
    >>> bp.media
    PriorSpec(kind='HalfNormal', params={'sigma': 1.0})
    """

    interaction: Interaction = field(default_factory=Interaction)
    priors: dict[str, PriorSpec] = field(default_factory=dict)
    pymc_priors: dict[str, object] = field(
        init=False, default_factory=dict, repr=False, compare=False
    )
    coords: dict[str, list[str]] = field(
        init=False, default_factory=dict, repr=False, compare=False
    )

    media: PriorSpec = field(
        default_factory=lambda: PriorSpec("HalfNormal", {"sigma": 1.0})
    )
    control: PriorSpec = field(
        default_factory=lambda: PriorSpec("Normal", {"mu": 0.0, "sigma": 1.0})
    )
    season: PriorSpec = field(
        default_factory=lambda: PriorSpec("Laplace", {"mu": 0.0, "b": 0.5})
    )

    is_well_specified: bool = field(init=False, repr=True, compare=False)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        self.is_well_specified = False
        self.check_priors()

        # self.coords = self.interaction.get_coords() | {
        #     "season": "season",
        #     "media": "media",
        #     "control": "control",
        #     "sigma": None,
        # }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def check_priors(self) -> None:
        """Validate that ``priors`` covers exactly the expected parameter set.

        The expected set is ``interaction.get_unique_parameter_names()``.
        Both missing priors and spurious extra entries raise ``ValueError``.

        Raises
        ------
        ValueError
            If any expected parameter lacks a prior, or if any key in
            ``priors`` does not correspond to a known interaction parameter.
        """
        expected = self.interaction.get_unique_parameter_names()

        if not expected:
            self.is_well_specified = True
            return

        if not self.priors:
            self.is_well_specified = False
            raise ValueError(
                f"No priors specified. Expected priors for: {sorted(expected)}"
            )

        unknown = set(self.priors) - expected
        if unknown:
            self.is_well_specified = False
            raise ValueError(
                f"Prior specified for unknown parameter(s): {sorted(unknown)}. "
                f"Known parameters: {sorted(expected)}"
            )

        missing = expected - set(self.priors)
        if missing:
            self.is_well_specified = False
            raise ValueError(f"Missing prior for parameter(s): {sorted(missing)}.")

        self.is_well_specified = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def missing_priors(self) -> set[str]:
        """Return parameter names that lack a prior entry.

        Useful for diagnosing an incomplete ``priors`` dict before
        constructing a ``BetaPriors`` instance (or after catching a
        ``ValueError``).

        Returns
        -------
        set[str]
            Names in ``interaction.get_unique_parameter_names()`` not
            present in ``priors``.
        """
        return self.interaction.get_unique_parameter_names() - set(self.priors)

    def extra_priors(self) -> set[str]:
        """Return prior keys that do not correspond to any known parameter.

        Returns
        -------
        set[str]
            Keys in ``priors`` not found in
            ``interaction.get_unique_parameter_names()``.
        """
        return set(self.priors) - self.interaction.get_unique_parameter_names()

    def build_pymc_priors(self) -> None:
        """Build and register all PyMC prior variables for model coefficients.

        Must be called inside an active ``pm.Model()`` context.

        This method populates :attr:`pymc_priors` with:

        - one prior per interaction coefficient in ``self.priors``
        - ``beta_media`` for media channels
        - ``beta_control`` for control variables
        - ``beta_season`` for seasonal components
        """
        # Each key is "beta_interaction_{var}" → vectorized prior with dim "interaction_{var}"
        self.pymc_priors = {
            name: _make_prior(
                name, spec, f"interaction_{name[len('beta_interaction_') :]}"
            )
            for name, spec in self.priors.items()
        }

        self.pymc_priors["beta_media"] = _make_prior("beta_media", self.media, "media")
        self.pymc_priors["beta_control"] = _make_prior(
            "beta_control", self.control, "control"
        )
        self.pymc_priors["beta_season"] = _make_prior(
            "beta_season", self.season, "season"
        )

    def get_beta_adjusted(self, x_m, x_c) -> dict[str, ptx.XTensorVariable]:  # pylint: disable=too-many-locals
        """Compute interaction-adjusted beta coefficients for all model components.

        For each media channel *m* with interaction formula, the effective
        (time-varying) beta is:

        * formula ``"1 + Z1 + Z2"`` →
          ``β_m · (1 + β_{m:Z1}·x_Z1 + β_{m:Z2}·x_Z2)``
        * formula ``"0 + Z1 + Z2"`` →
          ``β_m · (β_{m:Z1}·x_Z1 + β_{m:Z2}·x_Z2)``
        * formula ``"1"`` (no interaction) →
          ``β_m``

        where ``x_Zi`` is looked up in *x_m* when *Zi* is a media channel,
        or in *x_c* when *Zi* is a control.

        Must be called inside an active ``pm.Model`` context, after
        :meth:`build_pymc_priors`.

        Parameters
        ----------
        x_m : XTensorVariable
            Media data with dims ``("date", "media")``.
        x_c : XTensorVariable
            Control data with dims ``("date", "control")``.

        Returns
        -------
        dict[str, ptx.XTensorVariable]
            * **beta_adjusted_media** — dims ``("date", "media")``, one
              effective beta per channel and time step.
            * **beta_season** — season coefficients, dim ``"season"``
              (unchanged).
            * **beta_control** — control coefficients, dim ``"control"``
              (unchanged).

        Raises
        ------
        RuntimeError
            If called before :meth:`build_pymc_priors`.

        Examples
        --------
        >>> beta_adj, beta_s, beta_c = bp.get_beta_adjusted(x_m, x_c)
        >>> media_contribution = x_m_transformed * beta_adj
        """
        if not self.pymc_priors:
            raise RuntimeError("Call build_pymc_priors() before get_beta_adjusted().")

        beta_media = self.pymc_priors["beta_media"]  # dim: "media"
        beta_season = self.pymc_priors["beta_season"]  # dim: "season"
        beta_control = self.pymc_priors["beta_control"]  # dim: "control"

        media_names = self.interaction.media
        control_names = self.interaction.controls

        cols = []
        for j, m in enumerate(media_names):
            formula = self.interaction.parse_formula(m)
            beta_m = beta_media.isel(media=j)  # 0-d: "media" dim removed

            boost = float(formula.has_baseline)  # 1.0 or 0.0
            for term in formula.terms:
                # "beta_interaction_{term}" is a vector; slice the scalar for this channel
                param_vector = self.pymc_priors[f"beta_interaction_{term}"]
                idx = self.interaction.get_lhs_index(m, term)
                param_value = param_vector.isel(**{f"interaction_{term}": idx})

                if term in self.interaction.media:
                    x_term = x_m.isel(media=media_names.index(term))
                else:
                    x_term = x_c.isel(control=control_names.index(term))

                boost = boost + param_value * x_term

            col = (beta_m * boost).expand_dims(dim="media")
            cols.append(col)

        beta_adjusted_media = ptx.concat(cols, dim="media").transpose("date", "media")

        return {
            "media": beta_adjusted_media,
            "season": beta_season,
            "control": beta_control,
        }
