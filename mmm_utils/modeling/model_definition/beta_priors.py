"""
Beta priors module for MMM models.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

import pytensor.tensor as pt
from pytensor.graph.replace import graph_replace
from pytensor.xtensor.type import as_xtensor
import pymc as pm
import pymc.dims as pmd

from .formulae import Interaction, InteractionCoordinates
from .roas_priors import RoasPriors
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
    controls_without_effect : list[str], optional
        Control variables that must remain available as data (present in
        ``control_names``/``x_c``, usable as interaction terms) but that
        should **not** receive their own additive ``beta_control``
        coefficient. Typically used for a control that only makes sense as
        a modulator of a media channel (e.g. ``"Promo:boost"``) and should
        not also be added as a standalone regressor. Must be a subset of
        ``interaction.controls``.
    prior_intercept : PriorSpec, optional
        Prior for the model intercept.
        Default: ``Normal(mu=0, sigma=2)``.
    prior_media : PriorSpec, optional
        Shared prior for all media baseline coefficients.
        Default: ``HalfNormal(sigma=1)``.
    control : PriorSpec, optional
        Prior for control-variable coefficients (dim ``"control_active"``,
        see :meth:`get_control_own_effect_names`). Array-valued params must
        have length ``len(get_control_own_effect_names())`` and be ordered
        the same way — not ``len(interaction.controls)``.
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
    ...     priors={"beta_interaction_TV": PriorSpec("HalfNormal", {"sigma": 1.0})},
    ... )
    >>> bp.is_well_specified
    True
    >>> bp.media
    PriorSpec(kind='HalfNormal', params={'sigma': 1.0})
    """

    interaction: Interaction = field(default_factory=Interaction)
    priors: dict[str, PriorSpec] = field(default_factory=dict)
    controls_without_effect: list[str] = field(default_factory=list)
    pymc_priors: dict[str, object] = field(
        init=False, default_factory=dict, repr=False, compare=False
    )
    coords: dict[str, list[str]] = field(
        init=False, default_factory=dict, repr=False, compare=False
    )
    _interaction_coords: InteractionCoordinates = field(
        init=False, repr=False, compare=False
    )
    # Media split set by build_pymc_priors() when ROAS priors are used:
    # coefficients of _media_roas are derived from roas_media, _media_free keep
    # the ``media`` prior (beta_media_free).  Both empty without ROAS priors.
    _media_roas: list[str] = field(
        init=False, default_factory=list, repr=False, compare=False
    )
    _media_free: list[str] = field(
        init=False, default_factory=list, repr=False, compare=False
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
    expressions_to_compute: list[str] = field(
        init=False, repr=False, compare=False, default_factory=list
    )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        self._interaction_coords = InteractionCoordinates(self.interaction)
        self.is_well_specified = False
        self.check_priors()
        self._validate_controls_without_effect()
        self._validate_control_prior_shape()

        # self.coords = self._interaction_coords.get_coords() | {
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
        expected = self._interaction_coords.get_unique_parameter_names()

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

    def _validate_controls_without_effect(self) -> None:
        """Validate ``controls_without_effect`` against the interaction config.

        Raises
        ------
        ValueError
            If any entry is not a known control variable (i.e. not in
            ``interaction.controls``).

        Warns
        -----
        UserWarning
            If an entry is excluded from its own effect but is never
            referenced as an interaction term — it would then have no
            effect on the model at all.
        """
        unknown = set(self.controls_without_effect) - set(self.interaction.controls)
        if unknown:
            raise ValueError(
                f"controls_without_effect contains unknown control(s): "
                f"{sorted(unknown)}. Known controls: {sorted(self.interaction.controls)}"
            )

        inert = [
            c
            for c in self.controls_without_effect
            if c not in self.interaction.get_all_interaction_terms()
        ]
        if inert:
            warnings.warn(
                f"Control(s) {sorted(inert)} are excluded from their own effect "
                "(controls_without_effect) but are not referenced in any "
                "interaction formula — they will have no effect on the model at all.",
                UserWarning,
                stacklevel=3,
            )

    def _validate_control_prior_shape(self) -> None:
        """Check that array-valued ``control`` prior params match ``control_active``.

        ``beta_control`` is built with dim ``"control_active"``
        (:meth:`get_control_own_effect_names`), not ``"control"``
        (``interaction.controls``). Any array-valued entry in
        ``control.params`` must therefore have exactly
        ``len(get_control_own_effect_names())`` elements, in that same
        order — otherwise PyMC silently takes the array's own length for
        ``beta_control``, which then misaligns with (or conflicts against)
        the ``"control_active"`` coordinate registered on the model.

        Raises
        ------
        ValueError
            If an array-valued parameter's length does not match
            ``len(get_control_own_effect_names())``.
        """
        own_effect_names = self.get_control_own_effect_names()
        for param_name, value in self.control.params.items():
            if isinstance(value, np.ndarray) and len(value) != len(own_effect_names):
                raise ValueError(
                    f"control.params['{param_name}'] has length {len(value)}, but "
                    f"'beta_control' uses dim 'control_active' of length "
                    f"{len(own_effect_names)} ({own_effect_names}). Size and order "
                    "must match get_control_own_effect_names(), not "
                    "interaction.controls."
                )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_control_own_effect_names(self) -> list[str]:
        """Return control names that keep their own additive ``beta_control`` term.

        Returns
        -------
        list[str]
            ``interaction.controls`` minus ``controls_without_effect``, in
            the original order. This is the coordinate list backing the
            ``"control_active"`` dimension.

        Examples
        --------
        >>> ia = Interaction(
        ...     formulas={"Y1": "1 + Promo:boost"},
        ...     media=["Y1"],
        ...     controls=["Promo", "trend"],
        ... )
        >>> bp = BetaPriors(
        ...     interaction=ia,
        ...     priors={"beta_interaction_Promo": PriorSpec("HalfNormal", {"sigma": 1.0})},
        ...     controls_without_effect=["Promo"],
        ... )
        >>> bp.get_control_own_effect_names()
        ['trend']
        """
        excluded = set(self.controls_without_effect)
        return [c for c in self.interaction.controls if c not in excluded]

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
        return self._interaction_coords.get_unique_parameter_names() - set(self.priors)

    def extra_priors(self) -> set[str]:
        """Return prior keys that do not correspond to any known parameter.

        Returns
        -------
        set[str]
            Keys in ``priors`` not found in
            ``interaction.get_unique_parameter_names()``.
        """
        return set(self.priors) - self._interaction_coords.get_unique_parameter_names()

    def build_pymc_priors(self, roas_priors: RoasPriors | None = None) -> None:
        """Build and register all PyMC prior variables for model coefficients.

        Must be called inside an active ``pm.Model()`` context.

        This method populates :attr:`pymc_priors` with:

        - one prior per interaction coefficient in ``self.priors``
        - ``beta_media`` for media channels, or — when ``roas_priors`` covers
          some channels — ``roas_media`` (dim ``"media_roas"``) and
          ``beta_media_free`` (dim ``"media_free"``, only if some channels
          have no ROAS prior); ``beta_media`` is then assembled later by
          :meth:`get_beta_adjusted`
        - ``beta_control`` for control variables with their own effect
          (dim ``"control_active"``, see :meth:`get_control_own_effect_names`)
        - ``beta_season`` for seasonal components

        Parameters
        ----------
        roas_priors : RoasPriors | None, optional
            ROAS priors for a subset of media channels.  ``None`` or an empty
            :class:`RoasPriors` keeps the plain ``beta_media`` prior.
        """
        # Each key is "beta_interaction_{var}" → vectorized prior with dim "interaction_{var}"
        self.pymc_priors = {
            name: _make_prior(
                name, spec, f"interaction_{name[len('beta_interaction_') :]}"
            )
            for name, spec in self.priors.items()
        }

        roas_priors = roas_priors if roas_priors is not None else RoasPriors()
        self._media_roas = [
            m for m in self.interaction.media if m in roas_priors.priors
        ]
        self._media_free = [
            m for m in self.interaction.media if m not in roas_priors.priors
        ]
        if self._media_roas:
            self._build_roas_media_priors(roas_priors)
        else:
            self.pymc_priors["beta_media"] = _make_prior(
                "beta_media", self.media, "media"
            )

        self.pymc_priors["beta_control"] = _make_prior(
            "beta_control", self.control, "control_active"
        )
        self.pymc_priors["beta_season"] = _make_prior(
            "beta_season", self.season, "season"
        )

    def _build_roas_media_priors(self, roas_priors: RoasPriors) -> None:
        """Create ``roas_media`` and ``beta_media_free`` for a ROAS-split media set.

        Must be called inside an active ``pm.Model()`` context, after
        ``_media_roas`` / ``_media_free`` have been set.

        Parameters
        ----------
        roas_priors : RoasPriors
            ROAS priors covering every channel of ``_media_roas``.

        Raises
        ------
        ValueError
            If ``roas_priors`` targets a channel absent from
            ``interaction.media``.
        """
        unknown = [m for m in roas_priors.media_names if m not in self._media_roas]
        if unknown:
            raise ValueError(
                f"roas_priors contains channels {unknown} that are not in "
                "interaction.media"
            )

        model = pm.modelcontext(None)
        model.add_coords({"media_roas": self._media_roas})
        self.pymc_priors["roas_media"] = _make_prior(
            "roas_media", roas_priors.to_prior_spec(self._media_roas), "media_roas"
        )

        if self._media_free:
            model.add_coords({"media_free": self._media_free})
            self.pymc_priors["beta_media_free"] = _make_prior(
                "beta_media_free", self.media_spec_for(self._media_free), "media_free"
            )

    def media_spec_for(self, names: list[str]) -> PriorSpec:
        """Restrict the ``media`` prior spec to a subset of channels.

        Scalar hyperparameters are kept as is; per-channel arrays (one entry
        per ``interaction.media`` channel) are sliced to ``names``.

        Parameters
        ----------
        names : list[str]
            Channels to keep, in the desired order.

        Returns
        -------
        PriorSpec
            ``media`` prior restricted to ``names``.

        Raises
        ------
        ValueError
            If an array hyperparameter does not have one entry per channel.
        """
        all_media = self.interaction.media
        idx = np.array([all_media.index(m) for m in names], dtype=np.intp)
        params = {}
        for key, value in self.media.params.items():
            if np.ndim(value) == 0:
                params[key] = value
            elif np.shape(value) == (len(all_media),):
                params[key] = np.asarray(value)[idx]
            else:
                raise ValueError(
                    f"media.params['{key}'] has shape {np.shape(value)}, expected "
                    f"a scalar or one value per media channel ({len(all_media)})"
                )
        return PriorSpec(self.media.kind, params)

    def _get_beta_media(self, base_signal, spend_scaled_for_roas, frozen_data):
        """Return ``beta_media`` (dim ``"media"``), deriving ROAS channels from ``roas_media``.

        Without ROAS priors this is the plain ``beta_media`` prior.  Otherwise,
        for each ROAS channel *m* (no ``product`` term, see
        :meth:`~.mmm_config.MMMConfig._validate_roas_priors`)::

            contribution_m = beta_m * sum_t(base_signal_m,t)
            roas_m         = contribution_m / spend_scaled_for_roas_m
            => beta_m      = roas_m * spend_scaled_for_roas_m / sum_t(base_signal_m,t)

        ``sum_t(base_signal)`` is evaluated on the training data
        (``frozen_data``) so that swapping ``channel_data``/``control_data``
        with :func:`pymc.do` (counterfactuals, budget optimisation) does not
        recompute the coefficients.

        Parameters
        ----------
        base_signal : XTensorVariable
            ``x_m_transformed * boost`` with dims ``("date", "media")``.
        spend_scaled_for_roas : np.ndarray | None
            Total spend per channel in the target scale, ``media`` order
            (see ``MMMDataHandler.get_spend_scaled_for_roas``).
        frozen_data : dict | None
            ``{data_variable: constant}`` replacements pinning the training
            data in the denominator.

        Returns
        -------
        XTensorVariable
            Media coefficients with dim ``"media"``.

        Raises
        ------
        ValueError
            If ROAS priors are used without ``spend_scaled_for_roas`` /
            ``frozen_data``, or if a ROAS channel has no positive spend.
        """
        if not self._media_roas:
            return self.pymc_priors["beta_media"]

        if spend_scaled_for_roas is None or frozen_data is None:
            raise ValueError(
                "spend_scaled_for_roas and frozen_data are required when "
                "ROAS priors are used"
            )

        media_names = self.interaction.media
        roas_idx = np.array([media_names.index(m) for m in self._media_roas])
        spend_roas = np.asarray(spend_scaled_for_roas, dtype=np.float64)[roas_idx]
        if np.any(spend_roas <= 0):
            no_spend = [m for m, s in zip(self._media_roas, spend_roas) if s <= 0]
            raise ValueError(
                f"Channels {no_spend} have a ROAS prior but no positive spend"
            )

        base_total = graph_replace(
            base_signal.sum(dim="date"), frozen_data, strict=False
        ).transpose("media")
        beta_roas = (
            self.pymc_priors["roas_media"].values
            * spend_roas
            / base_total.values[roas_idx]
        )

        # Concatenate [free, roas] then permute back to canonical media order
        parts = [beta_roas]
        ordered = list(self._media_roas)
        if self._media_free:
            parts.insert(0, self.pymc_priors["beta_media_free"].values)
            ordered = self._media_free + ordered
        perm = np.array([ordered.index(m) for m in media_names], dtype=np.intp)

        beta_media = pmd.Deterministic(
            "beta_media",
            as_xtensor(pt.concatenate(parts)[perm], dims=("media",)),
            dims="media",
        )
        self.pymc_priors["beta_media"] = beta_media
        return beta_media

    def get_beta_control(self):
        """Return the full control-dim vector of control coefficients.

        Returns
        -------
        XTensorVariable
            Control coefficients with dim ``"control"``.  Entries for
            controls in ``controls_without_effect`` are zeroed out, while
            the remaining controls are taken from the ``beta_control``
            prior (dim ``"control_active"``).
        Raises
        ------
        RuntimeError
            If called before :meth:`build_pymc_priors`.
        """

        if not self.pymc_priors:
            raise RuntimeError("Call build_pymc_priors() before get_beta_control().")

        control_names = self.interaction.controls

        # beta_control (dim "control_active") only covers controls that keep
        # their own effect; embed it into a full "control"-dim vector with 0
        # for entries in controls_without_effect (they still get an
        # interaction-term contribution below, just no standalone term).
        own_effect_names = self.get_control_own_effect_names()
        own_idx = {c: i for i, c in enumerate(own_effect_names)}
        valid_mask = np.zeros(len(control_names), dtype=np.float64)
        gather_idx = np.zeros(len(control_names), dtype=np.intp)
        for j, c in enumerate(control_names):
            if c in own_idx:
                valid_mask[j] = 1.0
                gather_idx[j] = own_idx[c]
        beta_control_active = self.pymc_priors["beta_control"]
        beta_control = as_xtensor(
            beta_control_active.values[gather_idx] * valid_mask,
            dims=("control",),
        )
        return beta_control

    def get_beta_adjusted(  # pylint: disable=too-many-locals
        self,
        x_m_transformed,
        x_c,
        spend_scaled_for_roas: np.ndarray | None = None,
        frozen_data: dict | None = None,
    ):
        """Compute interaction-adjusted beta coefficients for all model components.

        For each media channel *m* with interaction formula, the effective
        (time-varying) beta combines two kinds of terms (see
        :meth:`~.formulae.Interaction.get_interaction_mode`):

        * ``boost`` terms multiply *m*'s own beta:
          ``β_m · (baseline_m + Σ β_{m:Zi}·x_Zi)`` where ``baseline_m`` is
          ``1`` if the formula has ``"1"``, else ``0``.
        * ``product`` terms contribute an independent additive term:
          ``+ Σ β_{m:Zj}·x_Zj``.

        E.g. with ``Z1`` in ``boost`` mode and ``Z2`` in ``product`` mode,
        formula ``"1 + Z1 + Z2"`` gives
        ``β_m · (1 + β_{m:Z1}·x_Z1) + β_{m:Z2}·x_Z2``.

        ``x_Zi`` is looked up in *x_m* when *Zi* is a media channel, or in
        *x_c* when *Zi* is a control — independently of its ``boost``/
        ``product`` mode.

        Must be called inside an active ``pm.Model`` context, after
        :meth:`build_pymc_priors`.

        Parameters
        ----------
        x_m_transformed : XTensorVariable
            Transformed media data with dims ``("date", "media")``.
        x_c : XTensorVariable
            Control data with dims ``("date", "control")``.
        spend_scaled_for_roas : np.ndarray | None, optional
            Total spend per channel in the target scale (``media`` order).
            Required only when ROAS priors were passed to
            :meth:`build_pymc_priors`.
        frozen_data : dict | None, optional
            ``{data_variable: constant}`` replacements pinning the training
            data when deriving ROAS-based coefficients.  Required only when
            ROAS priors were passed to :meth:`build_pymc_priors`.

        Returns
        -------
        dict[str, (Variable[Unknown, Unknown] | Unknown | XTensorConstant[XTensorType])]
            * **beta_adjusted_media** — dims ``("date", "media")``, one
              effective beta per channel and time step.
            * **beta_season** — season coefficients, dim ``"season"``
              (unchanged).
            * **beta_control** — control coefficients, dim ``"control"``
              (unchanged).
            * **interaction_contributions** — dims ``("date", "media")``,
              same shape as ``media_contribution``. Column *m* holds the
              total contribution of using *m* as an interaction term (i.e.
              ``sum_media(x_m * beta_permedia)`` across every channel *m*
              moderates), or 0 if *m* is never used as a moderator.

        Raises
        ------
        RuntimeError
            If called before :meth:`build_pymc_priors`.

        Examples
        --------
        >>> result = bp.get_beta_adjusted(x_m_transformed, x_c)  # doctest: +SKIP
        >>> media_contribution = x_m_transformed * result["media"]  # doctest: +SKIP
        """
        if not self.pymc_priors:
            raise RuntimeError("Call build_pymc_priors() before get_beta_adjusted().")

        media_names = self.interaction.media
        control_names = self.interaction.controls

        # O(1) index lookups — avoid repeated list.index() calls in the loop
        media_idx = {m: i for i, m in enumerate(media_names)}
        ctrl_idx = {c: i for i, c in enumerate(control_names)}

        interaction_terms = sorted(self.interaction.get_all_interaction_terms())

        # —————————————————————————————————————————————————————————————————————
        # 1. Gather idx and create beta_per_media
        # —————————————————————————————————————————————————————————————————————

        # pylint:disable=missing-return-doc, missing-return-type-doc, missing-any-param-doc
        def create_beta_per_media(inter_term):
            param_vector = self.pymc_priors[f"beta_interaction_{inter_term}"]
            valid_mask = np.zeros(len(media_names), dtype=np.float64)
            gather_idx = np.zeros(len(media_names), dtype=np.intp)
            for j, m in enumerate(media_names):
                if inter_term in self.interaction.parse_formula(m).terms:
                    valid_mask[j] = 1.0
                    gather_idx[j] = self._interaction_coords.get_lhs_index(
                        m, inter_term
                    )

            # One gather node for all media instead of N scalar isel nodes
            beta_per_media = as_xtensor(
                param_vector.values[gather_idx] * valid_mask,
                dims=("media",),
            )
            return beta_per_media

        # —————————————————————————————————————————————————————————————————————
        # 2. Compute interaction contributions
        # —————————————————————————————————————————————————————————————————————
        def get_media_interaction(term):
            """Compute the interaction contribution for a media term."""
            assert term in self.interaction.media
            x_term = x_m_transformed.isel(media=media_idx[term])
            beta_per_media = create_beta_per_media(term)

            interaction_contribution = x_term * beta_per_media
            return interaction_contribution

        def get_control_interaction(term):
            """Compute the interaction contribution for a control term."""
            assert term in self.interaction.controls
            x_term = x_c.isel(control=ctrl_idx[term])
            beta_per_media = create_beta_per_media(term)

            interaction_contribution = x_term * beta_per_media
            return interaction_contribution

        # pylint:enable=missing-return-doc, missing-return-type-doc, missing-any-param-doc

        # Iterate over distinct interaction terms, not over media channels.
        # For N media with T terms each, the original loop ran N×T isel-scalar ops.
        # Here we run T gather ops, one per distinct term regardless of N.
        boost_contributions = {}
        media_adjusted = {"boost": [], "prod": []}
        for term in interaction_terms:
            # create interaction_contribution for each term :
            # prod : beta_permedia * x_term
            # boost : beta_permedia * x_term
            # e.g.: term = "TV"
            # -> beta_permedia = beta_interaction_TV
            # -> interaction_contribution = beta_permedia * x_TV

            if term in self.interaction.media:
                interaction_contribution = get_media_interaction(term)
            else:
                interaction_contribution = get_control_interaction(term)

            mode = self.interaction.get_interaction_mode(term)
            bucket = "prod" if mode == "product" else "boost"
            media_adjusted[bucket].append(interaction_contribution)

            if bucket == "boost":
                boost_contributions[term] = interaction_contribution
            else:
                warnings.warn(
                    f"Interaction term {term} is in 'product' mode is not yet"
                    "supported for computing interaction_contributions. "
                )

        # Baseline per channel as a constant xtensor vector
        boost = as_xtensor(
            np.array(
                [
                    float(self.interaction.parse_formula(m).has_baseline)
                    for m in media_names
                ]
            ),
            dims=("media",),
        )

        # 1. Apply boost contributions
        for term_contrib in media_adjusted["boost"]:
            boost = boost + term_contrib

        # boost does not depend on beta_media, so ROAS-based coefficients
        # can be derived from the boosted signal x_m_transformed * boost.
        beta_media = self._get_beta_media(
            x_m_transformed * boost, spend_scaled_for_roas, frozen_data
        )  # dim: "media"
        term_totals = {
            term: beta_media * contrib * x_m_transformed
            for term, contrib in boost_contributions.items()
        }

        beta_media_adjusted = beta_media * boost
        # 2. Apply product contributions
        for term_contrib in media_adjusted["prod"]:
            beta_media_adjusted = beta_media_adjusted + term_contrib

        # if "date" not in boost.dims:
        #     boost = boost.broadcast_like(x_m.isel(media=0))

        self.create_interaction_contributions(term_totals)

        return {
            "media": beta_media_adjusted,
            "season": self.pymc_priors["beta_season"],  # dim: "season"
            "control": self.get_beta_control(),
            "interaction_contributions": None,
        }

    def create_interaction_contributions(self, term_totals):
        """Create interaction contributions for each term in term_totals.

        Parameters
        ----------
        term_totals : dict
            Dictionary mapping interaction terms to their corresponding contributions.
        """
        # One column per media channel, same "media" dim/coords as
        # `media_contribution`: `term_totals[m]` when `m` is itself used as
        # an interaction term (moderates other channels), else 0.
        # zero_column = ptx.zeros_like(x_m_transformed.isel(media=0))
        for term, value in term_totals.items():
            name = f"{term}_interaction_contributions"

            _ = pmd.Deterministic(
                name,
                value,
                # ptx.concat(
                #     [
                #         term_totals.get(m, zero_column).expand_dims(dim="media")
                #         for m in media_names
                #     ],
                #     dim="media",
                # ),
                dims=("date", "media"),
            )
            # if len(interaction_terms) != 0:
            self.expressions_to_compute.append(name)
