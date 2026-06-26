"""
Configuration classes for the MMM model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .beta_priors import BetaPriors

from ..prior import PriorSpec
from ..adstocks import AdstockType
from ..saturation import SaturationType


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


def _adstock_struct_key(spec: MediaTransformSpec) -> tuple:
    """Structural key for adstock grouping.

    ``l_max`` is intentionally excluded: channels that share the same adstock
    type, normalize flag, and prior distribution kinds can be batched together
    even if their ``l_max`` values differ — :func:`_compute_adstock_groups`
    will unify them to the maximum ``l_max`` and emit a warning.
    """
    return (
        spec.adstock,
        spec.adstock_params.get("normalize"),
        frozenset((n, p.kind) for n, p in spec.adstock_priors.items()),
    )


def _saturation_struct_key(spec: MediaTransformSpec) -> tuple:
    """Structural key for saturation grouping.

    Two channels with the same key can share a single vectorized saturation
    call regardless of their adstock configuration.
    """
    return (
        spec.saturation,
        frozenset((n, p.kind) for n, p in spec.saturation_priors.items()),
    )


def _spec_struct_key(spec: MediaTransformSpec) -> tuple:
    """Combined structural key (adstock + saturation) for full compatibility.

    Kept for backward compatibility with :func:`_compute_groups`.  Use
    :func:`_adstock_struct_key` or :func:`_saturation_struct_key` directly
    when working with the two-phase :class:`~.TransformHandler`.
    """
    return (
        spec.adstock,
        spec.adstock_params.get("l_max"),
        spec.adstock_params.get("normalize"),
        spec.saturation,
        frozenset((n, p.kind) for n, p in spec.adstock_priors.items()),
        frozenset((n, p.kind) for n, p in spec.saturation_priors.items()),
    )


def _compute_adstock_groups(
    media_names: list[str],
    media_transforms: dict[str, "MediaTransformSpec"],
) -> list[list[str]]:
    """Group channels by adstock compatibility (excludes ``l_max`` from the key).

    Channels in the same group can share vectorized adstock priors.  When
    ``l_max`` values differ inside a group, :class:`~.TransformHandler`
    unifies them to the maximum and emits a :func:`warnings.warn`.
    """
    seen: dict[tuple, list[str]] = {}
    for name in media_names:
        key = _adstock_struct_key(media_transforms.get(name, MediaTransformSpec()))
        seen.setdefault(key, []).append(name)
    return list(seen.values())


def _compute_saturation_groups(
    media_names: list[str],
    media_transforms: dict[str, "MediaTransformSpec"],
) -> list[list[str]]:
    """Group channels by saturation compatibility.

    Channels in the same group can share vectorized saturation priors,
    independently of their adstock configuration.
    """
    seen: dict[tuple, list[str]] = {}
    for name in media_names:
        key = _saturation_struct_key(media_transforms.get(name, MediaTransformSpec()))
        seen.setdefault(key, []).append(name)
    return list(seen.values())


def _compute_groups(
    media_names: list[str],
    media_transforms: dict[str, "MediaTransformSpec"],
) -> list[list[str]]:
    """Group media channels by combined adstock+saturation compatibility.

    Kept for backward compatibility.  :class:`~.TransformHandler` uses the
    finer-grained :func:`_compute_adstock_groups` and
    :func:`_compute_saturation_groups` instead.
    """
    seen: dict[tuple, list[str]] = {}
    for name in media_names:
        key = _spec_struct_key(media_transforms.get(name, MediaTransformSpec()))
        seen.setdefault(key, []).append(name)
    return list(seen.values())


@dataclass(slots=True)
class MMMConfig:  # pylint: disable=too-many-instance-attributes
    """Main configuration object for the MMM model."""

    beta_priors: BetaPriors = field(default_factory=BetaPriors)

    date_name: str = "date"
    media_names: list[str] = field(default_factory=list)
    control_names: list[str] = field(default_factory=list)
    seasonality_order: int = 0
    media_transforms: dict[str, MediaTransformSpec] = field(default_factory=dict)
    random_seed: int = 42

    prior_sigma: PriorSpec = field(
        default_factory=lambda: PriorSpec("HalfNormal", {"sigma": 1.0})
    )

    def var_names(self) -> list[str]:
        """List all variable names in the model, including media and control parameters.

        Adstock and saturation variables are named independently based on their
        respective groupings (:func:`_compute_adstock_groups` and
        :func:`_compute_saturation_groups`):

        * **Single group** (all channels compatible): one vectorized variable
          per stochastic parameter, no suffix (e.g. ``"adstock_alpha"``).
        * **Multiple groups, group of N≥2**: suffix ``_agrp{i}`` for adstock
          (e.g. ``"adstock_alpha_agrp0"``) and ``_sgrp{j}`` for saturation.
        * **Singleton group**: channel-name suffix
          (e.g. ``"adstock_alpha[TV]"``).

        Returns
        -------
        list[str]
            List of variable names in the model.
        """
        adstock_groups = _compute_adstock_groups(
            self.media_names, self.media_transforms
        )
        saturation_groups = _compute_saturation_groups(
            self.media_names, self.media_transforms
        )
        single_adstock = len(adstock_groups) == 1
        single_saturation = len(saturation_groups) == 1

        adstock_vars: list[str] = []
        for grp_idx, group_names in enumerate(adstock_groups):
            spec = self.media_transforms.get(group_names[0], MediaTransformSpec())
            if single_adstock:
                adstock_vars += [f"adstock_{p}" for p in spec.adstock_priors]
            elif len(group_names) == 1:
                n = group_names[0]
                adstock_vars += [f"adstock_{p}[{n}]" for p in spec.adstock_priors]
            else:
                adstock_vars += [
                    f"adstock_{p}_agrp{grp_idx}" for p in spec.adstock_priors
                ]

        sat_vars: list[str] = []
        for grp_idx, group_names in enumerate(saturation_groups):
            spec = self.media_transforms.get(group_names[0], MediaTransformSpec())
            if single_saturation:
                sat_vars += [f"saturation_{p}" for p in spec.saturation_priors]
            elif len(group_names) == 1:
                n = group_names[0]
                sat_vars += [f"saturation_{p}[{n}]" for p in spec.saturation_priors]
            else:
                sat_vars += [
                    f"saturation_{p}_sgrp{grp_idx}" for p in spec.saturation_priors
                ]

        return [
            "beta_media",
            "beta_control",
            *adstock_vars,
            *sat_vars,
            "beta_season",
            "sigma",
            *self.beta_priors.interaction.get_unique_parameter_names(),
        ]
