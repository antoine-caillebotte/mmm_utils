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

        Returns
        -------
        list[str]
            List of variable names in the model.
        """
        return [
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
            *self.beta_priors.interaction.get_unique_parameter_names(),
        ]
