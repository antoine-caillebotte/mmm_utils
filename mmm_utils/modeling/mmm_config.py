"""
Configuration classes for the MMM model.
"""

from dataclasses import dataclass, field

from .prior import PriorSpec
from .adstocks import AdstockType
from .saturation import SaturationType


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

    # def get_media_with_transforms(
    #     self, transform_fct: Callable[[str, MediaTransformSpec], bool]
    # ) -> dict[str, MediaTransformSpec]:
    #     """List media channels with a specific transform function."""
    #     media = {}
    #     for m, spec in self.media_transforms.items():
    #         if transform_fct(m, spec):
    #             media[m] = spec
    #     return media

    # def get_list_media_with_priors(self) -> dict[str, list[str]]:
    #     """List media channels with priors for adstock/saturation params."""

    #     x = {}
    #     for adstock_type in AdstockType.__args__:
    #         x[f"adstock_{adstock_type}"] = self.get_media_with_transforms(
    #             lambda m, spec: spec.adstock == adstock_type
    #         )

    #     for saturation_type in SaturationType.__args__:
    #         x[f"saturation_{saturation_type}"] = self.get_media_with_transforms(
    #             lambda m, spec: spec.saturation == saturation_type
    #         )
    #     return x

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
            *[f"product_media[{m}]" for m in self.prior_product_media],
        ]
