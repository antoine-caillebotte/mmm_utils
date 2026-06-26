"""
Transform utilities for MMM media processing.

Imports :class:`Transform` and :func:`validate_params` from :mod:`.transform`
and defines :class:`TransformHandler` which owns the full adstock-and-saturation
pipeline for all media channels via two independent grouping phases.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pymc as pm
import pytensor.xtensor as ptx
from pytensor.xtensor.type import as_xtensor

from .transform import Transform, validate_params  # noqa: F401
from .adstocks import Adstock
from .saturation import Saturation
from .prior import _make_prior, PriorSpec
from .model_definition.mmm_config import (
    MediaTransformSpec,
    _compute_adstock_groups,
    _compute_saturation_groups,
)

if TYPE_CHECKING:
    from pytensor.xtensor import XTensorVariable

__all__ = ["Transform", "validate_params", "TransformHandler"]


# ── Module-level helpers ───────────────────────────────────────────────────────


def _has_dim(v, dim: str) -> bool:
    """Return ``True`` if *v* is an XTensorVariable carrying dimension *dim*."""
    return hasattr(v, "type") and hasattr(v.type, "dims") and dim in v.type.dims


def _has_media_dim(v) -> bool:
    """Return ``True`` if *v* is an XTensorVariable carrying a ``"media"`` dim."""
    return _has_dim(v, "media")


def _slice_params(params: dict, dim: str, j: int) -> dict:
    """Return a copy of *params* with any XTensor having dimension *dim* sliced to index *j*."""
    return {
        k: (v.isel(**{dim: j}) if _has_dim(v, dim) else v) for k, v in params.items()
    }


# ── TransformHandler ───────────────────────────────────────────────────────────


@dataclass
class TransformHandler:
    """Adstock and saturation pipeline for all media channels.

    The pipeline runs in two independent phases, each with its own grouping:

    **Phase 1 — Adstock** (:func:`_compute_adstock_groups`):
    Channels are grouped by adstock type, ``normalize``, and prior distribution
    kinds.  ``l_max`` is *excluded* from the key: channels with different
    ``l_max`` values land in the same group and are unified to the maximum
    value (a :func:`warnings.warn` is emitted).

    **Phase 2 — Saturation** (:func:`_compute_saturation_groups`):
    Channels are re-grouped by saturation type and prior distribution kinds,
    independently of their adstock grouping.  This lets channels that share
    the same saturation but differ in adstock (or vice-versa) still benefit
    from vectorized priors.

    **Naming convention**:

    * Single adstock/saturation group (all channels): ``"adstock_alpha"`` /
      ``"saturation_lam"`` with ``dims="media"`` — backward-compatible.
    * Multiple adstock groups, group of N≥2: ``"adstock_alpha_agrp{i}"``
      with ``dims="media_agrp{i}"``.
    * Multiple saturation groups, group of N≥2: ``"saturation_lam_sgrp{j}"``
      with ``dims="media_sgrp{j}"``.
    * Singleton group: ``"adstock_alpha[{channel}]"`` / ``"saturation_lam[{channel}]"``
      as scalars.

    Parameters
    ----------
    media_names : list[str]
        Ordered list of media-channel names.
    media_transforms : dict[str, MediaTransformSpec]
        Per-channel transform specification.

    Attributes
    ----------
    adstocks : dict[str, Adstock]
        Per-channel instances, populated by :meth:`apply`.
    saturations : dict[str, Saturation]
        Per-channel instances, populated by :meth:`apply`.
    """

    media_names: list[str]
    media_transforms: dict[str, MediaTransformSpec]

    adstocks: dict[str, Adstock] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    saturations: dict[str, Saturation] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    # ── Public API ─────────────────────────────────────────────────────────────

    def apply(self, x_m) -> XTensorVariable:
        """Apply adstock then saturation to all media channels.

        Must be called inside an active ``pm.Model`` context.

        Parameters
        ----------
        x_m : XTensorVariable
            Raw media data, dims ``("date", "media")``.

        Returns
        -------
        XTensorVariable
            Transformed media matrix, dims ``("date", "media")``.
        """
        x_m = as_xtensor(x_m.values, dims=("date", "media"))
        x_adstocked = self._apply_adstock_phase(x_m)
        x_result = self._apply_saturation_phase(x_adstocked)
        return x_result.transpose("date", "media")

    # ── Phase 1: adstock ───────────────────────────────────────────────────────

    def _apply_adstock_phase(self, x_m: XTensorVariable) -> XTensorVariable:  # pylint: disable=too-many-locals
        """Apply adstock to all channels grouped by adstock compatibility.

        Returns a ``("date", "media")`` XTensorVariable.
        """
        adstock_groups = _compute_adstock_groups(
            self.media_names, self.media_transforms
        )
        single_group = len(adstock_groups) == 1

        if single_group:
            group_names = adstock_groups[0]
            specs = self._specs_for(group_names)
            spec_ref = specs[group_names[0]]
            l_max = self._resolve_lmax(group_names, specs)

            params = self._build_vectorized_params(
                spec_ref.adstock_params,
                spec_ref.adstock_priors,
                specs,
                "adstock",
                "",
                "media",
            )
            ad = self._make_adstock(spec_ref, params, l_max)
            x_adstocked = ad(x_m)

            for j, name in enumerate(group_names):
                self.adstocks[name] = self._make_adstock(
                    spec_ref, _slice_params(params, "media", j), l_max
                )
            return x_adstocked.transpose("date", "media")

        col_map: dict[str, XTensorVariable] = {}
        for grp_idx, group_names in enumerate(adstock_groups):
            specs = self._specs_for(group_names)
            spec_ref = specs[group_names[0]]
            l_max = self._resolve_lmax(group_names, specs)

            if len(group_names) == 1:
                name = group_names[0]
                col = as_xtensor(
                    x_m.values[:, self.media_names.index(name)], dims=("date",)
                )
                params = self._build_scalar_params(
                    spec_ref.adstock_params,
                    spec_ref.adstock_priors,
                    "adstock",
                    f"[{name}]",
                )
                ad = self._make_adstock(spec_ref, params, l_max)
                col_map[name] = ad(col)
                self.adstocks[name] = ad
            else:
                grp_dim = f"media_agrp{grp_idx}"
                pm.modelcontext(None).add_coords({grp_dim: group_names})
                grp_idx_arr = np.array([self.media_names.index(n) for n in group_names])
                x_grp = as_xtensor(x_m.values[:, grp_idx_arr], dims=("date", grp_dim))

                params = self._build_vectorized_params(
                    spec_ref.adstock_params,
                    spec_ref.adstock_priors,
                    specs,
                    "adstock",
                    f"_agrp{grp_idx}",
                    grp_dim,
                )
                ad = self._make_adstock(spec_ref, params, l_max)
                x_grp_ad = ad(x_grp)

                for j, name in enumerate(group_names):
                    self.adstocks[name] = self._make_adstock(
                        spec_ref, _slice_params(params, grp_dim, j), l_max
                    )
                    col_map[name] = x_grp_ad.isel(**{grp_dim: j})

        cols = [col_map[n].expand_dims(dim="media") for n in self.media_names]
        return ptx.concat(cols, dim="media")

    # ── Phase 2: saturation ────────────────────────────────────────────────────

    def _apply_saturation_phase(self, x_adstocked: XTensorVariable) -> XTensorVariable:  # pylint: disable=too-many-locals
        """Apply saturation to all channels grouped by saturation compatibility.

        Parameters
        ----------
        x_adstocked : XTensorVariable
            Adstock-transformed media, dims ``("date", "media")``.

        Returns a ``("date", "media")`` XTensorVariable.
        """
        saturation_groups = _compute_saturation_groups(
            self.media_names, self.media_transforms
        )
        single_group = len(saturation_groups) == 1

        if single_group:
            group_names = saturation_groups[0]
            specs = self._specs_for(group_names)
            spec_ref = specs[group_names[0]]

            params = self._build_vectorized_params(
                spec_ref.saturation_params,
                spec_ref.saturation_priors,
                specs,
                "saturation",
                "",
                "media",
            )
            sat = self._make_saturation(spec_ref, params)
            x_result = sat(x_adstocked)

            for j, name in enumerate(group_names):
                self.saturations[name] = self._make_saturation(
                    spec_ref, _slice_params(params, "media", j)
                )
            return x_result

        # x_adstocked must be in (date, media) order for integer slicing
        x_ad = x_adstocked.transpose("date", "media")

        col_map: dict[str, XTensorVariable] = {}
        for grp_idx, group_names in enumerate(saturation_groups):
            specs = self._specs_for(group_names)
            spec_ref = specs[group_names[0]]

            if len(group_names) == 1:
                name = group_names[0]
                col = as_xtensor(
                    x_ad.values[:, self.media_names.index(name)], dims=("date",)
                )
                params = self._build_scalar_params(
                    spec_ref.saturation_params,
                    spec_ref.saturation_priors,
                    "saturation",
                    f"[{name}]",
                )
                sat = self._make_saturation(spec_ref, params)
                col_map[name] = sat(col)
                self.saturations[name] = sat
            else:
                grp_dim = f"media_sgrp{grp_idx}"
                pm.modelcontext(None).add_coords({grp_dim: group_names})
                grp_idx_arr = np.array([self.media_names.index(n) for n in group_names])
                x_grp = as_xtensor(x_ad.values[:, grp_idx_arr], dims=("date", grp_dim))

                params = self._build_vectorized_params(
                    spec_ref.saturation_params,
                    spec_ref.saturation_priors,
                    specs,
                    "saturation",
                    f"_sgrp{grp_idx}",
                    grp_dim,
                )
                sat = self._make_saturation(spec_ref, params)
                x_grp_sat = sat(x_grp)

                for j, name in enumerate(group_names):
                    self.saturations[name] = self._make_saturation(
                        spec_ref, _slice_params(params, grp_dim, j)
                    )
                    col_map[name] = x_grp_sat.isel(**{grp_dim: j})

        cols = [col_map[n].expand_dims(dim="media") for n in self.media_names]
        return ptx.concat(cols, dim="media")

    # ── Private: helpers ───────────────────────────────────────────────────────

    def _specs_for(self, names: list[str]) -> dict[str, MediaTransformSpec]:
        return {n: self.media_transforms.get(n, MediaTransformSpec()) for n in names}

    @staticmethod
    def _resolve_lmax(
        group_names: list[str], specs: dict[str, MediaTransformSpec]
    ) -> int:
        """Return the unified ``l_max`` for a group, warning when values differ."""
        lmax_values = [specs[n].adstock_params.get("l_max", 12) for n in group_names]
        l_max = max(lmax_values)
        if len(set(lmax_values)) > 1:
            warnings.warn(
                f"Channels {group_names} have different l_max values {lmax_values}. "
                f"Using l_max={l_max} (maximum) for all channels in this adstock group.",
                UserWarning,
                stacklevel=4,
            )
        return l_max

    # ── Private: param builders ────────────────────────────────────────────────

    @staticmethod
    def _build_scalar_params(
        fixed_params: dict,
        prior_specs: dict[str, PriorSpec],
        kind: str,
        suffix: str,
    ) -> dict:
        """Merge fixed params with scalar (per-channel) stochastic priors."""
        params = dict(fixed_params)
        for pname, pspec in prior_specs.items():
            params[pname] = _make_prior(f"{kind}_{pname}{suffix}", pspec)
        return params

    @staticmethod
    def _build_vectorized_params(  # pylint: disable=too-many-positional-arguments, too-many-arguments
        fixed_params: dict,
        prior_specs: dict[str, PriorSpec],
        specs_per_channel: dict[str, MediaTransformSpec],
        kind: str,
        suffix: str,
        grp_dim: str,
    ) -> dict:
        """Merge fixed params with vectorized stochastic priors.

        Hyperparameters are gathered from every channel and stacked into
        ``np.ndarray`` vectors, so a single ``pm.Distribution`` of shape
        ``(n_channels,)`` covers all channels with per-element hyperparams.
        """
        params = dict(fixed_params)
        channel_names = list(specs_per_channel.keys())
        for pname, base_pspec in prior_specs.items():
            channel_pspecs = [
                getattr(specs_per_channel[n], f"{kind}_priors")[pname]
                for n in channel_names
            ]
            vector_hyperparams = {
                hp: np.array([ps.params[hp] for ps in channel_pspecs], dtype=np.float64)
                for hp in base_pspec.params
            }
            params[pname] = _make_prior(
                f"{kind}_{pname}{suffix}",
                PriorSpec(base_pspec.kind, vector_hyperparams),
                grp_dim,
            )
        return params

    # ── Private: instance factories ───────────────────────────────────────────

    @staticmethod
    def _make_adstock(
        spec: MediaTransformSpec, params: dict, l_max: int | None = None
    ) -> Adstock:
        return Adstock.from_spec(
            kind=spec.adstock,
            dim="date",
            l_max=l_max if l_max is not None else spec.adstock_params.get("l_max"),
            normalize=spec.adstock_params.get("normalize"),
            params=params,
        )

    @staticmethod
    def _make_saturation(spec: MediaTransformSpec, params: dict) -> Saturation:
        return Saturation.from_spec(kind=spec.saturation, params=params)
