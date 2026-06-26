"""Pytest unit tests for _compute_groups and _spec_struct_key."""

import pytest

from mmm_utils.modeling.model_definition.mmm_config import (
    MediaTransformSpec,
    _compute_groups,
    _spec_struct_key,
)
from mmm_utils.modeling.prior import PriorSpec

# pylint: skip-file


# ── Fixtures ──────────────────────────────────────────────────────────────────

GEOMETRIC_BETA = MediaTransformSpec(
    adstock="Geometric",
    adstock_params={"l_max": 12},
    adstock_priors={"alpha": PriorSpec("Beta", {"alpha": 2.0, "beta": 2.0})},
    saturation="Logistic",
)

GEOMETRIC_BETA_DIFFERENT_HYPERPARAMS = MediaTransformSpec(
    adstock="Geometric",
    adstock_params={"l_max": 12},
    adstock_priors={"alpha": PriorSpec("Beta", {"alpha": 4.0, "beta": 0.5})},
    saturation="Logistic",
)

GEOMETRIC_HALFNORMAL = MediaTransformSpec(
    adstock="Geometric",
    adstock_params={"l_max": 12},
    adstock_priors={"alpha": PriorSpec("HalfNormal", {"sigma": 0.5})},
    saturation="Logistic",
)

GEOMETRIC_DELAYED_BETA = MediaTransformSpec(
    adstock="GeometricDelayed",
    adstock_params={"l_max": 12},
    adstock_priors={
        "alpha": PriorSpec("Beta", {"alpha": 4.0, "beta": 0.5}),
        "theta": PriorSpec("Normal", {"mu": 2.0, "sigma": 0.2}),
    },
    saturation="Logistic",
    saturation_params={},
    saturation_priors={"lam": PriorSpec("LogNormal", {"mu": 0.0, "sigma": 1.0})},
)

GEOMETRIC_DELAYED_BETA_2 = MediaTransformSpec(
    adstock="GeometricDelayed",
    adstock_params={"l_max": 12},
    adstock_priors={
        "alpha": PriorSpec("Beta", {"alpha": 3.0, "beta": 1.0}),
        "theta": PriorSpec("Normal", {"mu": 3.0, "sigma": 0.3}),
    },
    saturation="Logistic",
    saturation_params={},
    saturation_priors={"lam": PriorSpec("LogNormal", {"mu": 0.5, "sigma": 0.5})},
)

DIFFERENT_LMAX = MediaTransformSpec(
    adstock="Geometric",
    adstock_params={"l_max": 6},
    adstock_priors={"alpha": PriorSpec("Beta", {"alpha": 2.0, "beta": 2.0})},
    saturation="Logistic",
)

NO_PRIORS = MediaTransformSpec(
    adstock="Geometric",
    saturation="Logistic",
)


# ── _spec_struct_key ──────────────────────────────────────────────────────────


def test_struct_key_same_for_identical_specs():
    assert _spec_struct_key(GEOMETRIC_BETA) == _spec_struct_key(GEOMETRIC_BETA)


def test_struct_key_same_for_different_hyperparams():
    """Two specs with the same distribution kind but different values → same key."""
    assert _spec_struct_key(GEOMETRIC_BETA) == _spec_struct_key(
        GEOMETRIC_BETA_DIFFERENT_HYPERPARAMS
    )


def test_struct_key_differs_for_different_distribution_kind():
    """Beta vs HalfNormal → different structural key even for the same param name."""
    assert _spec_struct_key(GEOMETRIC_BETA) != _spec_struct_key(GEOMETRIC_HALFNORMAL)


def test_struct_key_differs_for_different_adstock_type():
    assert _spec_struct_key(GEOMETRIC_BETA) != _spec_struct_key(GEOMETRIC_DELAYED_BETA)


def test_struct_key_differs_for_different_lmax():
    assert _spec_struct_key(GEOMETRIC_BETA) != _spec_struct_key(DIFFERENT_LMAX)


def test_struct_key_differs_for_extra_prior_param():
    """alpha only vs alpha+theta → different keys."""
    spec_alpha_only = MediaTransformSpec(
        adstock="GeometricDelayed",
        adstock_params={"l_max": 12},
        adstock_priors={"alpha": PriorSpec("Beta", {"alpha": 2.0, "beta": 2.0})},
        saturation="Logistic",
        saturation_params={},
        saturation_priors={"lam": PriorSpec("LogNormal", {"mu": 0.0, "sigma": 1.0})},
    )
    assert _spec_struct_key(spec_alpha_only) != _spec_struct_key(GEOMETRIC_DELAYED_BETA)


# ── _compute_groups: basic ────────────────────────────────────────────────────


def test_empty_media_names():
    groups = _compute_groups([], {})
    assert groups == []


def test_single_channel():
    groups = _compute_groups(["TV"], {"TV": GEOMETRIC_BETA})
    assert groups == [["TV"]]


def test_all_channels_compatible_form_one_group():
    """Different hyperparams but same distribution kind → one group."""
    groups = _compute_groups(
        ["TV", "SEA", "Social"],
        {
            "TV": GEOMETRIC_BETA,
            "SEA": GEOMETRIC_BETA_DIFFERENT_HYPERPARAMS,
            "Social": MediaTransformSpec(
                adstock="Geometric",
                adstock_params={"l_max": 12},
                adstock_priors={
                    "alpha": PriorSpec("Beta", {"alpha": 1.0, "beta": 1.0})
                },
                saturation="Logistic",
            ),
        },
    )
    assert len(groups) == 1
    assert sorted(groups[0]) == ["SEA", "Social", "TV"]


def test_incompatible_channels_form_separate_groups():
    """Geometric vs GeometricDelayed → two separate groups."""
    groups = _compute_groups(
        ["TV", "SEA"],
        {"TV": GEOMETRIC_DELAYED_BETA, "SEA": GEOMETRIC_BETA},
    )
    assert len(groups) == 2


def test_different_distribution_kinds_form_separate_groups():
    """Beta vs HalfNormal for alpha → two separate groups."""
    groups = _compute_groups(
        ["TV", "SEA"],
        {"TV": GEOMETRIC_BETA, "SEA": GEOMETRIC_HALFNORMAL},
    )
    assert len(groups) == 2


# ── _compute_groups: grouping correctness ────────────────────────────────────


def test_two_compatible_one_incompatible():
    """TV and TV2 share GeometricDelayed; SEA uses Geometric → 2 groups."""
    groups = _compute_groups(
        ["TV", "TV2", "SEA"],
        {
            "TV": GEOMETRIC_DELAYED_BETA,
            "TV2": GEOMETRIC_DELAYED_BETA_2,
            "SEA": GEOMETRIC_BETA,
        },
    )
    assert len(groups) == 2
    grp_by_size = sorted(groups, key=len, reverse=True)
    assert sorted(grp_by_size[0]) == ["TV", "TV2"]
    assert grp_by_size[1] == ["SEA"]


def test_three_distinct_specs_produce_three_singleton_groups():
    groups = _compute_groups(
        ["TV", "SEA", "Social"],
        {"TV": GEOMETRIC_DELAYED_BETA, "SEA": GEOMETRIC_BETA, "Social": DIFFERENT_LMAX},
    )
    assert len(groups) == 3
    assert all(len(g) == 1 for g in groups)


# ── _compute_groups: ordering ─────────────────────────────────────────────────


def test_channel_order_is_preserved_within_group():
    """Channels in a group must appear in the same order as media_names."""
    groups = _compute_groups(
        ["Social", "SEA", "TV"],
        {
            "Social": GEOMETRIC_BETA,
            "SEA": GEOMETRIC_BETA_DIFFERENT_HYPERPARAMS,
            "TV": GEOMETRIC_BETA,
        },
    )
    assert len(groups) == 1
    assert groups[0] == ["Social", "SEA", "TV"]


def test_group_order_follows_first_channel_occurrence():
    """Group order reflects the first occurrence of each structural key in media_names."""
    groups = _compute_groups(
        ["TV", "SEA", "TV2"],
        {
            "TV": GEOMETRIC_DELAYED_BETA,
            "SEA": GEOMETRIC_BETA,
            "TV2": GEOMETRIC_DELAYED_BETA_2,
        },
    )
    # GeometricDelayed (TV first) appears before Geometric (SEA)
    assert groups[0] == ["TV", "TV2"]
    assert groups[1] == ["SEA"]


# ── _compute_groups: missing spec defaults ────────────────────────────────────


def test_missing_spec_uses_default():
    """Channels absent from media_transforms get a default MediaTransformSpec."""
    groups = _compute_groups(["TV", "unknown"], {"TV": NO_PRIORS})
    # Both use the same default Geometric/Logistic with no priors → one group
    assert len(groups) == 1
    assert sorted(groups[0]) == ["TV", "unknown"]


def test_explicit_default_matches_missing_spec():
    """Explicitly passing MediaTransformSpec() equals the fallback for a missing key."""
    transforms_explicit = {"TV": NO_PRIORS, "SEA": NO_PRIORS}
    transforms_missing = {"TV": NO_PRIORS}
    groups_explicit = _compute_groups(["TV", "SEA"], transforms_explicit)
    groups_missing = _compute_groups(["TV", "SEA"], transforms_missing)
    assert groups_explicit == groups_missing
