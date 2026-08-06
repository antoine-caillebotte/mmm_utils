"""Pytest unit tests for Interaction, InteractionCoordinates and SharingGroup."""

import warnings

import pytest

from mmm_utils.modeling import (
    Interaction,
    InteractionCoordinates,
    InteractionFormula,
    SharingGroup,
)

# pylint: skip-file


# ---------------------------------------------------------------------------
# SharingGroup
# ---------------------------------------------------------------------------


def test_sharing_group_basic() -> None:
    group = SharingGroup(interact_var="TV", media=("Digital", "SEA"))
    assert group.interact_var == "TV"
    assert group.media == ("Digital", "SEA")


def test_sharing_group_too_few_media_raises() -> None:
    with pytest.raises(ValueError):
        SharingGroup(interact_var="TV", media=("Digital",))


def test_sharing_group_empty_interact_var_raises() -> None:
    with pytest.raises(ValueError):
        SharingGroup(interact_var="  ", media=("A", "B"))


def test_sharing_group_is_frozen() -> None:
    group = SharingGroup(interact_var="TV", media=("A", "B"))
    with pytest.raises(AttributeError):
        group.interact_var = "SEA"  # type: ignore[misc]


def test_sharing_group_coerce_from_tuple() -> None:
    group = SharingGroup._coerce(("TV", "Digital", "SEA"))
    assert group == SharingGroup(interact_var="TV", media=("Digital", "SEA"))


def test_sharing_group_coerce_passthrough() -> None:
    group = SharingGroup(interact_var="TV", media=("A", "B"))
    assert SharingGroup._coerce(group) is group


# ---------------------------------------------------------------------------
# Interaction — raw tuple compatibility for is_shared_with
# ---------------------------------------------------------------------------


def test_is_shared_with_accepts_raw_tuples() -> None:
    ia = Interaction(
        formulas={"Y1": "1 + Y3", "Y2": "1 + Y3", "Y3": "1"},
        is_shared_with=[("Y3", "Y1", "Y2")],
        media=["Y1", "Y2"],
    )
    assert ia.is_shared_with == [SharingGroup(interact_var="Y3", media=("Y1", "Y2"))]


def test_is_shared_with_accepts_sharing_group_instances() -> None:
    ia = Interaction(
        formulas={"Y1": "1 + Y3", "Y2": "1 + Y3", "Y3": "1"},
        is_shared_with=[SharingGroup(interact_var="Y3", media=("Y1", "Y2"))],
        media=["Y1", "Y2"],
    )
    assert ia.is_shared_with == [SharingGroup(interact_var="Y3", media=("Y1", "Y2"))]


# ---------------------------------------------------------------------------
# Interaction — validation
# ---------------------------------------------------------------------------


def test_validation_sharing_group_too_small_raises() -> None:
    with pytest.raises(ValueError):
        Interaction(
            formulas={"Y1": "1 + Y3", "Y3": "1"},
            is_shared_with=[("Y3", "Y1")],  # only one media, needs ≥2
        )


def test_validation_sharing_group_media_without_explicit_formula_raises() -> None:
    # Y2 has no explicit formula, so Y3 cannot be verified in it
    with pytest.raises(ValueError):
        Interaction(
            formulas={"Y1": "1 + Y3", "Y3": "1"},
            is_shared_with=[("Y3", "Y1", "Y2")],
        )


def test_validation_sharing_group_interact_var_not_in_formula_raises() -> None:
    # Y1 does not have Y3 in its formula
    with pytest.raises(ValueError):
        Interaction(
            formulas={"Y1": "1 + C", "Y2": "1 + Y3", "C": "1", "Y3": "1"},
            is_shared_with=[("Y3", "Y1", "Y2")],
        )


def test_validation_shared_parameter_name_empty_raises() -> None:
    with pytest.raises(ValueError):
        Interaction(
            formulas={"Y1": "1 + Y3", "Y2": "1 + Y3", "Y3": "1"},
            is_shared_with=[("", "Y1", "Y2")],
        )


def test_validation_conflicting_sharing_groups_raises() -> None:
    # (Y1, Y3) cannot belong to two different sharing groups at once.
    with pytest.raises(ValueError, match="Conflicting sharing groups"):
        Interaction(
            formulas={"Y1": "1 + Y3", "Y2": "1 + Y3", "Y4": "1 + Y3", "Y3": "1"},
            is_shared_with=[("Y3", "Y1", "Y2"), ("Y3", "Y1", "Y4")],
            media=["Y1", "Y2", "Y4"],
        )


def test_undefined_term_raises() -> None:
    with pytest.raises(ValueError, match="undefined variable 'Cospirit'"):
        Interaction(
            formulas={"TV": "1", "SEA": "1 + TV + Cospirit"},
            media=["TV", "SEA"],
        )


def test_undefined_term_ok_when_in_media() -> None:
    # TV is in media but absent from formulas — still a valid term reference
    Interaction(
        formulas={"SEA": "1 + TV"},
        media=["TV", "SEA"],
    )


def test_undefined_term_raises_without_media_arg() -> None:
    # Terms must be in formulas.keys() when media is not provided
    with pytest.raises(ValueError, match="undefined variable 'Y3'"):
        Interaction(formulas={"Y1": "1 + Y3"})


# ---------------------------------------------------------------------------
# Interaction — parse_formula / get_all_interaction_terms / get_default_controls
# ---------------------------------------------------------------------------


def test_parse_formula_registered_formula_is_returned() -> None:
    ia = Interaction(formulas={"Y1": "1 + Y2", "Y2": "1"})
    formula = ia.parse_formula("Y1")
    assert isinstance(formula, InteractionFormula)
    assert formula.terms == ["Y2"]


def test_parse_formula_missing_formula_returns_default() -> None:
    ia = Interaction()
    formula = ia.parse_formula("unknown")
    assert formula.is_default() is True
    assert formula.media_name == "unknown"


def test_get_all_interaction_terms_returns_union() -> None:
    ia = Interaction(formulas={"Y1": "1 + Y3", "Y2": "1 + C", "Y3": "1", "C": "1"})
    assert ia.get_all_interaction_terms() == {"Y3", "C"}


def test_get_all_interaction_terms_empty() -> None:
    ia = Interaction()
    assert ia.get_all_interaction_terms() == set()


def test_get_all_interaction_terms_deduplicated_across_formulas() -> None:
    ia = Interaction(formulas={"Y1": "1 + Y3", "Y2": "1 + Y3", "Y3": "1"})
    assert ia.get_all_interaction_terms() == {"Y3"}


def test_get_default_controls() -> None:
    ia = Interaction(
        formulas={"Y1": "1 + C1", "Y2": "1"},
        controls=["C1", "C2"],
    )
    assert ia.get_default_controls() == ["C2"]


def test_get_default_controls_none_mentioned() -> None:
    ia = Interaction(formulas={"Y1": "1"}, controls=["C1", "C2"])
    assert ia.get_default_controls() == ["C1", "C2"]


def test_get_default_controls_all_mentioned() -> None:
    ia = Interaction(formulas={"Y1": "1 + C1 + C2"}, controls=["C1", "C2"])
    assert ia.get_default_controls() == []


# ---------------------------------------------------------------------------
# Interaction — get_interaction_mode (boost vs product)
# ---------------------------------------------------------------------------


def test_interaction_mode_defaults_to_product_for_media() -> None:
    ia = Interaction(formulas={"Y1": "1 + Y2", "Y2": "1"}, media=["Y1", "Y2"])
    assert ia.get_interaction_mode("Y2") == "product"


def test_interaction_mode_defaults_to_boost_for_control() -> None:
    ia = Interaction(formulas={"Y1": "1 + C", "C": "1"}, media=["Y1"], controls=["C"])
    assert ia.get_interaction_mode("C") == "boost"


def test_interaction_mode_explicit_override_control_to_product() -> None:
    ia = Interaction(
        formulas={"Y1": "1 + C:product", "C": "1"}, media=["Y1"], controls=["C"]
    )
    assert ia.get_interaction_mode("C") == "product"


def test_interaction_mode_explicit_override_media_to_boost() -> None:
    ia = Interaction(formulas={"Y1": "1 + Y2:boost", "Y2": "1"}, media=["Y1", "Y2"])
    assert ia.get_interaction_mode("Y2") == "boost"


def test_interaction_mode_consistent_explicit_tags_across_formulas_ok() -> None:
    ia = Interaction(
        formulas={"Y1": "1 + Y3:product", "Y2": "1 + Y3:product", "Y3": "1"},
        media=["Y1", "Y2"],
    )
    assert ia.get_interaction_mode("Y3") == "product"


def test_interaction_mode_missing_explicit_tag_elsewhere_raises() -> None:
    with pytest.raises(ValueError, match="Add the tag in: \\['Y2'\\]"):
        Interaction(
            formulas={"Y1": "1 + Y3:product", "Y2": "1 + Y3", "Y3": "1"},
            media=["Y1", "Y2"],
        )


def test_interaction_mode_conflicting_explicit_tags_raises() -> None:
    with pytest.raises(ValueError, match="conflicting explicit modes"):
        Interaction(
            formulas={"Y1": "1 + Y3:product", "Y2": "1 + Y3:boost", "Y3": "1"},
            media=["Y1", "Y2"],
        )


def test_interaction_mode_only_used_in_one_formula_no_conflict_possible() -> None:
    ia = Interaction(formulas={"Y1": "1 + Y3:boost", "Y3": "1"}, media=["Y1"])
    assert ia.get_interaction_mode("Y3") == "boost"


def test_interaction_mode_implicit_warns() -> None:
    with pytest.warns(UserWarning, match="'C' \\(inferred: 'boost'\\)"):
        Interaction(formulas={"Y1": "1 + C", "C": "1"}, media=["Y1"], controls=["C"])


def test_interaction_mode_explicit_everywhere_does_not_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Interaction(
            formulas={"Y1": "1 + C:boost", "C": "1"}, media=["Y1"], controls=["C"]
        )


def test_interaction_mode_no_interaction_terms_does_not_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Interaction(formulas={"Y1": "1"}, media=["Y1"])


# ---------------------------------------------------------------------------
# InteractionCoordinates — get_parameter_name / resolve_shared_groups
# ---------------------------------------------------------------------------


def test_get_parameter_name_no_sharing() -> None:
    ia = Interaction(formulas={"Y1": "1 + Y3", "Y3": "1"}, media=["Y1"])
    assert InteractionCoordinates(ia).get_parameter_name("Y3") == "beta_interaction_Y3"


def test_get_parameter_name_is_independent_of_sharing() -> None:
    ia = Interaction(
        formulas={"Y1": "1 + Y3", "Y2": "1 + Y3", "Y3": "1"},
        is_shared_with=[("Y3", "Y1", "Y2")],
        media=["Y1", "Y2"],
    )
    assert InteractionCoordinates(ia).get_parameter_name("Y3") == "beta_interaction_Y3"


def test_resolve_shared_groups_mixed() -> None:
    ia = Interaction(
        formulas={"Y1": "1 + Y3 + C", "Y2": "1 + Y3", "Y3": "1", "C": "1"},
        is_shared_with=[("Y3", "Y1", "Y2")],
        media=["Y1", "Y2"],
        controls=["C"],
    )
    result = InteractionCoordinates(ia).resolve_shared_groups("Y1")
    assert result == {"Y3": "beta_interaction_Y3", "C": "beta_interaction_C"}


def test_resolve_shared_groups_default_formula() -> None:
    ia = Interaction()
    result = InteractionCoordinates(ia).resolve_shared_groups("Y_missing")
    assert result == {}


# ---------------------------------------------------------------------------
# InteractionCoordinates — get_unique_parameter_names
# ---------------------------------------------------------------------------


def test_get_unique_parameter_names_without_sharing() -> None:
    ia = Interaction(formulas={"Y1": "1 + Y3", "Y2": "1 + Y3", "Y3": "1"})
    assert InteractionCoordinates(ia).get_unique_parameter_names() == {
        "beta_interaction_Y3"
    }


def test_get_unique_parameter_names_with_sharing_still_one_param() -> None:
    # Sharing does not change how many *distinct* parameters exist — only
    # how their LHS coordinate is grouped (see get_coords).
    ia = Interaction(
        formulas={"Y1": "1 + Y3", "Y2": "1 + Y3", "Y3": "1"},
        is_shared_with=[("Y3", "Y1", "Y2")],
        media=["Y1", "Y2"],
    )
    assert InteractionCoordinates(ia).get_unique_parameter_names() == {
        "beta_interaction_Y3"
    }


# ---------------------------------------------------------------------------
# InteractionCoordinates — get_lhs_index / get_coords
# ---------------------------------------------------------------------------


def test_get_lhs_index_shared() -> None:
    ia = Interaction(
        formulas={"Y1": "1 + Y3", "Y2": "1 + Y3", "Y3": "1"},
        is_shared_with=[("Y3", "Y1", "Y2")],
        media=["Y1", "Y2"],
    )
    assert InteractionCoordinates(ia).get_lhs_index("Y1", "Y3") == 0


def test_get_lhs_index_unshared() -> None:
    ia = Interaction(
        formulas={"Y1": "1 + Y3", "Y2": "1 + Y3", "Y3": "1"},
        media=["Y1", "Y2"],
    )
    assert InteractionCoordinates(ia).get_lhs_index("Y2", "Y3") == 1


def test_get_coords_user_example() -> None:
    """The canonical scenario: TV shared between SEA and Digital, other
    interaction terms are per-media."""
    ia = Interaction(
        formulas={
            "TV": "1",
            "SEA": "1 + TV + Cospirit",
            "Digital": "0 + TV + Cospirit + Concurence",
            "Cospirit": "1",
            "Concurence": "0",
        },
        is_shared_with=[("TV", "Digital", "SEA")],
        media=["TV", "SEA", "Digital"],
        controls=["Cospirit", "Concurence", "trend"],
    )
    coords = InteractionCoordinates(ia).get_coords()
    assert coords["interaction_TV"] == ["Digital,SEA"]  # shared → single component
    assert coords["interaction_Cospirit"] == ["Digital", "SEA"]  # not shared
    assert coords["interaction_Concurence"] == ["Digital"]  # only Digital has it


def test_get_coords_no_interactions() -> None:
    ia = Interaction(formulas={"TV": "1", "SEA": "1"})
    assert InteractionCoordinates(ia).get_coords() == {}


def test_get_coords_empty() -> None:
    ia = Interaction()
    assert InteractionCoordinates(ia).get_coords() == {}


def test_get_coords_all_shared() -> None:
    ia = Interaction(
        formulas={"A": "1 + Z", "B": "1 + Z", "Z": "1"},
        is_shared_with=[("Z", "A", "B")],
    )
    coords = InteractionCoordinates(ia).get_coords()
    assert coords["interaction_Z"] == ["A,B"]
