"""Pytest unit tests for the Interaction dataclass."""

import pytest

from mmm_utils.modeling import Interaction, InteractionFormula

# pylint: skip-file


def test_get_parameter_name_no_sharing() -> None:
    ia = Interaction(formulas={"Y1": "1 + Y3", "Y3": "1"})
    assert ia.get_parameter_name("Y1", "Y3") == "Y1:Y3"


def test_get_parameter_name_with_sharing() -> None:
    ia = Interaction(
        formulas={"Y1": "1 + Y3", "Y2": "1 + Y3", "Y3": "1"},
        is_shared_with=[("Y3", "Y1", "Y2")],
    )
    assert ia.get_parameter_name("Y1", "Y3") == "Y1,Y2:Y3"
    assert ia.get_parameter_name("Y2", "Y3") == "Y1,Y2:Y3"


def test_get_parameter_name_unshared_term_uses_individual_name() -> None:
    # Y3 is shared between Y1 and Y2, but C is not — each media keeps its own
    ia = Interaction(
        formulas={"Y1": "1 + Y3 + C", "Y2": "1 + Y3 + C", "Y3": "1", "C": "1"},
        is_shared_with=[("Y3", "Y1", "Y2")],
    )
    assert ia.get_parameter_name("Y1", "Y3") == "Y1,Y2:Y3"
    assert ia.get_parameter_name("Y2", "Y3") == "Y1,Y2:Y3"
    assert ia.get_parameter_name("Y1", "C") == "Y1:C"
    assert ia.get_parameter_name("Y2", "C") == "Y2:C"


def test_get_parameter_name_shared_name_uses_sorted_media_names() -> None:
    ia = Interaction(
        formulas={"B": "1 + Z", "A": "1 + Z", "Z": "1"},
        is_shared_with=[("Z", "B", "A")],
    )
    # Sorted alphabetically → A,B
    assert ia.get_parameter_name("A", "Z") == "A,B:Z"
    assert ia.get_parameter_name("B", "Z") == "A,B:Z"


def test_resolve_shared_groups_mixed() -> None:
    # Y3 is shared, C is not — each channel gets its own C parameter
    ia = Interaction(
        formulas={"Y1": "1 + Y3 + C", "Y2": "1 + Y3 + C", "Y3": "1", "C": "1"},
        is_shared_with=[("Y3", "Y1", "Y2")],
    )
    result_y1 = ia.resolve_shared_groups("Y1")
    assert result_y1["Y3"] == "Y1,Y2:Y3"
    assert result_y1["C"] == "Y1:C"
    result_y2 = ia.resolve_shared_groups("Y2")
    assert result_y2["Y3"] == "Y1,Y2:Y3"
    assert result_y2["C"] == "Y2:C"


def test_resolve_shared_groups_default_formula() -> None:
    ia = Interaction()
    result = ia.resolve_shared_groups("Y_missing")
    assert result == {}


def test_get_all_interaction_terms_returns_union() -> None:
    ia = Interaction(formulas={"Y1": "1 + Y3", "Y2": "1 + C", "Y3": "1", "C": "1"})
    assert ia.get_all_interaction_terms() == {"Y3", "C"}


def test_get_all_interaction_terms_empty() -> None:
    ia = Interaction()
    assert ia.get_all_interaction_terms() == set()


def test_get_all_interaction_terms_deduplicated() -> None:
    ia = Interaction(formulas={"Y1": "1 + Y3", "Y2": "1 + Y3", "Y3": "1"})
    assert ia.get_all_interaction_terms() == {"Y3"}


def test_get_unique_parameter_names_without_sharing() -> None:
    ia = Interaction(formulas={"Y1": "1 + Y3", "Y2": "1 + Y3", "Y3": "1"})
    assert ia.get_unique_parameter_names() == {"Y1:Y3", "Y2:Y3"}


def test_get_unique_parameter_names_with_sharing_collapses_to_one() -> None:
    ia = Interaction(
        formulas={"Y1": "1 + Y3", "Y2": "1 + Y3", "Y3": "1"},
        is_shared_with=[("Y3", "Y1", "Y2")],
    )
    assert ia.get_unique_parameter_names() == {"Y1,Y2:Y3"}


def test_validation_sharing_group_too_small_raises() -> None:
    with pytest.raises(ValueError):
        Interaction(
            formulas={"Y1": "1 + Y3"},
            is_shared_with=[("Y3", "Y1")],  # only one media, needs ≥2
        )


def test_validation_sharing_group_media_without_explicit_formula_raises() -> None:
    # Y2 has no explicit formula, so Y3 cannot be verified in it
    with pytest.raises(ValueError):
        Interaction(
            formulas={"Y1": "1 + Y3"},
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
            formulas={"Y1": "1 + Y3", "Y2": "1 + Y3"},
            is_shared_with=[("", "Y1", "Y2")],
        )


def test_user_example_tv_sea_digital() -> None:
    """Exact scenario from the user: TV interaction shared between SEA and
    Digital, Cospirit interaction is independent for each."""
    ia = Interaction(
        formulas={
            "TV": "1",
            "SEA": "1 + TV + Cospirit",
            "Digital": "1 + TV + Cospirit",
            "Cospirit": "1",
        },
        is_shared_with=[("TV", "Digital", "SEA")],
    )
    # TV interaction is shared → same parameter name for both
    assert ia.get_parameter_name("SEA", "TV") == "Digital,SEA:TV"
    assert ia.get_parameter_name("Digital", "TV") == "Digital,SEA:TV"
    # Cospirit interaction is NOT shared → each channel has its own
    assert ia.get_parameter_name("SEA", "Cospirit") == "SEA:Cospirit"
    assert ia.get_parameter_name("Digital", "Cospirit") == "Digital:Cospirit"
    # Three distinct parameters total
    assert ia.get_unique_parameter_names() == {
        "Digital,SEA:TV",
        "SEA:Cospirit",
        "Digital:Cospirit",
    }


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


# ---------------------------------------------------------------------------
# Tests for term-definition validation
# ---------------------------------------------------------------------------


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
# Tests for media baseline validation
# ---------------------------------------------------------------------------


def test_media_with_literal_baseline_ok() -> None:
    Interaction(
        formulas={"TV": "1", "SEA": "1 + TV", "TV_coef": "1"},
        media=["TV", "SEA"],
    )


def test_media_with_borrowed_baseline_ok() -> None:
    # SEA has no '1' but references TV (a media channel) → borrowed baseline
    Interaction(
        formulas={"TV": "1", "SEA": "TV"},
        media=["TV", "SEA"],
    )


def test_media_without_baseline_raises() -> None:
    # SEA only references Cospirit (a control var), no baseline available
    with pytest.raises(ValueError, match="Media channel 'SEA' has no baseline"):
        Interaction(
            formulas={"TV": "1", "SEA": "Cospirit", "Cospirit": "1"},
            media=["TV", "SEA"],
        )


def test_media_absent_from_formulas_gets_default_baseline() -> None:
    # TV not in formulas → defaults to "1" → passes baseline check
    Interaction(
        formulas={"SEA": "1 + TV"},
        media=["TV", "SEA"],
    )


def test_non_media_without_baseline_ok() -> None:
    # Cospirit is a control var — no baseline required
    Interaction(
        formulas={"TV": "1", "SEA": "1 + TV", "Cospirit": "TV"},
        media=["TV", "SEA"],
    )


def test_media_with_zero_plus_media_term_ok() -> None:
    # "0 + TV" is an explicit no-baseline formula; TV is media → borrowed baseline ok
    Interaction(
        formulas={"TV": "1", "SEA": "0 + TV"},
        media=["TV", "SEA"],
    )


def test_media_with_zero_only_raises() -> None:
    # "0" means no baseline, no terms → invalid for a media channel
    with pytest.raises(ValueError, match="Media channel 'SEA' has no baseline"):
        Interaction(
            formulas={"TV": "1", "SEA": "0"},
            media=["TV", "SEA"],
        )


def test_full_user_example() -> None:
    """TV/SEA/Digital/Cospirit scenario from the feature request."""
    ia = Interaction(
        formulas={
            "TV": "1",
            "SEA": "1 + TV + Cospirit",
            "Digital": "1 + TV + Cospirit",
            "Cospirit": "1",
        },
        media=["TV", "SEA", "Digital"],
        is_shared_with=[("TV", "Digital", "SEA")],
    )
    assert ia.get_parameter_name("SEA", "TV") == "Digital,SEA:TV"
    assert ia.get_unique_parameter_names() == {
        "Digital,SEA:TV",
        "SEA:Cospirit",
        "Digital:Cospirit",
    }


def test_full_user_example_missing_cospirit_raises() -> None:
    with pytest.raises(ValueError, match="undefined variable 'Cospirit'"):
        Interaction(
            formulas={
                "TV": "1",
                "SEA": "1 + TV + Cospirit",
                "Digital": "1 + TV + Cospirit",
            },
            media=["TV", "SEA", "Digital"],
        )


# ---------------------------------------------------------------------------
# Tests for get_coords
# ---------------------------------------------------------------------------


def test_get_coords_user_example() -> None:
    """The canonical scenario from the feature request.
    No media arg → no baseline_media; interaction keys are prefixed interaction_."""
    ia = Interaction(
        formulas={
            "TV": "1",
            "SEA": "1 + TV + Cospirit",
            "Digital": "1 + TV + Cospirit + Concurence",
            "Cospirit": "1",
            "Concurence": "0",
        },
        is_shared_with=[("TV", "Digital", "SEA")],
    )
    coords = ia.get_coords()
    assert coords["interaction_TV"] == ["Digital,SEA"]  # shared → single component
    assert coords["interaction_Cospirit"] == [
        "Digital",
        "SEA",
    ]  # not shared → per-media
    assert coords["interaction_Concurence"] == ["Digital"]  # only Digital has it
    assert "baseline_media" not in coords  # no media arg → baseline filter empty


def test_get_coords_with_media_baseline() -> None:
    """When media is provided, baseline_media lists the media channels with baseline."""
    ia = Interaction(
        formulas={
            "TV": "1",
            "SEA": "1 + TV + Cospirit",
            "Digital": "1 + TV + Cospirit + Concurence",
            "Cospirit": "1",
            "Concurence": "0",
        },
        media=["TV", "SEA", "Digital"],
        is_shared_with=[("TV", "Digital", "SEA")],
    )
    coords = ia.get_coords()
    # Only media channels with has_baseline=True appear here
    assert coords["baseline_media"] == ["Digital", "SEA", "TV"]
    # Concurence has "0" formula → not in baseline
    assert "Concurence" not in coords["baseline_media"]


def test_get_coords_no_interactions() -> None:
    # No media arg and no interaction terms → empty coords
    ia = Interaction(formulas={"TV": "1", "SEA": "1"})
    coords = ia.get_coords()
    assert coords == {}


def test_get_coords_empty() -> None:
    ia = Interaction()
    coords = ia.get_coords()
    assert coords == {}


def test_get_coords_all_shared() -> None:
    ia = Interaction(
        formulas={"A": "1 + Z", "B": "1 + Z", "Z": "1"},
        is_shared_with=[("Z", "A", "B")],
    )
    coords = ia.get_coords()
    assert coords["interaction_Z"] == ["A,B"]
    assert "baseline_media" not in coords  # no media arg


def test_get_coords_media_absent_from_formulas_in_baseline() -> None:
    # TV is in media but not in formulas → defaults to "1" → appears in baseline_media
    ia = Interaction(
        formulas={"SEA": "1 + TV"},
        media=["TV", "SEA"],
    )
    coords = ia.get_coords()
    assert "TV" in coords["baseline_media"]
    assert "SEA" in coords["baseline_media"]
