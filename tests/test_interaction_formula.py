"""Pytest unit tests for the InteractionFormula dataclass."""

import pytest

from mmm_utils.modeling import InteractionFormula

# pylint: skip-file


def test_default_formula() -> None:
    parsed_formula = InteractionFormula(media_name="Y", raw="1")
    assert parsed_formula.terms == []
    assert parsed_formula.has_baseline is True
    assert parsed_formula.is_default() is True


def test_single_interaction() -> None:
    parsed_formula = InteractionFormula(media_name="Y1", raw="1 + Y2")
    assert parsed_formula.terms == ["Y2"]
    assert parsed_formula.has_baseline is True
    assert parsed_formula.is_default() is False


def test_multiple_interactions() -> None:
    parsed_formula = InteractionFormula(media_name="Y1", raw="1 + Y2 + C")
    assert parsed_formula.terms == ["Y2", "C"]
    assert parsed_formula.has_baseline is True


def test_invalid_token_raises() -> None:
    with pytest.raises(ValueError):
        InteractionFormula(media_name="Y", raw="1 + 2bad")


def test_invalid_expression_raises() -> None:
    with pytest.raises(ValueError):
        InteractionFormula(media_name="Y", raw="1 + Y2 * C")


def test_empty_token_raises() -> None:
    with pytest.raises(ValueError):
        InteractionFormula(media_name="Y", raw="1 +  + C")


def test_whitespace_tolerance() -> None:
    parsed_formula = InteractionFormula(media_name="Y1", raw="  1  +  Y2  ")
    assert parsed_formula.terms == ["Y2"]
    assert parsed_formula.has_baseline is True


def test_missing_baseline_marker_raises() -> None:
    # Neither "1" nor "0" is present — the formula must declare one explicitly.
    with pytest.raises(ValueError, match="has no baseline"):
        InteractionFormula(media_name="Y1", raw="Y2 + C")


# ---------------------------------------------------------------------------
# Tests for the "0" (explicit no-baseline) token
# ---------------------------------------------------------------------------


def test_zero_token_no_baseline() -> None:
    parsed_formula = InteractionFormula(media_name="Y1", raw="0")
    assert parsed_formula.terms == []
    assert parsed_formula.has_baseline is False
    assert parsed_formula.is_default() is False


def test_zero_plus_term() -> None:
    parsed_formula = InteractionFormula(media_name="Y1", raw="0 + TV")
    assert parsed_formula.terms == ["TV"]
    assert parsed_formula.has_baseline is False


def test_zero_and_one_conflict_raises() -> None:
    with pytest.raises(ValueError, match="cannot contain both '1' and '0'"):
        InteractionFormula(media_name="Y1", raw="1 + 0")


# ---------------------------------------------------------------------------
# Tests for duplicate terms
# ---------------------------------------------------------------------------


def test_duplicate_term_raises() -> None:
    with pytest.raises(ValueError, match="Duplicate term 'Y2'"):
        InteractionFormula(media_name="Y1", raw="1 + Y2 + Y2")


# ---------------------------------------------------------------------------
# Tests for the ":boost"/":product" mode suffix
# ---------------------------------------------------------------------------


def test_term_without_mode_suffix_has_no_entry() -> None:
    parsed_formula = InteractionFormula(media_name="Y1", raw="1 + Y2")
    assert parsed_formula.term_modes == {}


def test_term_with_boost_suffix() -> None:
    parsed_formula = InteractionFormula(media_name="Y1", raw="1 + C:boost")
    assert parsed_formula.terms == ["C"]
    assert parsed_formula.term_modes == {"C": "boost"}


def test_term_with_product_suffix() -> None:
    parsed_formula = InteractionFormula(media_name="Y1", raw="1 + C:product")
    assert parsed_formula.terms == ["C"]
    assert parsed_formula.term_modes == {"C": "product"}


def test_term_with_mode_suffix_and_whitespace_tolerance() -> None:
    parsed_formula = InteractionFormula(media_name="Y1", raw="1 +  C : boost ")
    assert parsed_formula.terms == ["C"]
    assert parsed_formula.term_modes == {"C": "boost"}


def test_mixed_suffixed_and_plain_terms() -> None:
    parsed_formula = InteractionFormula(media_name="Y1", raw="1 + Y2 + C:boost")
    assert parsed_formula.terms == ["Y2", "C"]
    assert parsed_formula.term_modes == {"C": "boost"}


def test_invalid_mode_suffix_raises() -> None:
    with pytest.raises(ValueError, match="Invalid interaction mode 'weird'"):
        InteractionFormula(media_name="Y1", raw="1 + C:weird")


def test_duplicate_term_with_mode_suffix_still_raises() -> None:
    with pytest.raises(ValueError, match="Duplicate term 'C'"):
        InteractionFormula(media_name="Y1", raw="1 + C:boost + C:product")
