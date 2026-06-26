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


def test_no_baseline() -> None:
    parsed_formula = InteractionFormula(media_name="Y1", raw="Y2 + C")
    assert parsed_formula.terms == ["Y2", "C"]
    assert parsed_formula.has_baseline is False
    assert parsed_formula.is_default() is False


def test_duplicate_terms_preserved() -> None:
    parsed_formula = InteractionFormula(media_name="Y1", raw="1 + Y2 + Y2")
    assert parsed_formula.terms == ["Y2", "Y2"]


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


def test_zero_token_no_baseline() -> None:
    parsed_formula = InteractionFormula(media_name="Y1", raw="0")
    assert parsed_formula.terms == []
    assert parsed_formula.has_baseline is False
    assert parsed_formula.is_default() is False


def test_zero_plus_term() -> None:
    parsed_formula = InteractionFormula(media_name="Y1", raw="0 + TV")
    assert parsed_formula.terms == ["TV"]
    assert parsed_formula.has_baseline is False
