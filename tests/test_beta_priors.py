import numpy as np
import pytest

import pymc as pm
import pymc.dims as pmd

from mmm_utils.modeling import BetaPriors
from mmm_utils.modeling import Interaction
from mmm_utils.modeling import InteractionCoordinates
from mmm_utils.modeling import PriorSpec

# pylint: skip-file


# ---------------------------------------------------------------------------
# BetaPriors — controls_without_effect
# ---------------------------------------------------------------------------


def test_controls_without_effect_unknown_control_raises():
    ia = Interaction(
        formulas={"Y1": "1 + Promo:boost"}, media=["Y1"], controls=["Promo"]
    )
    with pytest.raises(ValueError, match="unknown control"):
        BetaPriors(
            interaction=ia,
            priors={"beta_interaction_Promo": PriorSpec("HalfNormal", {"sigma": 1.0})},
            controls_without_effect=["NotAControl"],
        )


def test_controls_without_effect_unreferenced_control_warns():
    ia = Interaction(formulas={"Y1": "1"}, media=["Y1"], controls=["Promo"])
    with pytest.warns(UserWarning, match="no effect on the model"):
        BetaPriors(interaction=ia, controls_without_effect=["Promo"])


def test_get_control_own_effect_names_excludes_listed_control():
    ia = Interaction(
        formulas={"Y1": "1 + Promo:boost"},
        media=["Y1"],
        controls=["Promo", "trend"],
    )
    bp = BetaPriors(
        interaction=ia,
        priors={"beta_interaction_Promo": PriorSpec("HalfNormal", {"sigma": 1.0})},
        controls_without_effect=["Promo"],
    )
    assert bp.get_control_own_effect_names() == ["trend"]


def test_get_control_own_effect_names_defaults_to_all_controls():
    ia = Interaction(formulas={"Y1": "1"}, media=["Y1"], controls=["Promo", "trend"])
    bp = BetaPriors(interaction=ia)
    assert bp.get_control_own_effect_names() == ["Promo", "trend"]


# ---------------------------------------------------------------------------
# BetaPriors — control prior array shape vs control_active
# ---------------------------------------------------------------------------


def test_control_array_prior_sized_for_all_controls_raises():
    # 3 controls, 1 excluded -> control_active has 2 entries, not 3
    ia = Interaction(
        formulas={"Y1": "1 + Promo:boost"},
        media=["Y1"],
        controls=["Promo", "trend", "c3"],
    )
    with pytest.raises(ValueError, match="control.params\\['mu'\\] has length 3"):
        BetaPriors(
            interaction=ia,
            priors={"beta_interaction_Promo": PriorSpec("HalfNormal", {"sigma": 1.0})},
            controls_without_effect=["Promo"],
            control=PriorSpec("Normal", {"mu": np.zeros(3), "sigma": 1.0}),
        )


def test_control_array_prior_sized_for_control_active_ok():
    ia = Interaction(
        formulas={"Y1": "1 + Promo:boost"},
        media=["Y1"],
        controls=["Promo", "trend", "c3"],
    )
    bp = BetaPriors(
        interaction=ia,
        priors={"beta_interaction_Promo": PriorSpec("HalfNormal", {"sigma": 1.0})},
        controls_without_effect=["Promo"],
        control=PriorSpec("Normal", {"mu": np.zeros(2), "sigma": np.ones(2)}),
    )
    assert bp.get_control_own_effect_names() == ["trend", "c3"]


def test_control_scalar_prior_always_ok():
    ia = Interaction(
        formulas={"Y1": "1 + Promo:boost"},
        media=["Y1"],
        controls=["Promo", "trend", "c3"],
    )
    BetaPriors(
        interaction=ia,
        priors={"beta_interaction_Promo": PriorSpec("HalfNormal", {"sigma": 1.0})},
        controls_without_effect=["Promo"],
        control=PriorSpec("Normal", {"mu": 0.0, "sigma": 1.0}),
    )


def test_excluded_control_has_zero_standalone_beta_but_keeps_interaction_effect():
    n = 30
    rng = np.random.default_rng(0)
    ia = Interaction(
        formulas={"Y1": "1 + Promo:boost"},
        media=["Y1"],
        controls=["Promo", "trend"],
    )
    bp = BetaPriors(
        interaction=ia,
        priors={"beta_interaction_Promo": PriorSpec("HalfNormal", {"sigma": 1.0})},
        controls_without_effect=["Promo"],
    )

    coords = {
        "date": np.arange(n),
        "media": ["Y1"],
        "control": ["Promo", "trend"],
        "control_active": bp.get_control_own_effect_names(),
        "season": ["sin[1]", "cos[1]"],
    } | InteractionCoordinates(ia).get_coords()

    with pm.Model(coords=coords) as model:
        x_m = pmd.Data("media_data", rng.random((n, 1)), dims=("date", "media"))
        x_c = pmd.Data("control_data", rng.random((n, 2)), dims=("date", "control"))
        bp.build_pymc_priors()
        result = bp.get_beta_adjusted(x_m, x_c)

    values = pm.draw(result["control"].values, random_seed=0)
    promo_idx = coords["control"].index("Promo")
    trend_idx = coords["control"].index("trend")
    assert values[promo_idx] == 0.0
    assert values[trend_idx] != 0.0


if __name__ == "__main__":
    ia = Interaction(
        formulas={
            "TV": "1",
            "SEA": "1 + TV + Cospirit",
            "Digital": "1 + TV + Cospirit + Concurence",
            "Cospirit": "1",
            "Concurence": "0",
        },
        is_shared_with=[("TV", "Digital", "SEA")],
        media=["TV", "SEA", "Digital"],
        controls=["Cospirit", "Concurence", "trend"],
    )
    ia_coords = InteractionCoordinates(ia)
    print(ia_coords.get_unique_parameter_names())
    print(ia_coords.get_coords())

    prior = PriorSpec(
        "TruncatedNormal",
        params={"mu": 0, "sigma": 0.1, "lower": -1, "upper": 1},
    )

    x = BetaPriors(
        interaction=ia,
        priors={
            "beta_interaction_TV": prior,
            "beta_interaction_Cospirit": prior,
            "beta_interaction_Concurence": prior,
        },
    )

    coords = {
        "date": np.arange(100),
        "media": ["TV", "SEA", "Digital"],
        "control": ["Cospirit", "Concurence", "trend"],
        "season": ["sin[1]", "cos[1]", "sin[2]", "cos[2]"],
    } | ia_coords.get_coords()
    with pm.Model(coords=coords) as model:
        x_m = pmd.Data(
            "media_data",
            np.random.rand(100, 3),
            dims=("date", "media"),
        )
        x_c = pmd.Data(
            "control_data",
            np.random.rand(100, 3),
            dims=("date", "control"),
        )
        x.build_pymc_priors(coords)

    beta_adjusted_media = x.get_beta_adjusted(x_m, x_c)[0]

    # beta_adjusted_media = [\beta_adjusted_TV, \beta_adjusted_SEA, \beta_adjusted_Digital]
    # beta_adjusted_TV = \beta_TV
    # beta_adjusted_SEA = \beta_SEA (1 + \beta_interaction_TV * TV + \beta_interaction_Cospirit * Cospirit)
    print(beta_adjusted_media.owner.dprint(depth=10))
