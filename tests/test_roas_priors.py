import numpy as np
import pytest
from scipy.stats import lognorm

from mmm_utils.modeling.model_definition.roas_priors import RoasPriors

# pylint: skip-file


# ---------------------------------------------------------------------------
# RoasPriors — validation
# ---------------------------------------------------------------------------


def test_default_is_empty():
    rp = RoasPriors()
    assert rp.media_names == []
    assert rp.coverage == 0.9


@pytest.mark.parametrize("coverage", [0.0, 1.0, -0.1, 1.5])
def test_invalid_coverage_raises(coverage):
    with pytest.raises(ValueError, match="coverage"):
        RoasPriors({"TV": (1.0, 2.0)}, coverage=coverage)


@pytest.mark.parametrize(
    "bounds",
    [(2.0, 1.0), (1.0, 1.0), (0.0, 1.0), (-1.0, 2.0)],
)
def test_invalid_range_order_raises(bounds):
    with pytest.raises(ValueError, match="0 < a < b"):
        RoasPriors({"TV": bounds})


def test_non_finite_range_raises():
    with pytest.raises(ValueError, match="finite"):
        RoasPriors({"TV": (1.0, np.inf)})


def test_range_not_a_pair_raises():
    with pytest.raises(ValueError, match="pair"):
        RoasPriors({"TV": (1.0, 2.0, 3.0)})


# ---------------------------------------------------------------------------
# RoasPriors — LogNormal conversion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("coverage", [0.8, 0.9, 0.95])
def test_lognormal_quantiles_match_range(coverage):
    rp = RoasPriors({"TV": (0.5, 3.0), "SEA": (2.0, 8.0)}, coverage=coverage)
    mu, sigma = rp.lognormal_params()

    tail = (1.0 - coverage) / 2.0
    for i, name in enumerate(rp.media_names):
        dist = lognorm(s=sigma[i], scale=np.exp(mu[i]))
        a, b = rp.priors[name]
        np.testing.assert_allclose(dist.ppf([tail, 1.0 - tail]), [a, b], rtol=1e-10)


def test_lognormal_params_follow_requested_order():
    rp = RoasPriors({"TV": (1.0, 2.0), "SEA": (3.0, 9.0)})
    mu_default, _ = rp.lognormal_params()
    mu_reordered, _ = rp.lognormal_params(["SEA", "TV"])
    np.testing.assert_allclose(mu_reordered, mu_default[::-1])


def test_lognormal_params_unknown_channel_raises():
    rp = RoasPriors({"TV": (1.0, 2.0)})
    with pytest.raises(ValueError, match="No ROAS prior"):
        rp.lognormal_params(["TV", "Radio"])


def test_to_prior_spec():
    rp = RoasPriors({"TV": (1.0, 2.0), "SEA": (3.0, 9.0)})
    spec = rp.to_prior_spec()
    assert spec.kind == "LogNormal"
    assert spec.params["mu"].shape == (2,)
    assert spec.params["sigma"].shape == (2,)


# ---------------------------------------------------------------------------
# MMMConfig — roas_priors integration
# ---------------------------------------------------------------------------

from mmm_utils.modeling import BetaPriors, Interaction, MMMConfig, PriorSpec


def _config(roas_priors=None, formulas=None, controls=("Promo",)):
    ia = Interaction(
        formulas=formulas or {}, media=["TV", "SEA", "Radio"], controls=list(controls)
    )
    priors = {
        f"beta_interaction_{term}": PriorSpec("HalfNormal", {"sigma": 1.0})
        for term in ia.get_all_interaction_terms()
    }
    kwargs = {} if roas_priors is None else {"roas_priors": roas_priors}
    return MMMConfig(
        beta_priors=BetaPriors(interaction=ia, priors=priors),
        media_names=["TV", "SEA", "Radio"],
        control_names=list(controls),
        **kwargs,
    )


@pytest.mark.filterwarnings("ignore:media_names contains")
def test_config_default_has_no_roas_and_unchanged_var_names():
    cfg = _config()
    assert cfg.roas_priors.media_names == []
    assert cfg.media_names_with_roas == []
    assert cfg.media_names_without_roas == ["TV", "SEA", "Radio"]
    assert "roas_media" not in cfg.var_names
    assert "beta_media_free" not in cfg.var_names


@pytest.mark.filterwarnings("ignore:media_names contains")
def test_config_partitions_media_in_media_names_order():
    cfg = _config(RoasPriors({"Radio": (1.0, 3.0), "TV": (0.5, 2.0)}))
    assert cfg.media_names_with_roas == ["TV", "Radio"]
    assert cfg.media_names_without_roas == ["SEA"]
    assert {"beta_media", "roas_media", "beta_media_free"} <= set(cfg.var_names)


@pytest.mark.filterwarnings("ignore:media_names contains")
def test_config_all_media_with_roas_has_no_free_beta():
    cfg = _config(
        RoasPriors({"TV": (1.0, 2.0), "SEA": (1.0, 2.0), "Radio": (1.0, 2.0)})
    )
    assert "roas_media" in cfg.var_names
    assert "beta_media_free" not in cfg.var_names


@pytest.mark.filterwarnings("ignore:media_names contains")
def test_config_unknown_roas_channel_raises():
    with pytest.raises(ValueError, match="not in media_names"):
        _config(RoasPriors({"Cinema": (1.0, 2.0)}))


@pytest.mark.filterwarnings("ignore:media_names contains")
def test_config_roas_priors_wrong_type_raises():
    with pytest.raises(TypeError, match="RoasPriors"):
        _config({"TV": (1.0, 2.0)})


@pytest.mark.filterwarnings("ignore:media_names contains")
def test_config_roas_channel_with_product_term_raises():
    with pytest.raises(ValueError, match="product"):
        _config(RoasPriors({"TV": (1.0, 2.0)}), formulas={"TV": "1 + Promo:product"})


@pytest.mark.filterwarnings("ignore:media_names contains")
def test_config_roas_channel_with_boost_term_ok():
    cfg = _config(RoasPriors({"TV": (1.0, 2.0)}), formulas={"TV": "1 + Promo:boost"})
    assert cfg.media_names_with_roas == ["TV"]


@pytest.mark.filterwarnings("ignore:media_names contains")
def test_config_product_term_on_channel_without_roas_ok():
    cfg = _config(RoasPriors({"TV": (1.0, 2.0)}), formulas={"SEA": "1 + Promo:product"})
    assert cfg.media_names_without_roas == ["SEA", "Radio"]
