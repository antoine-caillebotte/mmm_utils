import numpy as np
import pandas as pd
import pytest

import pymc as pm

from mmm_utils.modeling import BetaPriors, Interaction, MMM, MMMConfig, PriorSpec
from mmm_utils.modeling.model_definition.roas_priors import RoasPriors

# pylint: skip-file

pytestmark = pytest.mark.filterwarnings(
    "ignore:media_names contains",
    "ignore::FutureWarning",
    "ignore::UserWarning",
)

MEDIA = ["TV", "SEA", "Radio"]
# TV also carries a boost interaction, SEA keeps the default beta prior
ROAS_RANGES = {"TV": (0.5, 2.0), "Radio": (3.0, 6.0)}
N_DRAWS = 1000
SEED = 1


def _data(n=52, zero_radio=False):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="W"),
            "TV": rng.uniform(0, 5000, n),
            "SEA": rng.uniform(0, 800, n),
            "Radio": np.zeros(n) if zero_radio else rng.uniform(0, 300, n),
            "Promo": rng.uniform(0, 1, n),
        }
    )
    y = pd.Series(rng.uniform(1e4, 3e4, n))
    return X, y


def _build(roas_priors=None, zero_radio=False):
    ia = Interaction(
        formulas={"TV": "1 + Promo:boost"}, media=MEDIA, controls=["Promo"]
    )
    bp = BetaPriors(
        interaction=ia,
        priors={"beta_interaction_Promo": PriorSpec("HalfNormal", {"sigma": 0.5})},
    )
    kwargs = {} if roas_priors is None else {"roas_priors": roas_priors}
    mmm = MMM(
        MMMConfig(beta_priors=bp, media_names=MEDIA, control_names=["Promo"], **kwargs)
    )
    mmm.build(*_data(zero_radio=zero_radio))
    return mmm


def _prior(model, var_names, draws=N_DRAWS):
    with model:
        return pm.sample_prior_predictive(
            draws=draws, random_seed=SEED, var_names=var_names
        ).prior


@pytest.fixture(scope="module")
def roas_mmm():
    return _build(RoasPriors(ROAS_RANGES, coverage=0.9))


@pytest.fixture(scope="module")
def roas_prior(roas_mmm):
    return _prior(
        roas_mmm.model, ["roas", "roas_media", "beta_media", "beta_media_free"]
    )


# ---------------------------------------------------------------------------
# Prior predictive — implied ROAS falls in the requested ranges
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("channel", list(ROAS_RANGES))
def test_prior_predictive_roas_matches_range(roas_prior, channel):
    """5 % / 95 % quantiles of the implied ROAS recover (a, b) at 90 % coverage."""
    a, b = ROAS_RANGES[channel]
    roas = roas_prior["roas"].sel(media=channel).values.ravel()

    q_low, q_high = np.quantile(roas, [0.05, 0.95])
    # Monte-Carlo tolerance on tail quantiles with N_DRAWS draws
    np.testing.assert_allclose([q_low, q_high], [a, b], rtol=0.1)
    assert 0.85 <= np.mean((roas >= a) & (roas <= b)) <= 0.95


@pytest.mark.parametrize("channel", list(ROAS_RANGES))
def test_implied_roas_equals_roas_prior(roas_prior, channel):
    """Recomputed contribution / spend equals the sampled ROI, draw by draw."""
    np.testing.assert_allclose(
        roas_prior["roas"].sel(media=channel).values,
        roas_prior["roas_media"].sel(media_roas=channel).values,
        rtol=1e-8,
    )


def test_channel_without_roas_keeps_free_beta(roas_prior):
    np.testing.assert_array_equal(
        roas_prior["beta_media"].sel(media="SEA").values,
        roas_prior["beta_media_free"].sel(media_free="SEA").values,
    )


# ---------------------------------------------------------------------------
# Graph structure
# ---------------------------------------------------------------------------


def test_roas_model_free_rvs(roas_mmm):
    free_rvs = {v.name for v in roas_mmm.model.free_RVs}
    assert {"roas_media", "beta_media_free"} <= free_rvs
    assert "beta_media" not in free_rvs
    assert "beta_media" in roas_mmm.model.named_vars
    assert list(roas_mmm.model.coords["media_roas"]) == ["TV", "Radio"]
    assert list(roas_mmm.model.coords["media_free"]) == ["SEA"]


def test_without_roas_priors_model_is_unchanged():
    mmm = _build()
    free_rvs = {v.name for v in mmm.model.free_RVs}
    assert "beta_media" in free_rvs
    assert "roas_media" not in mmm.model.named_vars
    assert "beta_media_free" not in mmm.model.named_vars


# ---------------------------------------------------------------------------
# Counterfactuals — ROAS-derived betas are pinned to the training data
# ---------------------------------------------------------------------------


def test_beta_media_invariant_under_channel_data_swap(roas_mmm):
    var_names = ["beta_media", "roas_media", "beta_media_free"]
    swapped = pm.do(roas_mmm.model, {"channel_data": roas_mmm.data.X_media * 3.0})
    ref = _prior(roas_mmm.model, var_names, draws=50)
    new = _prior(swapped, var_names, draws=50)
    np.testing.assert_allclose(ref["beta_media"].values, new["beta_media"].values)


# ---------------------------------------------------------------------------
# Input validation at build time
# ---------------------------------------------------------------------------


def test_roas_channel_without_spend_raises():
    with pytest.raises(ValueError, match="no positive spend"):
        _build(RoasPriors(ROAS_RANGES), zero_radio=True)


def test_media_array_prior_is_restricted_to_free_channels():
    bp = BetaPriors(
        interaction=Interaction(media=MEDIA),
        media=PriorSpec("HalfNormal", {"sigma": np.array([1.0, 2.0, 3.0])}),
    )
    spec = bp.media_spec_for(["SEA"])
    np.testing.assert_array_equal(spec.params["sigma"], [2.0])


def test_media_array_prior_with_wrong_length_raises():
    bp = BetaPriors(
        interaction=Interaction(media=MEDIA),
        media=PriorSpec("HalfNormal", {"sigma": np.array([1.0, 2.0])}),
    )
    with pytest.raises(ValueError, match="one value per media channel"):
        bp.media_spec_for(["SEA"])


# ---------------------------------------------------------------------------
# Prior plotting — ROAS variables resolve to their prior specification
# ---------------------------------------------------------------------------

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mmm_utils.modeling.prior import (
    _resolve_prior_spec_for_var,
    plot_prior_vs_posterior,
)


@pytest.mark.parametrize("var", ["roas_media", "roas"])
def test_resolve_roas_prior_spec(roas_mmm, var):
    spec = _resolve_prior_spec_for_var(roas_mmm, var, "Radio")
    expected = roas_mmm.config.roas_priors.to_prior_spec(["Radio"])
    assert spec.kind == "LogNormal"
    np.testing.assert_allclose(spec.params["mu"], expected.params["mu"])
    np.testing.assert_allclose(spec.params["sigma"], expected.params["sigma"])


def test_resolve_beta_media_free_prior_spec(roas_mmm):
    spec = _resolve_prior_spec_for_var(roas_mmm, "beta_media_free", "SEA")
    assert spec.kind == roas_mmm.config.beta_priors.media.kind


@pytest.mark.parametrize(
    "var, media, expected",
    [("beta_media", "TV", None), ("roas", "SEA", None)],
)
def test_resolve_deterministic_prior_spec_is_none(roas_mmm, var, media, expected):
    assert _resolve_prior_spec_for_var(roas_mmm, var, media) is expected


def test_resolve_beta_media_of_free_channel_keeps_media_prior(roas_mmm):
    spec = _resolve_prior_spec_for_var(roas_mmm, "beta_media", "SEA")
    assert spec is roas_mmm.config.beta_priors.media


def test_resolve_roas_media_unknown_channel_raises(roas_mmm):
    with pytest.raises(ValueError, match="no ROAS prior"):
        _resolve_prior_spec_for_var(roas_mmm, "roas_media", "SEA")


@pytest.fixture(scope="module")
def roas_mmm_with_idata(roas_mmm):
    with roas_mmm.model:
        idata = pm.sample_prior_predictive(
            draws=200,
            random_seed=SEED,
            var_names=["roas_media", "beta_media", "beta_media_free"],
        )
    # Prior draws stand in for the posterior: only the plotting path is tested
    idata["posterior"] = idata["prior"]
    roas_mmm.idata = idata
    return roas_mmm


def test_plot_prior_vs_posterior_roas_media(roas_mmm_with_idata):
    fig, _ = plot_prior_vs_posterior(
        roas_mmm_with_idata, "roas_media", list(ROAS_RANGES), separately=True
    )
    plt.close(fig)


def test_plot_prior_vs_posterior_beta_media_warns_for_roas_channels(
    roas_mmm_with_idata,
):
    with pytest.warns(UserWarning, match="deterministic for 'TV'"):
        fig, _ = plot_prior_vs_posterior(
            roas_mmm_with_idata, "beta_media", MEDIA, separately=True
        )
    plt.close(fig)
