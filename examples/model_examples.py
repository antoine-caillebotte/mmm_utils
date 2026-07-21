# pylint:skip-file
import numpy as np

from mmm_utils.modeling import MMM, MMMConfig, MediaTransformSpec
from mmm_utils.modeling.prior import PriorSpec, plot_prior_vs_posterior
from mmm_utils.modeling import Interaction, BetaPriors


def get_builded_mmm(X, y, media, controls):
    """Build an MMM model with a given configuration and fit it to the data.

    Parameters
    ----------
    X : pd.DataFrame
        Input features (media and controls).
    y : pd.Series
        Target variable.
    media : list[str]
        List of media channel names.
    controls : list[str]
        List of control variable names.

    Returns
    -------
    mmm : MMM
        The fitted MMM model.
    """
    interaction = Interaction(
        formulas={
            "TV": "1",
            "SEA": "1",  # + TV",
            "Digital": "1",  # + TV",
        },
        media=media,
        # is_shared_with=[("TV", "SEA", "Digital")],
    )
    print("Priors that need to be defined :")
    interaction.get_unique_parameter_names()

    beta_priors = BetaPriors(
        interaction=interaction,
        priors={},  # "beta_interaction_TV": PriorSpec("HalfNormal", {"sigma": 0.5})},
        season=PriorSpec("Laplace", {"mu": 0.0, "b": np.array([0.5, 0.5, 0.1, 0.1])}),
        control=PriorSpec("Normal", {"mu": 1.0, "sigma": 9.1}),
    )

    cfg = MMMConfig(
        media_names=media,
        control_names=controls,
        seasonality_order=2,
        beta_priors=beta_priors,
        media_transforms={
            "TV": MediaTransformSpec(
                adstock="Geometric",
                adstock_params={"l_max": 12 * 6},
                adstock_priors={
                    "alpha": PriorSpec("Beta", {"alpha": 4.0, "beta": 0.5}),
                    # "theta": PriorSpec("Normal", {"mu": 2, "sigma": 0.2}),
                },
                saturation="Logistic",
                saturation_params={"lam": 3.0},
                saturation_priors={
                    # "lam": PriorSpec("LogNormal", {"mu": 0.0, "sigma": 1.0})
                },
            ),
            "SEA": MediaTransformSpec(
                adstock="Geometric",
                adstock_params={"l_max": 12, "alpha": 0.0},
                adstock_priors={},  # "alpha": PriorSpec("Beta", {"alpha": 2.0, "beta": 2.0})},
                saturation="Logistic",
                saturation_params={"lam": 1.0},
                # saturation_priors={
                #     "lam": PriorSpec("LogNormal", {"mu": 0.0, "sigma": 1.0})
                # },
            ),
            "Digital": MediaTransformSpec(
                adstock="Geometric",
                adstock_params={"l_max": 12},
                adstock_priors={
                    "alpha": PriorSpec("Beta", {"alpha": 4.0, "beta": 0.5}),
                },
                saturation="Logistic",
                saturation_params={"lam": 1.0},
                saturation_priors={
                    # "lam": PriorSpec("LogNormal", {"mu": 0.0, "sigma": 1.0})
                },
            ),
        },
    )

    mmm = MMM(cfg)

    mmm.build(X, y, rescale=False)
    return mmm
