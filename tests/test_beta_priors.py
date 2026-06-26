import numpy as np

import pymc as pm
import pymc.dims as pmd

from mmm_utils.modeling import BetaPriors
from mmm_utils.modeling import Interaction
from mmm_utils.modeling import PriorSpec

# pylint: skip-file

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
    print(ia.get_unique_parameter_names())
    print(ia.get_coords())

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
    } | x.interaction.get_coords()
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
