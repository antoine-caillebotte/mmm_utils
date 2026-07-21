"""Streamlit page to build, fit and diagnose an MMM model on synthetic data.


Lauch with : poetry run streamlit run examples/app.py
"""

# pylint: skip-file
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mmm_utils.plot as mmm_plot
import mmm_utils.post_modeling as mmm_post_plot
from mmm_utils import Timeline
from model_examples import get_builded_mmm

st.set_page_config(page_title="Fit MMM", layout="wide")

DATA_PATH = Path(__file__).resolve().parent.parent / "synthetic_mm_data.csv"
MEDIA = ["TV", "SEA", "Digital"]
CONTROLS = ["intercept", "trend", "school_holidays", "price"]


@st.cache_data(show_spinner="Chargement des données...")
def load_data():
    """Load and prepare the synthetic dataset shared with the fitting notebook."""
    data = pd.read_csv(DATA_PATH, sep=";", decimal=".").fillna(0)
    data["date"] = pd.to_datetime(data["date"])
    data["intercept"] = 1
    data["trend"] = np.linspace(0, 1, len(data))
    data["SEA"] = data["SEA"] - data["SEA"].min()

    X = data[MEDIA + CONTROLS + ["date"]]
    y = data["y"]
    return data, X, y


st.title("Fit d'un modèle MMM")
st.caption(
    "Construction (`model_examples.get_builded_mmm`), ajustement (`mmm.fit`) "
    "et analyse post-modélisation d'un modèle MMM."
)

data, X, y = load_data()

st.subheader("Données utilisées")
st.dataframe(data.head(10), width="stretch")

if "fitted_mmm" not in st.session_state:
    st.session_state.fitted_mmm = None

if st.button("Lancer l'ajustement (fit)", type="primary"):
    with st.spinner(
        "Échantillonnage MCMC en cours (500 draws, 500 tune, 2 chaînes)... "
        "cela peut prendre plusieurs minutes."
    ):
        mmm = get_builded_mmm(X, y, MEDIA, CONTROLS)
        mmm.fit(
            draws=500,
            tune=500,
            chains=2,
            cores=2,
            target_accept=0.975,
        )
        mmm.sample_posterior_predictive()
        _ = mmm.compute_contributions()
    st.session_state.fitted_mmm = mmm
    st.success("Ajustement terminé.")

mmm = st.session_state.fitted_mmm

if mmm is None:
    st.info("Cliquez sur « Lancer l'ajustement (fit) » pour entraîner le modèle.")
else:
    st.subheader("Prédiction a posteriori")
    fig, ax = mmm_post_plot.plot_posterior_predictive_y(mmm, True, True)
    st.pyplot(fig, width="content")

    st.subheader("Timeline des contributions")
    timeline = Timeline(
        mmm.idata.posterior,
        data,
        media=MEDIA,
        controls=["trend", "school_holidays", "price"],
        baseline_components=["intercept", "yearly_seasonality"],
        target_scale=mmm.data.scale("y"),
        target="y",
    )

    fig, ax = mmm_plot.plot_contributions(
        timeline,
        channels=MEDIA + ["trend", "price", "school_holidays"],
        decomposition=False,
        plot_y=True,
        remove_baseline=False,
        ascending=False,
    )
    st.pyplot(fig, width="content")

    fig, ax = mmm_plot.plot_summary_contributions(
        timeline,
        controls=[],
        baseline_override=[],
    )
    st.pyplot(fig, width="content")

    fig, ax = mmm_plot.plot_summary_contributions_per_media(
        timeline, controls=["trend", "promo", "price"]
    )
    st.pyplot(fig, width="content")
