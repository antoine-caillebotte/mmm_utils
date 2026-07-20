"""Streamlit app to interactively explore the MMM synthetic data simulation.$


Lauch with : poetry run streamlit run examples/app.py
"""

# pylint: skip-file
import io
import sys
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mmm_simulation import MMM_parameter, default_params, make_synthetic_data
from plotting import render_simulation_dashboard

MEDIA_NAMES = ["TV", "SEA", "Digital"]
CONTROL_NAMES = ["price", "school_holidays"]

st.set_page_config(page_title="MMM Simulation", layout="wide")


@st.cache_data(show_spinner="Simulation en cours...")
def run_simulation(
    media_values,
    control_values,
    trend,
    season_values,
    intercept,
    sigma,
    adstock_alpha_items,
    adstock_theta_items,
    saturation_lam_items,
    umbrella,
    n,
    seed,
):
    """Run make_synthetic_data from hashable primitives (cache-friendly)."""
    beta = MMM_parameter(
        media=np.array(media_values, dtype=float),
        control=np.array(control_values, dtype=float),
        trend=trend,
        season=np.array(season_values, dtype=float),
        intercept=intercept,
        sigma=sigma,
        adstock_alpha=dict(adstock_alpha_items),
        adstock_theta=dict(adstock_theta_items),
        saturation_lam=dict(saturation_lam_items),
        umbrella=umbrella,
    )
    ds, media_names, control_names, true_params, true_contributions, ds_transformed = (
        make_synthetic_data(beta=beta, n=n, seed=seed)
    )
    df = ds.to_dataframe().reset_index()
    df_transformed = ds_transformed.to_dataframe().reset_index()
    return (
        df,
        media_names,
        control_names,
        true_params,
        true_contributions,
        df_transformed,
    )


def optional_slider(
    label, key, enabled_default, value_default, min_value, max_value, step
):
    """A slider that can be toggled on/off; returns None when disabled."""
    enabled = st.checkbox(label, value=enabled_default, key=f"{key}_enabled")
    value = st.slider(
        label,
        min_value=min_value,
        max_value=max_value,
        value=value_default,
        step=step,
        key=key,
        disabled=not enabled,
        label_visibility="collapsed",
    )
    return value if enabled else None


def zeroable_slider(label, key, value_default, min_value, max_value, step):
    """A slider with a checkbox to force it to zero, regardless of the slider position."""
    force_zero = st.checkbox("Forcer à zéro", key=f"{key}_zero")
    value = st.slider(
        label,
        min_value=min_value,
        max_value=max_value,
        value=value_default,
        step=step,
        key=key,
        disabled=force_zero,
    )
    return 0.0 if force_zero else value


def reset_media_and_control_to_zero():
    """Callback: force every media and control beta checkbox to zero."""
    for m in MEDIA_NAMES:
        st.session_state[f"media_{m}_zero"] = True
    for c in CONTROL_NAMES:
        st.session_state[f"control_{c}_zero"] = True


with st.sidebar:
    st.header("Paramètres de simulation")

    with st.expander("Simulation", expanded=True):
        n = st.slider(
            "Nombre de semaines (n)", min_value=52, max_value=520, value=180, step=4
        )
        seed = st.slider(
            "Graine aléatoire (seed)", min_value=0, max_value=9999, value=123, step=1
        )

    st.button(
        "Mettre à zéro médias & contrôles",
        on_click=reset_media_and_control_to_zero,
        width="stretch",
    )

    with st.expander("Effet des médias (beta_media)", expanded=True):
        media_values = [
            zeroable_slider(
                m,
                key=f"media_{m}",
                value_default=float(default_params.media[i]),
                min_value=-10.0,
                max_value=10.0,
                step=0.5,
            )
            for i, m in enumerate(MEDIA_NAMES)
        ]

    with st.expander("Effet des contrôles (beta_control)"):
        control_values = [
            zeroable_slider(
                c,
                key=f"control_{c}",
                value_default=float(default_params.control[i]),
                min_value=-10.0,
                max_value=10.0,
                step=0.5,
            )
            for i, c in enumerate(CONTROL_NAMES)
        ]

    with st.expander("Tendance, intercept, bruit"):
        trend = zeroable_slider(
            "Tendance (beta_trend)",
            key="trend",
            value_default=float(default_params.trend),
            min_value=-10.0,
            max_value=10.0,
            step=0.5,
        )
        intercept = zeroable_slider(
            "Intercept",
            key="intercept",
            value_default=float(default_params.intercept),
            min_value=0.0,
            max_value=50.0,
            step=1.0,
        )
        sigma = st.slider(
            "Bruit (sigma)",
            min_value=0.001,
            max_value=5.0,
            value=float(default_params.sigma),
            step=0.1,
        )

    with st.expander("Saisonnalité (beta_season)"):
        season_labels = ["sin[1]", "cos[1]", "sin[2]", "cos[2]"]
        season_values = [
            zeroable_slider(
                label,
                key=f"season_{label}",
                value_default=float(default_params.season[i]),
                min_value=-10.0,
                max_value=10.0,
                step=0.5,
            )
            for i, label in enumerate(season_labels)
        ]

    with st.expander("Adstock & saturation par média"):
        adstock_alpha = {}
        adstock_theta = {}
        saturation_lam = {}
        for m in MEDIA_NAMES:
            st.markdown(f"**{m}**")

            alpha_on = st.checkbox(
                f"Adstock ({m})",
                value=m in default_params.adstock_alpha,
                key=f"alpha_on_{m}",
            )
            alpha_val = st.slider(
                "alpha",
                min_value=0.0,
                max_value=0.99,
                value=float(default_params.adstock_alpha.get(m, 0.5)),
                step=0.01,
                key=f"alpha_val_{m}",
                disabled=not alpha_on,
            )
            if alpha_on:
                adstock_alpha[m] = alpha_val

            theta_on = st.checkbox(
                f"Adstock retardé, theta ({m})",
                value=m in default_params.adstock_theta,
                key=f"theta_on_{m}",
                disabled=not alpha_on,
            )
            theta_val = st.slider(
                "theta",
                min_value=0.0,
                max_value=1.0,
                value=float(default_params.adstock_theta.get(m, 0.3)),
                step=0.01,
                key=f"theta_val_{m}",
                disabled=not (alpha_on and theta_on),
            )
            if alpha_on and theta_on:
                adstock_theta[m] = theta_val

            sat_on = st.checkbox(
                f"Saturation ({m})",
                value=m in default_params.saturation_lam,
                key=f"sat_on_{m}",
            )
            sat_val = st.slider(
                "lambda",
                min_value=0.0,
                max_value=2.0,
                value=float(default_params.saturation_lam.get(m, 0.5)),
                step=0.05,
                key=f"sat_val_{m}",
                disabled=not sat_on,
            )
            if sat_on:
                saturation_lam[m] = sat_val

            st.divider()

    with st.expander("Effet ombrelle (umbrella, boost TV -> autres médias)"):
        umbrella_on = st.checkbox(
            "Activer l'effet ombrelle", value=default_params.umbrella != 0.0
        )
        umbrella_val = st.slider(
            "beta_umbrella",
            min_value=-1.0,
            max_value=10.0,
            value=float(default_params.umbrella)
            if default_params.umbrella != 0.0
            else 0.5,
            step=0.05,
            disabled=not umbrella_on,
        )
    umbrella = umbrella_val if umbrella_on else default_params.umbrella


df, media_names, control_names, true_params, true_contributions, df_transformed = (
    run_simulation(
        media_values=tuple(media_values),
        control_values=tuple(control_values),
        trend=trend,
        season_values=tuple(season_values),
        intercept=intercept,
        sigma=sigma,
        adstock_alpha_items=tuple(sorted(adstock_alpha.items())),
        adstock_theta_items=tuple(sorted(adstock_theta.items())),
        saturation_lam_items=tuple(sorted(saturation_lam.items())),
        umbrella=umbrella,
        n=n,
        seed=seed,
    )
)

st.title("Simulation de données MMM")
st.caption("Pilotage interactif de `make_synthetic_data` — mmm_simulation.py")

st.subheader("Contributions vraies (part de la variance expliquée)")
cols = st.columns(len(true_contributions))
for col, (name, value) in zip(cols, true_contributions.items()):
    col.metric(name, f"{value:.1%}")

st.subheader("Visualisation")
theme_type = st.context.theme.get("type") or "light"
fig = render_simulation_dashboard(
    df, media_names, control_names, df_transformed, theme=theme_type
)
st.plotly_chart(fig, theme=None, width="stretch")

st.subheader("Récapitulatif des paramètres sélectionnés")
with st.expander("Voir les paramètres actuels", expanded=False):
    st.json(
        {
            "n": n,
            "seed": seed,
            "beta_media": dict(zip(MEDIA_NAMES, media_values)),
            "beta_control": dict(zip(CONTROL_NAMES, control_values)),
            "beta_trend": trend,
            "intercept": intercept,
            "sigma": sigma,
            "beta_season": dict(zip(season_labels, season_values)),
            "adstock_alpha": adstock_alpha,
            "adstock_theta": adstock_theta,
            "saturation_lam": saturation_lam,
            "umbrella": umbrella,
        }
    )

st.subheader("Données simulées")
st.dataframe(df.head(20), width="stretch")

csv_buffer = io.StringIO()
df.to_csv(csv_buffer, index=False, sep=";", decimal=".")
st.download_button(
    "Télécharger les données simulées (CSV)",
    data=csv_buffer.getvalue(),
    file_name="synthetic_mm_data.csv",
    mime="text/csv",
)
