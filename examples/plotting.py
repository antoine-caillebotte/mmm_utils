"""Plotly dashboard for the MMM simulation, tuned to fit inside a Streamlit page.

Colors follow a fixed categorical order (identity, not rank) so the same media
or control channel keeps the same color across every panel.
"""
# pylint: skip-file

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

_CATEGORICAL = {
    "light": [
        "#2a78d6",
        "#1baf7a",
        "#eda100",
        "#008300",
        "#4a3aa7",
        "#e34948",
        "#e87ba4",
        "#eb6834",
    ],
    "dark": [
        "#3987e5",
        "#199e70",
        "#c98500",
        "#008300",
        "#9085e9",
        "#e66767",
        "#d55181",
        "#d95926",
    ],
}
_INK = {"light": "#0b0b0b", "dark": "#ffffff"}
_MUTED = {"light": "#898781", "dark": "#898781"}
_GRID = {"light": "#e1e0d9", "dark": "#2c2c2a"}
_AXIS = {"light": "#c3c2b7", "dark": "#383835"}

_MEDIA_ROW_TITLES = {
    "bruts": "Médias — bruts",
    "transformés": "Médias — transformés (adstock + saturation + ombrelle)",
}


def render_simulation_dashboard(
    df,
    beta,
    media_names,
    control_names,
    df_transformed,
    theme="light",
    media_view="bruts",
):
    """Build an interactive, theme-aware Plotly dashboard for the MMM simulation.

    Parameters
    ----------
    df : pd.DataFrame
        Main simulated dataframe (date, y, media, controls, trend, season).
    beta : MMM_parameter
        parameter values.
    media_names : list[str]
        Media channel column names.
    control_names : list[str]
        Control column names.
    df_transformed : pd.DataFrame
        Media columns after adstock/saturation/umbrella transforms.
    theme : {"light", "dark"}
        Selects the ink/grid/categorical steps validated for that surface.
    media_view : {"bruts", "transformés"}
        Which media series to plot: the raw inputs or the adstock/saturation/
        umbrella-transformed ones.

    Returns
    -------
    go.Figure
    """
    palette = _CATEGORICAL[theme]
    ink = _INK[theme]
    muted = _MUTED[theme]
    grid = _GRID[theme]
    axis_color = _AXIS[theme]

    entities = list(media_names) + list(control_names) + ["trend", "season"]
    colors = {name: palette[i % len(palette)] for i, name in enumerate(entities)}

    media_df = df if media_view == "bruts" else df_transformed
    row_titles = [
        "Variable cible (y)",
        _MEDIA_ROW_TITLES[media_view],
        "Contrôles",
        "Tendance & saisonnalité",
    ]

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        row_heights=[0.25, 0.25, 0.2, 0.2],
        subplot_titles=row_titles,
    )

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["y"],
            mode="lines",
            name="y",
            line=dict(color=ink, width=2),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    for m in media_names:
        if beta.media[media_names.index(m)] == 0:
            continue

        fig.add_trace(
            go.Scatter(
                x=media_df["date"],
                y=media_df[m],
                mode="lines",
                name=m,
                legendgroup=m,
                showlegend=True,
                line=dict(color=colors[m], width=2),
            ),
            row=2,
            col=1,
        )

    for c in control_names:
        if beta.control[control_names.index(c)] == 0:
            continue

        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df[c],
                mode="lines",
                name=c,
                legendgroup=c,
                showlegend=True,
                line=dict(color=colors[c], width=2),
            ),
            row=3,
            col=1,
        )

    if beta.trend != 0:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["trend"],
                mode="lines",
                name="Tendance",
                legendgroup="trend",
                showlegend=True,
                line=dict(color=colors["trend"], width=2),
            ),
            row=4,
            col=1,
        )

    if np.any(beta.season != 0):
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["season"],
                mode="lines",
                name="Saisonnalité",
                legendgroup="season",
                showlegend=True,
                line=dict(color=colors["season"], width=2),
            ),
            row=4,
            col=1,
        )

    fig.update_layout(
        height=760,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=ink, family="system-ui, -apple-system, 'Segoe UI', sans-serif"),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    fig.update_xaxes(
        showgrid=False,
        linecolor=axis_color,
        tickfont=dict(color=muted),
        ticks="outside",
        automargin=True,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=grid,
        zeroline=False,
        linecolor=axis_color,
        tickfont=dict(color=muted),
        automargin=True,
    )
    for annotation in fig.layout.annotations:
        annotation.font = dict(color=muted, size=13)
        annotation.x = 0
        annotation.xanchor = "left"

    return fig
