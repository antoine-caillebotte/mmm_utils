import numpy as np

import pandas as pd

from datetime import date

from mmm_utils.modeling.utils import max_abs_scaler
from mmm_utils.holidays import create_holiday_columns

import matplotlib.pyplot as plt
import seaborn as sns

from typing import NamedTuple

# pylint: skip-file


def adstock_np(x, d):
    out = np.zeros_like(x, dtype=float)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = x[i] + d * out[i - 1]
    return out


def delayed_adstock_np(x, d, theta):
    out = np.zeros_like(x, dtype=float)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = x[i] + d * out[i - 1] + theta * x[i - 1]
    return out


def saturation_np(x, lam):
    return (1 - np.exp(-lam * x)) / (1 + np.exp(-lam * x))


def umbrella_effect_df(df, boost_name, beta):
    """Add umbrella effect to a media channel in the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with media channels.
    boost_name : str
        Name of the media channel to boost.
    beta : float
        Coefficient for the umbrella effect.

    Returns
    -------
    pd.DataFrame
        Dataframe with umbrella effect added.
    """
    x = df.copy()
    if boost_name not in df.columns:
        raise ValueError(f"{boost_name} not in dataframe columns")

    # print(f"adding umbrella effect for {boost_name} with beta={beta}")
    boost = beta * df[boost_name]
    for m in df.columns:
        if m != boost_name:
            # print(
            #     f"adding umbrella effect for {boost_name} on {m} equuivalent to {(boost / x[m]).mean() * 100:.1f}% of boost"
            # )
            x[m] = boost * x[m]
        else:
            x[m] = 0

    return x


MMM_parameter = NamedTuple(
    "MMM_parameter",
    [
        ("media", np.ndarray),
        ("control", np.ndarray),
        ("trend", float),
        ("season", np.ndarray),
        ("intercept", float),
        ("sigma", float),
        ("adstock_alpha", dict[str, float]),
        ("adstock_theta", dict[str, float]),
        ("saturation_lam", dict[str, float]),
        ("umbrella", float),
    ],
)

default_params = MMM_parameter(
    media=np.array([3, 2, 1]),  # TV, SEA, Digital
    control=np.array([-2, 2]),
    trend=4.2,
    season=2 * np.array([1, 0.5, 1, 0]),
    intercept=10.0,
    sigma=0.5,
    adstock_alpha={"TV": 0.9, "Digital": 0.8},
    adstock_theta={},  # {"TV": 0.0},
    saturation_lam={"TV": 3, "SEA": 1.0, "Digital": 1.0},
    umbrella=0.0,
)


def make_synthetic_data(
    beta: MMM_parameter = default_params, n: int = 180, seed: int = 123
):
    """Generate a synthetic dataset for MMM smoke tests.

    Parameters
    ----------
    beta : MMM_parameter, optional
        True parameter values.
    n : int, optional
        Number of observations.
    seed : int, optional
        Random seed.

    Returns
    -------
    tuple[xr.Dataset, list[str], list[str], pd.DataFrame]
        Dataset, media names, control names, and true parameter table.
    """
    rng = np.random.default_rng(seed)
    media_names = ["TV", "SEA", "Digital"]
    control_names = ["price", "school_holidays"]

    # date range
    max_date = pd.to_datetime(date.today())
    min_date = max_date - pd.Timedelta(days=int(n * 7))

    df = pd.DataFrame(
        data={"date": pd.date_range(start=min_date, end=max_date, freq="W-MON")}
    )
    n = len(df)
    # === controls ===
    df["price"] = rng.normal(10.0, 2.0, size=n)
    df["promo"] = rng.binomial(1, 0.3, size=n).astype(float)
    df = create_holiday_columns(df, "date")
    df.drop(columns=["public_holidays"], inplace=True)

    # === media with adstock/saturation effects ===
    tv_burst = rng.choice([0, 1], size=n, p=[0.9, 0.1])
    df["TV"] = tv_burst * rng.normal(0.8, 0.1, size=n)

    df["SEA"] = rng.normal(0.8, 0.5, size=n)
    df["SEA"] = df["SEA"].rolling(window=3, min_periods=1).mean()

    digital_burst = rng.choice([0, 1], size=n, p=[0.85, 0.15])
    df["Digital"] = digital_burst * rng.normal(0.5, 0.1, size=n)

    for m in media_names + control_names:
        df[m], _ = max_abs_scaler(df[m].to_numpy(dtype=float))

    media_raw = df[media_names]

    for m in media_names:
        if m in beta.adstock_alpha:
            if m in beta.adstock_theta:
                media_raw[m] = delayed_adstock_np(
                    media_raw[m].to_numpy(dtype=float),
                    d=beta.adstock_alpha[m],
                    theta=beta.adstock_theta[m],
                )
            else:
                media_raw[m] = adstock_np(media_raw[m], d=beta.adstock_alpha[m])
        if m in beta.saturation_lam:
            media_raw[m] = saturation_np(media_raw[m], lam=beta.saturation_lam[m])

    # === build target with known parameters ===
    df["trend"] = np.linspace(start=0.0, stop=1, num=n)

    t = np.arange(len(df), dtype=np.float64)[:, None]
    season_features = np.column_stack(
        [
            f(2 * np.pi * k * t / (365.25 / 7))
            for f in (np.sin, np.cos)
            for k in range(1, beta.season.shape[0] // 2 + 1)
        ]
    )

    df["season"] = season_features @ beta.season
    df["noise"] = rng.normal(0, beta.sigma, n) if beta.sigma != 0 else np.zeros(n)

    df["y"], scale_y = (  # max_abs_scaler
        beta.intercept
        + media_raw[media_names].to_numpy(dtype=np.float64) @ beta.media
        + umbrella_effect_df(media_raw, "TV", beta=beta.umbrella).sum(axis=1)
        + df[control_names].to_numpy(dtype=np.float64) @ beta.control
        + beta.trend * df["trend"]
        + df["season"],
        1,
    )
    df["y"] += df["noise"]

    true_contributions: dict[str, float] = {
        "media": sum(media_raw[media_names].to_numpy(dtype=np.float64) @ beta.media),
        "control": sum(df[control_names].to_numpy(dtype=np.float64) @ beta.control),
        "trend": sum(beta.trend * df["trend"]),
        "season": sum(df["season"]),
    }
    all_contributions_sum = sum(true_contributions.values())
    all_contributions_sum = 1 if all_contributions_sum == 0 else all_contributions_sum
    for k, v in true_contributions.items():
        true_contributions[k] = v / all_contributions_sum

    true_params = MMM_parameter(
        media=beta.media / scale_y,
        control=beta.control / scale_y,
        trend=beta.trend / scale_y,
        season=beta.season / scale_y,
        intercept=beta.intercept / scale_y,
        sigma=beta.sigma / scale_y,
        adstock_alpha=beta.adstock_alpha,
        adstock_theta=beta.adstock_theta,
        saturation_lam=beta.saturation_lam,
        umbrella=beta.umbrella,
    )

    media_raw["date"] = df["date"]
    return (
        df,
        media_names,
        control_names,
        true_params,
        true_contributions,
        media_raw,
    )


def plot_example(df, media_names, control_names, df_transformed):
    fig, axes = plt.subplots(5, 1, figsize=(10, 10), sharex=True)

    sns.lineplot(data=df, x="date", y="y", ax=axes[0], color="black", label="target")
    for m in media_names:
        sns.lineplot(data=df, x="date", y=m, ax=axes[1], label=m)

    for m in media_names:
        sns.lineplot(data=df_transformed, x="date", y=m, ax=axes[2], label=m)

    for c in control_names:
        sns.lineplot(data=df, x="date", y=c, ax=axes[3], label=c)

    sns.lineplot(
        data=df, x="date", y="season", ax=axes[4], color="orange", label="seasonality"
    )
    sns.lineplot(data=df, x="date", y="trend", ax=axes[4], color="blue", label="trend")

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    df, media_names, control_names, true_params, true_contributions, df_transformed = (
        make_synthetic_data(n=52 * 6)
    )

    vars_cols = media_names + control_names
    variance_df = df[vars_cols].var()
    variance_df_transformed = df_transformed[media_names].var()

    print(f"variance of y: {df['y'].var()}, variance of noise: {df['noise'].var()}")
    print("variance_df:", variance_df)
    print("variance_df_transformed:", variance_df_transformed)

    print(true_contributions)

    plot_example(df, media_names, control_names, df_transformed)
    plt.show()

    df.drop(columns=["trend", "season", "noise"], inplace=True)
    df.to_csv("synthetic_mm_data.csv", index=False, sep=";", decimal=".")
