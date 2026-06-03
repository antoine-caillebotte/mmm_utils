import numpy as np

import pandas as pd
import xarray as xr

from datetime import date

from mmm_utils.modeling.utils import max_abs_scaler


import matplotlib.pyplot as plt
import seaborn as sns


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


def make_synthetic_data(n: int = 180, seed: int = 123):
    """Generate a synthetic dataset for MMM smoke tests.

    Parameters
    ----------
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
    media_names = ["TV", "SEA", "Social"]
    control_names = ["price", "promo"]

    # date range
    max_date = pd.to_datetime(date.today())
    min_date = max_date - pd.Timedelta(days=int(n * 7))

    df = pd.DataFrame(
        data={"date": pd.date_range(start=min_date, end=max_date, freq="W-MON")}
    ).assign(
        year=lambda x: x["date"].dt.year,
        month=lambda x: x["date"].dt.month,
        dayofyear=lambda x: x["date"].dt.dayofyear,
    )
    n = len(df)
    # === controls ===
    df["price"], _ = max_abs_scaler(rng.normal(10.0, 2.0, size=n))
    df["promo"], _ = max_abs_scaler(rng.binomial(1, 0.3, size=n).astype(float))

    # === media with adstock/saturation effects ===
    for m in media_names:
        df[m], _ = max_abs_scaler(rng.gamma(1.0, 1.0, size=n))

    media_raw = df[media_names]
    media_raw["TV"] = saturation_np(
        delayed_adstock_np(media_raw["TV"].to_numpy(dtype=float), d=0.99, theta=5),
        lam=0.5,
    )
    media_raw["SEA"] = adstock_np(media_raw["SEA"].to_numpy(dtype=float), d=0.4)
    media_raw["Social"] = saturation_np(
        media_raw["Social"].to_numpy(dtype=float), lam=0.5
    )

    # === build target with known parameters ===
    beta_m = np.array([1, 2, 3])
    beta_c = np.array([-2, 2])
    beta_s = 2 * np.array([1, 1, 0.5, 0])
    beta_t = 4.2
    beta_i = 2.0
    sigma = 2.0

    df["trend"] = np.linspace(start=0.0, stop=1, num=n)

    t = np.arange(len(df), dtype=np.float64)[:, None]
    season_features = np.column_stack(
        [
            f(2 * np.pi * k * t / (365.25 / 7))
            for k in range(1, beta_s.shape[0] // 2 + 1)
            for f in (np.sin, np.cos)
        ]
    )

    df["season"] = season_features @ beta_s

    df["y"], scale_y = (  # max_abs_scaler
        beta_i
        + media_raw[media_names].to_numpy(dtype=np.float64) @ beta_m
        + df[control_names].to_numpy(dtype=np.float64) @ beta_c
        + beta_t * df["trend"]
        + df["season"]
        + rng.normal(0, sigma, n),
        1,
    )

    true_contributions: dict[str, float] = {
        "media": sum(media_raw[media_names].to_numpy(dtype=np.float64) @ beta_m),
        "control": sum(df[control_names].to_numpy(dtype=np.float64) @ beta_c),
        "trend": sum(beta_t * df["trend"]),
        "season": sum(df["season"]),
    }
    all_contributions_sum = sum(true_contributions.values())
    for k, v in true_contributions.items():
        true_contributions[k] = v / all_contributions_sum

    # scale_y = scale_y[0]
    seas_name = sum(
        [[f"sin[{i}]", f"cos[{i}]"] for i in range(beta_s.shape[0] // 2)],
        [],
    )
    parameter_names = [
        "intercept",
        *[f"beta_media[{m}]" for m in media_names],
        *[f"beta_control[{c}]" for c in control_names],
        "beta_control[trend]",
        *seas_name,
        "sigma",
        "adstock_alpha[SEA]",
        "adstock_alpha[TV]",
        "adstock_theta[TV]",
        "saturation_lam[TV]",
        "saturation_lam[Social]",
    ]

    true_params_values = [
        beta_i / scale_y,
        *beta_m / scale_y,
        *beta_c / scale_y,
        beta_t / scale_y,
        *beta_s / scale_y,
        sigma / scale_y,
        0.4,  # adstock alpha SEA
        0.99,  # adstock alpha TV
        2.0,  # adstock theta TV
        0.5,  # saturation lam TV
        0.5,  # saturation lam Social
    ]

    true_params = pd.DataFrame(
        {"true": true_params_values, "parameter": parameter_names}
    )

    ds = xr.Dataset.from_dataframe(df.set_index("date"))
    return ds, media_names, control_names, true_params, true_contributions


def plot_example(df, media_names, control_names):

    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

    sns.lineplot(data=df, x="date", y="y", ax=axes[0], color="black", label="target")
    for m in media_names:
        sns.lineplot(data=df, x="date", y=m, ax=axes[1], label=m)
    for c in control_names:
        sns.lineplot(data=df, x="date", y=c, ax=axes[2], label=c)

    sns.lineplot(
        data=df, x="date", y="season", ax=axes[3], color="orange", label="seasonality"
    )
    sns.lineplot(data=df, x="date", y="trend", ax=axes[3], color="blue", label="trend")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    ds, media_names, control_names, true_params, true_contributions = (
        make_synthetic_data()
    )

    print(true_contributions)
    df = ds.to_dataframe().reset_index()

    plot_example(df, media_names, control_names)

    df.to_csv("synthetic_mmm_data.csv", index=False)
