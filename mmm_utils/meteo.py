"""Utilities to fetch and aggregate weather temperature series."""

import pandas as pd
import meteostat as ms
import matplotlib.pyplot as plt


def mean_by_week(data, date_column="date", value_column="value"):
    """Compute the mean of a value column by week.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe containing a date column and a value column.
    date_column : str, optional
        Name of the date column in ``data``.
    value_column : str, optional
        Name of the value column in ``data``.

    Returns
    -------
    pd.DataFrame
        Dataframe with weekly mean values.
    """
    data["week"] = data[date_column].dt.to_period("W").apply(lambda r: r.start_time)
    weekly_mean = (
        data.groupby("week")[value_column]
        .mean()
        .reset_index()
        .rename(columns={value_column: f"mean_{value_column}"})
    )

    return weekly_mean


def create_national_temperature_columns(data, date_column="date", rolling=14):  # pylint: disable=too-many-locals
    """Create a weekly national temperature series for metropolitan France.

    The function queries Meteostat stations in France, filters to metropolitan
    areas and moderate elevations, fetches daily temperatures, computes a
    rolling average, aggregates to weekly means, and merges the result into the
    input dataframe.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe containing at least a date column.
    date_column : str, optional
        Name of the date column in ``data`` used for merge and date range
        extraction.
    rolling : int, optional
        Window size (in days) for the centered rolling mean applied on daily
        temperatures before weekly aggregation.

    Returns
    -------
    pd.DataFrame
        A copy-like merged dataframe containing all original columns plus a
        ``temperature`` column representing the weekly national temperature.

    Notes
    -----
    Stations with insufficient data are skipped. If a station does not provide
    ``tavg``, the average of ``tmin`` and ``tmax`` is used when available.
    """
    print("Fetching national temperature data for France...")
    # --- Get French stations ---
    stations_fr = ms.stations.query(
        """
        SELECT s.id, n.name, s.country, s.region,
            s.latitude, s.longitude, s.elevation
        FROM stations s
        LEFT JOIN names n ON s.id = n.station AND n.language = 'en'
        WHERE s.country = 'FR'
        """,
        index_col="id",
    )
    print(f"{len(stations_fr)} French stations found")

    # --- filter for metropolitan France only ---
    stations_metro = stations_fr[
        (stations_fr["latitude"].between(41.3, 51.1))
        & (stations_fr["longitude"].between(-5.2, 9.6))
        & (stations_fr["elevation"] < 1500)  # exclude high mountain stations
    ]
    print(
        f"{len(stations_metro)} stations in metropolitan France (excluding mountains)"
    )

    # --- Get temperatures ---
    start = data[date_column].min() - pd.Timedelta(
        days=rolling
    )  # Start earlier for rolling mean
    end = data[date_column].max() + pd.Timedelta(
        days=rolling
    )  # End later for rolling mean

    all_temps = []
    for sid, row in stations_metro.iterrows():
        try:
            ts = ms.daily(sid, start, end)
            if ts.empty:
                continue

            df = ts.fetch()
            # Checking if the station has enough data (30 days)
            if "tavg" in df.columns and df["tavg"].notna().sum() > 30:
                all_temps.append(df["tavg"].rename(row["name"]))
            elif {"tmin", "tmax"}.issubset(df.columns):
                tavg = (df["tmin"] + df["tmax"]) / 2
                if tavg.notna().sum() > 30:
                    all_temps.append(tavg.rename(row["name"]))
        except (KeyError, ValueError, TypeError, OSError) as e:
            print(f"  Error {row['name']}: {e}")

    print(f"{len(all_temps)} stations with sufficient temperature data found")

    combined = pd.concat(all_temps, axis=1)
    france_avg = combined.mean(axis=1).rolling(window=rolling, center=True).mean()
    france_avg = france_avg.reset_index().rename(columns={0: "temp"})

    france_avg = mean_by_week(france_avg, "time", "temp")

    france_avg.rename(columns={"mean_temp": "temperature"}, inplace=True)

    out = data.merge(france_avg, left_on=date_column, right_on="week", how="left").drop(
        columns=["week"]
    )

    print("✅ National temperature data fetched and processed")
    return out


def create_temperature_columns(
    data, latitude, longitude, *, date_column="date", rolling=14
):
    """Create a weekly local temperature series from nearby weather stations.

    The function fetches daily temperature data around the provided geographic
    point, applies Meteostat interpolation, computes a centered rolling mean,
    aggregates values by week, and merges the final weekly series into the
    input dataframe.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe containing at least a date column.
    latitude : float
        Latitude of the target location.
    longitude : float
        Longitude of the target location.
    date_column : str, optional
        Name of the date column in ``data`` used for merge and date range
        extraction.
    rolling : int, optional
        Window size (in days) for the centered rolling mean applied on daily
        temperatures before weekly aggregation.

    Returns
    -------
    pd.DataFrame
        A dataframe containing all original columns plus a ``temperature``
        column with weekly local temperatures.
    """
    print(f"Fetching temperature data for location ({latitude}, {longitude})...")
    # Specify location and time range
    point = ms.Point(latitude, longitude, None)  # elevation
    start = data[date_column].min() - pd.Timedelta(
        days=rolling
    )  # Start earlier for rolling mean
    end = data[date_column].max() + pd.Timedelta(
        days=rolling
    )  # End later for rolling mean

    # Get nearby weather stations
    stations = ms.stations.nearby(point, limit=4)
    print(f"nearst stations found: {stations['name']}")

    # Get daily data & perform interpolation
    ts = ms.daily(stations, start, end)
    df = ms.interpolate(ts, point).fetch()
    df["temperature"] = (
        df[ms.Parameter.TEMP].rolling(window=rolling, center=True).mean()
    )
    df.reset_index(inplace=True)
    weekly_mean_temp = mean_by_week(df, "time", "temperature")

    df.drop(columns=["time", "temperature"], inplace=True)
    weekly_mean_temp.rename(columns={"mean_temperature": "temperature"}, inplace=True)

    out = data.merge(
        weekly_mean_temp, left_on=date_column, right_on="week", how="left"
    ).drop(columns=["week"])

    print("✅ Temperature data fetched and processed")
    return out


if __name__ == "__main__":
    data_test = pd.DataFrame(
        {"date": pd.date_range(start="2024-01-01", end="2026-06-23", freq="W-MON")}
    )

    data_test = (
        data_test.pipe(
            create_temperature_columns, latitude=48.799, longitude=-3.032, rolling=14
        )
        .rename(columns={"temperature": "temperature_PLOUBAZLANEC"})
        .pipe(create_national_temperature_columns, rolling=14)
        .rename(columns={"temperature": "temperature_france"})
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        data_test["date"],
        data_test["temperature_PLOUBAZLANEC"],
        label="PLOUBAZLANEC",
        color="blue",
    )
    ax.plot(
        data_test["date"],
        data_test["temperature_france"],
        label="France",
        color="red",
    )
    ax.set_title("Weekly Rolling Average : Temperature")
    ax.set_xlabel("Date")
    ax.set_ylabel("Temperature (°C)")
    ax.grid()
    ax.legend()
    plt.tight_layout()
    plt.show()
