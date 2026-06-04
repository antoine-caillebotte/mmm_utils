"""This module provides functions to create columns in a dataframe
with the number of holidays (public and school) in the week of a given date (in a column).

Sources:
https://github.com/vacanza/holidays
https://www.data.gouv.fr/datasets/vacances-scolaires-par-zones
https://github.com/AntoineAugusti/vacances-scolaires-france
"""

from datetime import date, timedelta
import pandas as pd

import holidays as holidays_pkg
from vacances_scolaires_france import SchoolHolidayDates


def create_holiday_columns(data, date_column="date", column_by_zone: bool = False):
    """Add weekly public and school-holiday indicators to a dataframe.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe containing a date column.
    date_column : str, optional
        Name of the date column in ``data``.
    column_by_zone : bool, optional
        If ``True``, add one school-holiday column per zone (A/B/C).
        If ``False``, add a single averaged ``school_holidays`` column.

    Returns
    -------
    pd.DataFrame
        Dataframe with holiday feature columns added.
    """

    school_holidays = SchoolHolidayDates()

    def count_public_holidays_in_week(year: int, week: int) -> int:  # pylint: disable=missing-return-doc
        public_holidays = holidays_pkg.France(years=[year])

        week_start = date.fromisocalendar(year, week, 1)  # Monday
        week_days = [week_start + timedelta(days=i) for i in range(7)]
        return sum(1 for d in week_days if d in public_holidays)

    def count_school_holidays_in_week(year: int, week: int, zone: str) -> int:  # pylint: disable=missing-return-doc
        wednesday = date.fromisocalendar(year, week, 3)  # Wednesday

        return int(school_holidays.is_holiday_for_zone(wednesday, zone=zone))

    data["public_holidays"] = data[date_column].apply(
        lambda x: count_public_holidays_in_week(x.year, x.isocalendar()[1])
    )
    if column_by_zone:
        for z in ["A", "B", "C"]:
            data[f"school_holidays_zone_{z}"] = data[date_column].apply(
                lambda x, z=z: count_school_holidays_in_week(
                    x.year, x.isocalendar()[1], zone=z
                )
            )
    else:
        data["school_holidays"] = data[date_column].apply(
            lambda x: (
                sum(
                    count_school_holidays_in_week(x.year, x.isocalendar()[1], zone=z)
                    for z in ["A", "B", "C"]
                )
                / 3
            )
        )

    return data


if __name__ == "__main__":
    df = pd.DataFrame(
        {"date": pd.date_range(start="2026-01-01", end="2026-12-31", freq="D")}
    )

    df = df.pipe(create_holiday_columns, date_column="date", column_by_zone=True)
    df = df.pipe(create_holiday_columns, date_column="date", column_by_zone=False)

    for _, row in df.iterrows():
        print(
            f"{row['date'].date()} "
            f"- Public: {row['public_holidays']} "
            f"- School: {row['school_holidays_zone_A']},"
            f" {row['school_holidays_zone_B']},"
            f" {row['school_holidays_zone_C']}"
            f"- Average school holidays: {row['school_holidays']}"
        )
