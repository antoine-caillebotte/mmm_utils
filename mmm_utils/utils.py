"""Utility helpers for exploring and preparing media mix modeling data.

This module provides:

- :func:`compute_spend_distribution`, which summarizes media spend shares
  and flags media with insufficient budget allocation.
- :class:`MixMediaDataCreator`, a context manager for incrementally
  building a media mix dataset via temporary CSV checkpoints.
"""

import os

import pandas as pd

from pyprojroot.here import here


def compute_spend_distribution(df, significance_threshold: float = 0.05):
    """Compute media spend shares and assign a decision flag.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing at least ``media`` and ``budget`` columns.
    significance_threshold : float, default=0.05
        Threshold used to assign decision labels from each media proportion.

    Returns
    -------
    pandas.DataFrame
        Dataframe sorted by descending spend with columns:
        ``column``, ``depenses``, ``proportion``, and ``decision``.
    """
    summary = (
        df.sum(axis=0).reset_index().rename(columns={"index": "media", 0: "depenses"})
    )

    total_depenses = summary["depenses"].sum()
    summary["proportion"] = (
        0 if total_depenses == 0 else summary["depenses"] / total_depenses
    )

    def _make_decision(x):
        if x >= significance_threshold:
            return "✅"
        if x < significance_threshold / 2:
            return "❌"
        return "⚠️"

    summary["decision"] = summary["proportion"].apply(_make_decision)

    out = (
        summary.sort_values("depenses", ascending=False)
        .reset_index(drop=True)
        .round({"proportion": 4})
    )
    print(out)
    print()
    return out


class MixMediaDataCreator:
    """Context manager to create and clean temporary CSV files.

    Notes
    -----
    Temporary files are created in ``<project_root>/data`` using the pattern
    ``tmp_building_mm_<n>.csv``. On exit, the user is prompted to remove
    them, and ``self.df`` is written to the final output file.

    Example
    -------
    >>> with MixMediaDataCreator("final_dataset") as manager:
    ...     manager.dump_to_tmp_csv(df)
    """

    def __init__(self, filename):
        """Initialize the manager with the final output filename.

        Parameters
        ----------
        filename : str
            Name (without extension) of the final CSV file written to
            ``<project_root>/data`` on exit.
        """
        self._filename = filename
        self._tmp = 0

        self.df: pd.DataFrame = None

    def __enter__(self):
        """Enter context by cleaning existing temp files and resetting counter.

        Returns
        -------
        MixMediaDataCreator
            The current instance.
        """
        self.delete_tmp_files()
        self._tmp = 0
        return self

    def __exit__(self, exc_type, exc, tb):
        """Exit context and optionally delete temporary files.

        Parameters
        ----------
        exc_type : type or None
            Exception type if an exception occurred, else ``None``.
        exc : BaseException or None
            Exception instance if an exception occurred, else ``None``.
        tb : traceback or None
            Traceback object if an exception occurred, else ``None``.
        """
        delete = input("Do you want to remove temporary files? (any/no): ") != "no"
        if delete:
            self.delete_tmp_files()

        self.df.to_csv(
            here() / "data" / f"{self._filename}.csv",
            index=False,
            sep=";",
            decimal=".",
        )
        print(f"Data saved to {here() / 'data' / f'{self._filename}.csv'}")

    def delete_tmp_files(self):
        """Delete sequential temporary CSV files from the data directory."""
        i = 0
        while True:
            tmp_file = here() / "data" / f"tmp_building_mm_{i}.csv"
            if not os.path.exists(tmp_file):
                break
            os.remove(tmp_file)
            i += 1

    def dump_to_tmp_csv(self, data: pd.DataFrame | None = None):
        """Write a dataframe to the next temporary CSV file.

        Parameters
        ----------
        data : pandas.DataFrame or None, default=None
            Dataframe to save. If ``None``, ``self.df`` is saved instead.
        """
        if data is not None:
            data.to_csv(
                here() / "data" / f"tmp_building_mm_{self._tmp}.csv",
                index=False,
                sep=";",
                decimal=".",
            )
        else:
            self.df.to_csv(
                here() / "data" / f"tmp_building_mm_{self._tmp}.csv",
                index=False,
                sep=";",
                decimal=".",
            )
        self._tmp += 1
