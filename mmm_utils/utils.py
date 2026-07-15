"""Utility helpers for exploring and preparing media mix modeling data.

This module provides:

- :func:`compute_spend_distribution`, which summarizes media spend shares
  and flags media with insufficient budget allocation.
- :class:`MixMediaDataCreator`, a context manager for incrementally
  building a media mix dataset via temporary CSV checkpoints.
"""

import os
import json
import re

import pandas as pd
import numpy as np

from pyprojroot.here import here
from pymongo import MongoClient


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

        # Assuring naming convention
        self.df.columns = self.df.columns.str.lower()
        self.df.columns = self.df.columns.str.replace(" ", "_")

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

    def rename(self, column, mapper: dict):
        """Rename values in a column using a mapping.

        Parameters
        ----------
        column : str
            Name of the column to update.
        mapper : dict
            Mapping of old values to new values.

        Returns
        -------
        self
            Current instance with updated dataframe.
        """
        for old_value, new_value in mapper.items():
            self.df.loc[:, self.df[column] == old_value, column] = new_value  # pylint: disable=unsubscriptable-object
        return self

    def rename_if(self, column, new_value: str, if_column: str, if_value: str):
        """Conditionally set a column value based on another column.

        Parameters
        ----------
        column : str
            Name of the column to update.
        new_value : Any
            Value assigned to ``column`` when condition is met.
        if_column : str
            Name of the column used for the condition.
        if_value : Any
            Value in ``if_column`` that triggers the update.

        Returns
        -------
        self
            Current instance with updated dataframe.
        """
        self.df.loc[:, self.df[if_column] == if_value, column] = new_value  # pylint: disable=unsubscriptable-object
        return self


def _check_doc_lelab(doc: dict) -> bool:
    """Validate presence of required keys in a LeLab document.

    Parameters
    ----------
    doc : dict
        Candidate document to validate.

    Returns
    -------
    bool
        ``True`` when all required keys are present.

    Raises
    ------
    ValueError
        If one or more required keys are missing.
    """
    if not isinstance(doc, dict):
        raise TypeError("'doc' must be a dictionary.")

    required_top_keys = {
        "code_module",
        "created_at",
        "labels",
        "mediagroups",
        "recommendations",
        "saturation",
        "timeline",
        "version",
    }
    missing_top_keys = sorted(required_top_keys - set(doc.keys()))
    if missing_top_keys:
        raise ValueError(f"Missing top-level keys: {missing_top_keys}")

    created_at = doc.get("created_at")
    if not isinstance(created_at, str):
        raise ValueError("'created_at' must be a string.")
    if not re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$", created_at):
        raise ValueError(
            "'created_at' must match format 'YYYY-MM-DDTHH:MM:SS.mmmZ'"
            "(e.g. '2026-02-16T17:43:31.206Z')."
        )

    labels = doc.get("labels", {})
    if not isinstance(labels, dict):
        raise ValueError("'labels' must be a dictionary containing required keys.")
    required_label_keys = {
        "currency",
        "outcome_unit",
        "outcome_contribution",
        "media_attributed_outcome",
        "cost_per_outcome",
    }
    missing_label_keys = sorted(required_label_keys - set(labels.keys()))
    if missing_label_keys:
        raise ValueError(f"Missing 'labels' keys: {missing_label_keys}")

    saturation = doc.get("saturation", {})
    if not isinstance(saturation, dict):
        raise ValueError("'saturation' must be a dictionary containing required keys.")
    required_saturation_keys = {"media", "details"}
    missing_saturation_keys = sorted(required_saturation_keys - set(saturation.keys()))
    if missing_saturation_keys:
        raise ValueError(f"Missing 'saturation' keys: {missing_saturation_keys}")

    return True


class LeLabDataBase:
    """Small wrapper around the LeLab MongoDB collection.

    Parameters
    ----------
    uri : str
        MongoDB connection URI.

    Raises
    ------
    ValueError
        If the expected database or collection is not available.
    AssertionError
        If the MongoDB client cannot be initialized.
    """

    def __init__(self, uri: str):
        """Initialize the MongoDB client and resolve the target collection.

        Parameters
        ----------
        uri : str
            MongoDB connection URI.

        Raises
        ------
        ValueError
            If database ``lelab`` or collection ``datas_modules`` is missing.
        AssertionError
            If the created client is ``None``.
        """

        self._client = MongoClient(uri)
        assert self._client is not None, "MongoDB client is None. Check your MONGO_URI."

        if "lelab" not in self._client.list_database_names():
            raise ValueError(
                "Database 'lelab' not found. Check your MongoDB connection."
            )
        if "datas_modules" not in self._client["lelab"].list_collection_names():
            raise ValueError(
                "Collection 'datas_modules' not found in database 'lelab'."
                "Check your MongoDB connection."
            )

        self._db = self._client["lelab"]["datas_modules"]

    def __del__(self):
        """Close the MongoDB client on deletion."""
        if self._client is not None:
            self._client.close()

    def dump_json(self, doc, logs_dir: str | None = None):
        """Serialize a document to a JSON file in the logs directory.

        Parameters
        ----------
        doc : dict
            Document to serialize. Must contain ``code_module`` and ``date`` keys.
        logs_dir : pathlib.Path, default=here() / "logs"
            Directory where the JSON file will be written.

        Notes
        -----
        The output filename follows the pattern
        ``<code_module>_<timestamp>.json``. NumPy scalars and arrays are
        automatically converted to native Python types.
        """
        if logs_dir is None:
            logs_dir = here() / "logs"

        code_module = doc["code_module"]
        ts = doc["date"].strftime("%Y_%m%dT%Hh%Mm%Ss_%fZ")

        json_path = logs_dir / f"{code_module}_{ts}.json"

        def _json_default(o):
            if isinstance(o, np.generic):
                return o.item()
            if isinstance(o, np.ndarray):
                return o.tolist()
            return str(o)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2, default=_json_default)

        print(f"📝 JSON saved: {json_path}")

    def list_documents(self, code_module: str):
        """List documents for a module sorted by most recent date first.

        Parameters
        ----------
        code_module : str
            Module code used to filter documents.

        Returns
        -------
        list[dict]
            Matching documents sorted in descending order of ``date``.
        """
        all_doc = list(self._db.find({"code_module": code_module}))

        all_doc.sort(key=lambda x: x["date"], reverse=True)
        return all_doc

    def print_all_documents(self, code_module: str):
        """Print document identifiers and dates for a module.

        Parameters
        ----------
        code_module : str
            Module code used to filter documents.
        """
        all_doc = self.list_documents(code_module)
        if not all_doc:
            print(f"No documents found with code_module: {code_module}")
            return

        for doc in all_doc:
            print(f"Document ID: {doc['_id']}, Date: {doc['date']}")

    def delete_documents(self, code_module: str):
        """Delete all documents for a module after interactive confirmation.

        Parameters
        ----------
        code_module : str
            Module code identifying records to delete.

        Raises
        ------
        ValueError
            If no matching document exists or if user cancels deletion.
        """

        old_one = self._db.find_one({"code_module": code_module})
        if not old_one:
            raise ValueError(f"No document found with code_module: {code_module}")

        while old_one:
            ask = input(
                f"⚠️ Are you sure you want to delete {code_module} (dated {old_one['date']})?"
                " Press Enter to continue..."
            )
            if ask != "":
                raise ValueError("Delete cancelled by user ❌")

            self._db.delete_one({"code_module": code_module, "date": old_one["date"]})
            print(f"{code_module} (dated {old_one['date']}) deleted successfully 🗑️")
            old_one = self._db.find_one({"code_module": code_module})

    def insert(self, doc: dict):
        """Insert a document after validation.

        Parameters
        ----------
        doc : dict
            Document payload containing at least ``code_module`` and ``date``.
        """
        _check_doc_lelab(doc)
        code_module = doc["code_module"]

        self._db.insert_one(doc)
        print(f"{code_module} (dated {doc['date']}) inserted successfully ✅")

    def update(self, doc: dict):
        """Update the most recent document for a module after confirmation.

        Parameters
        ----------
        doc : dict
            New payload used to replace fields in the latest document.

        Raises
        ------
        ValueError
            If no document exists for the module or if user cancels update.
        """
        _check_doc_lelab(doc)
        code_module = doc["code_module"]
        all_doc = self.list_documents(code_module)
        if not all_doc:
            raise ValueError(f"No document found with code_module: {code_module}")

        old_one = all_doc[0]
        ask = input(
            f"⚠️ Are you sure you want to update {code_module} (dated {old_one['date']})?"
            " Press Enter to continue..."
        )
        if ask != "":
            raise ValueError("Update cancelled by user ❌")

        self._db.update_one({"_id": old_one["_id"]}, {"$set": doc})
        print(f"{code_module} (dated {doc['date']}) updated successfully ✅")
