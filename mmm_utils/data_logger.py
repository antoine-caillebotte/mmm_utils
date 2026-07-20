"""Utilities to buffer tabular data and persist it as CSV files."""

import os
from functools import wraps

from pyprojroot import here
import pandas as pd
import numpy as np

MMM_PPTX_DIR = here().parent.parent / "mmm_pptx" / "data"


def skip_if_paused(func):
    """Decorator to skip a method call if the logger is paused.

    Parameters
    ----------
    func : callable
        The method to be decorated.

    Returns
    -------
    callable
        A wrapper function that checks the logger's pause state before calling
        the original method. If the logger is paused, the wrapper returns None
        without executing the method.

    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):  # pylint: disable=missing-function-docstring, missing-return-doc, missing-return-type-doc
        if getattr(self, "_pause", False):
            return None
        return func(self, *args, **kwargs)

    return wrapper


class DataLogger:
    """Buffer and persist data to CSV.

    Each plotting function can append labelled records to the internal buffer
    via :meth:`record`.  When all desired plots have been generated, call
    :meth:`flush_to_csv` to write every buffered record to disk as a tidy
    CSV file and clear the buffer.

    A module-level singleton :data:`plot_logger` is provided so that all
    plotting helpers in this package share the same buffer without requiring
    the caller to instantiate the class explicitly.

    Parameters
    ----------
    None

    Attributes
    ----------
    _buffer : dict
        Internal list of record dictionaries accumulated between
        :meth:`record` calls and the next :meth:`flush_to_csv` call.
    _dir : str
        Directory where CSV files will be written.  Defaults to ``logs`` in the
        project root.  Parent directories must already exist.


    Examples
    --------
    >>> plot_logger.record("spend", {"channel": "tv", "value": 1000.0})
    >>> plot_logger.record("spend", {"channel": "radio", "value": 500.0})
    >>> plot_logger.flush_to_csv("outputs/plot_data.csv")
    """

    def __init__(self) -> None:
        self._buffer: dict = {}
        self._dir = here() / "logs"
        self._pause = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def pause(self) -> bool:
        """Return whether the logger is currently paused.

        When paused, calls to :meth:`record` will be ignored.

        Returns
        -------
        bool
            True if the logger is paused, False otherwise.
        """
        return self._pause

    def pause_logger(self) -> None:
        """Pause the logger, preventing any new records from being buffered."""
        self._pause = True

    def resume_logger(self) -> None:
        """Resume the logger, allowing new records to be buffered."""
        self._pause = False

    def change_dir(self, new_dir: str) -> None:
        """Change the directory where CSV files will be written.

        Parameters
        ----------
        new_dir : str
            New directory path.  Parent directories must already exist.
        """
        self._dir = new_dir

    @skip_if_paused
    def record(self, dataframe: pd.DataFrame | None = None, **data) -> None:
        """Append a labelled data record to the internal buffer.

        Parameters
        ----------
        dataframe : pd.DataFrame | None, optional
            DataFrame containing the record to append.  If provided, the record
            will be extracted from the DataFrame.
        **data : dict
            Mapping of column names to scalar values for this record.
            Keys must be strings; values must be CSV-serialisable scalars
            (``int``, ``float``, ``str``, ``bool``, or ``None``).

        Examples
        --------
        >>> data_logger.record(channel="tv", spend=200, response=0.85)
        >>> data_logger.record(dataframe=df)
        """
        if dataframe is not None:
            data = {col: np.array(dataframe[col]) for col in dataframe.columns}

        self._buffer.update(data)

    @skip_if_paused
    def flush_to_csv(self, filename: str, *, append: bool = False) -> None:
        """Write the buffered records to a CSV file and clear the buffer.

        Parameters
        ----------
        filename : str
            Destination file path for the CSV file.  Parent directories must
            already exist.
        append : bool, default False
            When ``True``, rows are appended to an existing file instead of
            overwriting it.  The header row is written only when the file does
            not yet exist or when ``append`` is ``False``.

        Raises
        ------
        ValueError
            If the buffer is empty when this method is called.
        OSError
            If the file cannot be written (e.g. permission denied or missing
            parent directory).

        Examples
        --------
        >>> plot_logger.flush_to_csv("plot_data.csv")
        >>> plot_logger.flush_to_csv("plot_data.csv", append=True)
        """
        if not self._buffer:
            raise ValueError(
                "The buffer is empty. Call record() before flush_to_csv()."
            )

        buffered_df = pd.DataFrame(self._buffer)

        write_mode = "a" if append else "w"
        write_header = not (append and _csv_file_exists_and_nonempty(filename))

        buffered_df.to_csv(
            self._dir / filename,
            mode=write_mode,
            header=write_header,
            index=False,
        )
        self._buffer.clear()

    def clear(self) -> None:
        """Discard all buffered records without writing to disk."""
        self._buffer.clear()

    def __len__(self) -> int:
        """Return the number of records currently held in the buffer."""
        return len(self._buffer)

    def __repr__(self) -> str:
        return f"PlotDataLogger(buffered_records={len(self._buffer)})"

    @skip_if_paused
    def direct_to_csv(
        self, filename: str, dataframe: pd.DataFrame | None = None, **data
    ):
        """Write a single record to CSV without buffering.

        Parameters
        ----------
        filename : str
            Destination file path for the CSV file.  Parent directories must
            already exist.
        dataframe : pd.DataFrame | None, optional
            DataFrame containing the record to write.  If provided, the record
            will be extracted from the DataFrame.
        **data : dict
            Additional keyword arguments representing the record to write.
        """
        self.clear()
        self.record(dataframe=dataframe, **data)
        self.flush_to_csv(filename=filename, append=False)


def _csv_file_exists_and_nonempty(path: str) -> bool:
    """Return ``True`` if *path* points to a non-empty file.

    Parameters
    ----------
    path : str
        File system path to check.

    Returns
    -------
    bool
    """
    return os.path.isfile(path) and os.path.getsize(path) > 0


data_logger = DataLogger()
