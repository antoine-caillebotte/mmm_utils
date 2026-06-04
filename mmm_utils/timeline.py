"""Timeline utilities for media mix modeling."""

from __future__ import annotations
import dataclasses

import pandas as pd

import xarray as xr


@dataclasses.dataclass
class TimelineDataBuffer:
    """Internal buffer for caching timeline data and derived DataFrames."""

    timeline: dict[str, list[dict]] | None = None
    outcome_df: pd.DataFrame | None = None
    spend_df: pd.DataFrame | None = None


@dataclasses.dataclass
class DataHandler:
    """Validated accessor around the raw input dataframe and column names."""

    data: pd.DataFrame | None = None
    target_name: str | None = None
    media: list[str] | None = None
    controls: list[str] | None = None

    @property
    def dates(self) -> pd.Series:
        """Get the dates from the data.

        Returns
        -------
        pandas.Series
                Series of dates from the data, converted to datetime.
        """
        return pd.to_datetime(self.data["date"])

    @property
    def target(self) -> pd.Series:
        """Get the target variable from the data.


        Returns
        -------
        pandas.Series
                Series of target variable values from the data.
        """
        return self.data[self.target_name]

    def __post_init__(self) -> None:
        required_columns = [self.target_name, *self.media, *self.controls]
        missing_columns = [
            col for col in required_columns if col not in self.data.columns
        ]
        if missing_columns:
            raise ValueError(
                "Missing required columns in data: "
                f"{', '.join(sorted(set(missing_columns)))}"
            )

    def get_spendi(self, i: int, m: str) -> float:
        """Get media spend for channel *m* at row index *i*.

        Parameters
        ----------
        i : int
            Index of the row in the data.
        m : str
            Name of the media media.
        Returns
        -------
        float
            Media spend value for media *m* at index *i*, or 0.0 if
            the media is not found or index is out of bounds.
        """
        if m in self.data.columns:
            if i < len(self.data):
                return float(self.data[m].iloc[i])

        return 0.0


def _mean(x: xr.DataArray, accepted_dim=None) -> xr.DataArray:
    if accepted_dim is None:
        accepted_dim = ["chain", "draw", "sample"]
    dims = [d for d in x.dims if d in accepted_dim]
    return x.mean(dim=dims)


def _validate_posterior_and_data_dates(posterior, data):
    if "date" not in posterior.coords:
        raise ValueError("posterior must contain a 'date' coordinate.")
    if "date" not in data.columns:
        raise ValueError("data must contain a 'date' column.")

    posterior_n_dates = int(posterior.coords["date"].size)
    data_n_dates = int(len(data))
    if posterior_n_dates != data_n_dates:
        raise ValueError(
            "posterior and data must have the same number of dates: "
            f"{posterior_n_dates} != {data_n_dates}."
        )


class Timeline:
    """Builds and exposes a date-keyed timeline of media and baseline contributions.

    Parameters
    ----------
    posterior : xarray.Dataset
        Posterior samples containing contribution variables.
    data : pandas.DataFrame
        Raw input data with a ``date`` column and one column per media media.
    target : str
        Name of the target variable in *data* (e.g. ``"y"``).
    target_scale : float, default=1.0
        Multiplicative scale applied to every contribution value.
    baseline_components : list of str, optional
        List of baseline component names to include in the baseline.
        Valid values include ``"control"``, ``"yearly_seasonality"``.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        posterior,
        data: pd.DataFrame,
        *,
        media: list[str] | None = None,
        controls: list[str] | None = None,
        target="y",
        target_scale: float = 1.0,
        baseline_components=None,
        dim_name: dict[str, str] | None = None,
    ) -> None:
        _validate_posterior_and_data_dates(posterior, data)

        self._posterior = posterior
        self._data = DataHandler(
            data, target, media=media or [], controls=controls or []
        )

        self._target_scale = target_scale
        self._baseline_components = (
            baseline_components
            if baseline_components is not None
            else ["control", "yearly_seasonality"]
        )
        self._dim_name = {
            "date": "date",
            "media": "media",
            "yearly_seasonality": "yearly_seasonality",
            "trend": "trend",
            "control": "control",
            "intercept": "intercept",
        }
        for logical_name, dim in (dim_name or {}).items():
            if logical_name in self._dim_name:
                self._dim_name[logical_name] = dim
            else:
                raise ValueError(
                    f"Unknown logical dimension name: {logical_name}. "
                    f"Valid options are: {', '.join(self._dim_name.keys())}."
                )

        for comp in self._baseline_components:
            if comp not in ["control", "yearly_seasonality"]:
                raise ValueError(
                    f"Invalid baseline component: {comp}. "
                    "Valid options are 'control' and 'yearly_seasonality' or both."
                )

        self._buffer = TimelineDataBuffer(
            timeline=None,
            outcome_df=None,
            spend_df=None,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def dim(self, name: str) -> str:
        """Get the dimension name for a given logical dimension.

        Parameters
        ----------
        name : str
            Logical dimension name (e.g. ``"date"``, ``"media"``).

        Returns
        -------
        str
            Actual dimension name used in the posterior dataset.
        """
        assert name in self._dim_name, f"Unknown dimension name: {name}"
        return self._dim_name[name]

    def _add_media_to_entries(
        self,
        entries: list[dict],
        name: str,
        spend: float,
        outcome: float,
    ) -> list[dict]:
        """Append a media entry to *entries* and return it.

        Parameters
        ----------
        entries : list[dict]
            Running list of entries for the current date.
        name : str
            media name.
        spend : float
            Raw media spend value for this date.
        outcome : float
            Unscaled contribution value; will be multiplied by *target_scale*.

        Returns
        -------
        list[dict]
            The mutated *entries* list.
        """
        entries.append(
            {
                "type": "media",
                "name": name,
                "spend": spend,
                "outcome": float(outcome * self._target_scale),
            }
        )
        return entries

    def _add_baseline_to_entries(
        self,
        entries: list[dict],
        name: str,
        outcome: float,
    ) -> list[dict]:
        """Append a baseline entry to *entries* and return it.

        Parameters
        ----------
        entries : list[dict]
            Running list of entries for the current date.
        name : str
            Baseline component name (e.g. ``"Baseline"``).
        outcome : float
            Unscaled contribution value; will be multiplied by *target_scale*.

        Returns
        -------
        list[dict]
            The mutated *entries* list.
        """
        entries.append(
            {
                "type": "baseline",
                "name": name,
                "spend": 0,
                "outcome": float(outcome * self._target_scale),
            }
        )
        return entries

    def _to_dataframe(self, value: str) -> pd.DataFrame:
        """Convert the internal timeline to a wide-format DataFrame.

        Parameters
        ----------
        value : str
            Entry key to pivot (``"outcome"`` or ``"spend"``).

        Returns
        -------
        pandas.DataFrame
            Wide DataFrame with a ``date`` column and one column per media /
            baseline component.
        """
        rows: list[dict] = []
        for date_key, date_entries in self.timeline.items():
            parsed_date = pd.to_datetime(date_key, format="%Y-%m-%d", errors="coerce")
            if pd.isna(parsed_date):
                raise ValueError(f"Invalid date_key format: {date_key}")
            row: dict = {"date": parsed_date}
            for item in date_entries:
                name = item.get("name")
                v = item.get(value)
                if name is not None and v is not None:
                    row[name] = float(v)
            rows.append(row)

        out = pd.DataFrame(rows)
        out[self.target] = self._data.target.to_numpy()
        return out

    def _get_reduced_contribution(self, var_name: str) -> xr.DataArray:
        """Extract and reduce a contribution variable from the posterior.

        Parameters
        ----------
        var_name : str
            Name of the contribution variable to extract (e.g. ``"control"``,
            ``"yearly_seasonality"``).

        Returns
        -------
        xarray.DataArray
            Reduced contribution values with dimensions ``date`` and optionally
            ``media``.
        """
        var_key = var_name + "_contribution"
        dates = self._posterior.coords["date"]
        if var_key in self._posterior:
            out = _mean(self._posterior[var_key])
            if len(out.coords) == 0:
                out = out.expand_dims(date=dates)
            if len(out.coords) == 1 and "date" in out.coords:
                out = out.expand_dims({var_name: [var_name]}).transpose(
                    "date", var_name
                )
            return out

        dates = pd.to_datetime(dates).values
        return xr.DataArray(
            [[0.0] for _ in range(len(dates))],
            dims=["date", var_name],
            coords={"date": dates, var_name: [var_name]},
            name=var_name,
        )

    def _build_contributions(self) -> xr.Dataset:
        """Compute the contributions for all medias and baseline components,
        and the baseline timeline.

        Returns
        -------
        all_contributions : xarray.Dataset
            Dataset with dimensions ``date`` and ``media``, containing the
            contribution values for each media and baseline component
            (if not included in the baseline).
        baseline_timeline : xarray.DataArray
            1D array with dimension ``date``, containing the total baseline
            contribution for each date.
        """
        media = self._get_reduced_contribution(self.dim("media"))
        control = self._get_reduced_contribution(self.dim("control"))
        yearly_seasonality = self._get_reduced_contribution(
            self.dim("yearly_seasonality")
        )
        intercept = self._get_reduced_contribution(self.dim("intercept"))
        baseline_timeline = intercept.sum(dim=self.dim("intercept"))

        all_contributions = media.copy()
        for comp in ["control", "yearly_seasonality"]:
            if comp not in self._baseline_components:
                if comp == "control":
                    control = control.rename({self.dim("control"): self.dim("media")})
                    all_contributions = xr.concat(
                        [all_contributions, control], dim=self.dim("media")
                    )
                elif comp == "yearly_seasonality":
                    yearly_seasonality = yearly_seasonality.rename(
                        {self.dim("yearly_seasonality"): self.dim("media")}
                    )

                    all_contributions = xr.concat(
                        [all_contributions, yearly_seasonality], dim=self.dim("media")
                    )

            else:
                if comp == "control":
                    baseline_timeline += control.sum(dim=self.dim("control"))
                elif comp == "yearly_seasonality":
                    baseline_timeline += yearly_seasonality.sum(
                        dim=self.dim("yearly_seasonality")
                    )

                elif comp == "intercept":
                    pass  # already included
                else:
                    raise ValueError(f"Unknown baseline component: {comp}")

        return all_contributions, baseline_timeline

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build(self) -> Timeline:
        """Compute the timeline from the posterior and cache it.

        Returns
        -------
        Timeline
            *self*, to allow method chaining.
        """
        all_contributions, baseline_timeline = self._build_contributions()

        timeline: dict[str, list[dict]] = {}
        dates = all_contributions.coords[self.dim("date")].values
        contrib = all_contributions[self.dim("media")].values

        for i, date_val in enumerate(dates):
            entries: list[dict] = []
            date_str = pd.to_datetime(date_val).strftime("%Y-%m-%d")

            self._add_baseline_to_entries(
                entries, "Baseline", outcome=baseline_timeline[i]
            )

            for m in contrib:
                m_contri = float(
                    all_contributions.isel(date=i).sel({self.dim("media"): m}).values
                )
                spend = self._data.get_spendi(i, m)

                self._add_media_to_entries(entries, m, spend=spend, outcome=m_contri)

            timeline[date_str] = entries

        self._buffer.timeline = timeline
        # Invalidate cached DataFrames whenever the timeline is rebuilt
        self._buffer.outcome_df = None
        self._buffer.spend_df = None
        return self

    # ------------------------------------------------------------------
    # public direct getters
    # ------------------------------------------------------------------
    @property
    def target(self) -> str:
        """Name of the target variable.

        Returns
        -------
        str
             Target variable name.
        """
        return self._data.target_name

    @property
    def timeline(self) -> dict[str, list[dict]]:
        """Return the raw timeline dict, building it on first access.

        Returns
        -------
        dict[str, list[dict]]
            Date-keyed mapping of entry lists.
        """
        if self._buffer.timeline is None:
            self.build()
        return self._buffer.timeline  # type: ignore[return-value]

    @property
    def outcome_df(self) -> pd.DataFrame:
        """Wide DataFrame of scaled outcome contributions per media and date.

        Returns
        -------
        pandas.DataFrame
        """
        if self._buffer.outcome_df is None:
            self._buffer.outcome_df = self._to_dataframe(value="outcome")
        return self._buffer.outcome_df.copy()

    @property
    def spend_df(self) -> pd.DataFrame:
        """Wide DataFrame of media spend per media and date.

        Returns
        -------
        pandas.DataFrame
        """
        if self._buffer.spend_df is None:
            df = self._to_dataframe(value="spend")
            df = df[["date", self.target] + self._data.media]
            self._buffer.spend_df = df
        return self._buffer.spend_df.copy()

    # ------------------------------------------------------------------
    # public getters with processing
    # ------------------------------------------------------------------
    def get_media_roas(self) -> pd.Series:
        """Compute Return on Ad Spend (ROAS) per media media over the full period.

        ROAS is defined as total outcome divided by total spend for each media.
        medias with zero spend receive a ROAS of ``NaN``.

        Returns
        -------
        pandas.Series
            Index: media name. Values: ROAS ratio.
        """
        total_outcome = self.outcome_df.drop(
            columns=[self.target, "date", "Baseline"]
        ).sum()
        total_spend = self.spend_df.drop(
            columns=[self.target, "date", "Baseline"]
        ).sum()
        roas = total_outcome / total_spend.replace(0, float("nan"))
        roas.name = "roas"
        return roas

    def get_contribution_share(self) -> pd.DataFrame:
        """Compute each component's share of total predicted outcome per date.

        Returns
        -------
        pandas.DataFrame
            Same shape as :attr:`outcome_df` (excluding the ``date`` column)
            with values expressed as fractions of the row total.
        """
        df = self.outcome_df.set_index("date")
        row_totals = df.sum(axis=1)
        return df.div(row_totals, axis=0)

    # ------------------------------------------------------------------
    # public miscellaneous utility methods
    # ------------------------------------------------------------------
    def filter_date_range(
        self,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
    ) -> Timeline:
        """Return a new :class:`Timeline` restricted to *[start, end]*.

        The new instance shares the same posterior / data but has its internal
        timeline pre-populated with only the requested dates.

        Parameters
        ----------
        start : str or pandas.Timestamp
            Inclusive lower bound (``"YYYY-MM-DD"`` or Timestamp).
        end : str or pandas.Timestamp
            Inclusive upper bound (``"YYYY-MM-DD"`` or Timestamp).

        Returns
        -------
        Timeline
            Filtered :class:`Timeline` instance.
        """
        start_ts = pd.to_datetime(start)
        end_ts = pd.to_datetime(end)

        data_copy = self._data.data.copy()
        date_series = pd.to_datetime(data_copy["date"])

        filtered_posterior = self._posterior.loc[{"date": slice(start_ts, end_ts)}]

        filtered_data = data_copy.loc[
            (date_series >= start_ts) & (date_series <= end_ts)
        ].copy()

        return Timeline(
            posterior=filtered_posterior,
            data=filtered_data,
            target_scale=self._target_scale,
        )

    def summary(self) -> pd.DataFrame:
        """Aggregate statistics (sum, mean, std) for outcome and spend per media.

        Returns
        -------
        pandas.DataFrame
            Multi-level column DataFrame with top-level keys ``"outcome"`` and
            ``"spend"``, and second-level statistics ``"sum"``, ``"mean"``,
            ``"std"``.
        """
        outcome_stats = self.outcome_df.set_index("date").agg(["sum", "mean", "std"]).T
        spend_stats = self.spend_df.set_index("date").agg(["sum", "mean", "std"]).T
        return pd.concat({"outcome": outcome_stats, "spend": spend_stats}, axis=1)
