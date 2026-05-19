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


class Timeline:
    """Builds and exposes a date-keyed timeline of channel and baseline contributions.

    Parameters
    ----------
    posterior : xarray.Dataset
        Posterior samples containing contribution variables.
    data : pandas.DataFrame
        Raw input data with a ``date`` column and one column per media channel.
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
        target="y",
        target_scale: float = 1.0,
        baseline_components=None,
    ) -> None:
        self._posterior = posterior
        self._data = data

        self._target = target
        self._target_scale = target_scale
        self._baseline_components = (
            baseline_components
            if baseline_components is not None
            else ["control", "yearly_seasonality"]
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
            Channel name.
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
            Wide DataFrame with a ``date`` column and one column per channel /
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
        out[self._target] = self._data[self._target].values
        return out

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def _build_contributions(self) -> xr.Dataset:
        """Compute the contributions for all channels and baseline components,
        and the baseline timeline.

        Returns
        -------
        all_contributions : xarray.Dataset
            Dataset with dimensions ``date`` and ``channel``, containing the
            contribution values for each media channel and baseline component
            (if not included in the baseline).
        baseline_timeline : xarray.DataArray
            1D array with dimension ``date``, containing the total baseline
            contribution for each date.
        """
        channel = self._posterior["channel_contribution"].mean(dim=["chain", "draw"])
        control = self._posterior["control_contribution"].mean(dim=["chain", "draw"])
        yearly_seasonality = self._posterior["yearly_seasonality_contribution"].mean(
            dim=["chain", "draw"]
        )
        intercept = self._posterior["intercept_contribution"].mean(
            dim=["chain", "draw"]
        )

        baseline_timeline = channel.sum(dim="channel") * 0 + intercept

        all_contributions = channel.copy()
        for comp in ["control", "yearly_seasonality"]:
            if comp not in self._baseline_components:
                if comp == "control":
                    control = control.rename({"control": "channel"})
                    all_contributions = xr.concat(
                        [all_contributions, control], dim="channel"
                    )
                elif comp == "yearly_seasonality":
                    yearly_seasonality = yearly_seasonality.expand_dims(
                        channel=["yearly_seasonality"]
                    ).transpose("date", "channel")

                    all_contributions = xr.concat(
                        [all_contributions, yearly_seasonality], dim="channel"
                    )
            else:
                if comp == "control":
                    baseline_timeline += control.values.sum(axis=1)
                elif comp == "yearly_seasonality":
                    baseline_timeline += yearly_seasonality.values.sum(axis=1)
                elif comp == "intercept":
                    pass  # already included
                else:
                    raise ValueError(f"Unknown baseline component: {comp}")

        return all_contributions, baseline_timeline

    def build(self) -> Timeline:
        """Compute the timeline from the posterior and cache it.

        Returns
        -------
        Timeline
            *self*, to allow method chaining.
        """
        all_contributions, baseline_timeline = self._build_contributions()

        timeline: dict[str, list[dict]] = {}
        dates = self._data["date"].values
        contrib = all_contributions.channel.values

        for i, date_val in enumerate(dates):
            entries: list[dict] = []
            date_str = pd.to_datetime(date_val).strftime("%Y-%m-%d")

            self._add_baseline_to_entries(
                entries, "Baseline", outcome=baseline_timeline[i]
            )

            for m in contrib:
                m_contri = float(all_contributions.isel(date=i).sel(channel=m).values)

                spend = 0.0
                if m in self._data.columns:
                    spend = float(self._data[m].iloc[i])

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
        return self._target

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
        """Wide DataFrame of scaled outcome contributions per channel and date.

        Returns
        -------
        pandas.DataFrame
        """
        if self._buffer.outcome_df is None:
            self._buffer.outcome_df = self._to_dataframe(value="outcome")
        return self._buffer.outcome_df.copy()

    @property
    def spend_df(self) -> pd.DataFrame:
        """Wide DataFrame of media spend per channel and date.

        Returns
        -------
        pandas.DataFrame
        """
        if self._buffer.spend_df is None:
            self._buffer.spend_df = self._to_dataframe(value="spend")
        return self._buffer.spend_df.copy()

    # ------------------------------------------------------------------
    # public getters with processing
    # ------------------------------------------------------------------
    def get_channel_roas(self) -> pd.Series:
        """Compute Return on Ad Spend (ROAS) per media channel over the full period.

        ROAS is defined as total outcome divided by total spend for each channel.
        Channels with zero spend receive a ROAS of ``NaN``.

        Returns
        -------
        pandas.Series
            Index: channel name. Values: ROAS ratio.
        """
        total_outcome = self.outcome_df.drop(
            columns=[self._target, "date", "Baseline"]
        ).sum()
        total_spend = self.spend_df.drop(
            columns=[self._target, "date", "Baseline"]
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

        data_copy = self._data.copy()
        date_series = pd.to_datetime(data_copy["date"])
        filtered_data = data_copy.loc[
            (date_series >= start_ts) & (date_series <= end_ts)
        ].copy()

        return Timeline(
            posterior=self._posterior,
            data=filtered_data,
            target_scale=self._target_scale,
        )

    def summary(self) -> pd.DataFrame:
        """Aggregate statistics (sum, mean, std) for outcome and spend per channel.

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
