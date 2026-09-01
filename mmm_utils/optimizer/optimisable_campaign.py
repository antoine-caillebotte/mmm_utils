"""This module defines the OptimisableCampaign class
and CampaignModes enum for budget optimization."""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd
import pytensor.tensor as pt
import xarray as xr
from pytensor.xtensor.type import as_xtensor

from .optimizer_utils import (
    replace_variable_by_optimization_variable,
    replace_variable_by_repeated_optimization_variable,
    replace_variable_by_sparse_optimization_variable,
)


class CampaignModes(Enum):
    """Campaign modes for budget optimization."""

    CONSTANT = 1
    PLAIN = 2
    SPARSE = 3


@dataclass
class OptimisableCampaign:
    """Campaign configuration for budget optimization.

    Parameters
    ----------
    starting_date : pd.Timestamp
        Starting date of the campaign period.
    period : int
        Length of the campaign period in time steps (e.g., weeks).
    budget_by_media : dict
        Mapping from media channel name to initial/reference budget value.
    mode : CampaignModes, optional
        "CONSTANT" for a single allocation per media channel,
        "PLAIN" for a different allocation per media channel and period,
        or "SPARSE" for a sparse allocation strategy. Default is "CONSTANT".
    """

    starting_date: pd.Timestamp
    period: int
    budget_by_media: dict[str, float]
    mode: CampaignModes = CampaignModes.CONSTANT
    last_campaign: pd.DataFrame | None = None
    budget_template: xr.DataArray = field(init=False)

    def __post_init__(self):
        if self.mode == CampaignModes.SPARSE:
            if self.last_campaign is None:
                raise ValueError(
                    "For 'sparse' mode, last_campaign must be provided to determine "
                    "the last allocation date."
                )

            assert all(
                col in self.last_campaign.columns for col in self.budget_by_media
            ), "All media channels in budget_by_media must be present in last_campaign.columns."

            assert (
                self.starting_date > self.last_campaign.index.min()
            ), "Starting date must be after the first date in last_campaign for 'sparse' mode."
            assert (
                self.starting_date + pd.Timedelta(weeks=self.period)
                <= self.last_campaign.index.max()
            ), "Ending date must be before the last date in last_campaign for 'sparse' mode."

        self.budget_template = self._create_budget_template()

    def format_budget_optimized(self, budget_optimized: np.ndarray):
        """Format the optimized budget into a DataFrame with dates and media channels.

        Parameters
        ----------
        budget_optimized : np.ndarray
            Optimized budget array, either 1D (constant mode) or 2D (plain/sparse mode).

        Returns
        -------
        formated_budget : np.ndarray
            Formatted budget array with shape ``(campaign_period, n_media)``.
        """
        if self.mode == CampaignModes.CONSTANT:
            formated_budget = np.tile(budget_optimized, (self.period, 1))
        elif self.mode == CampaignModes.SPARSE:
            flat_indices = np.flatnonzero(self.budget_template)
            formated_budget = np.zeros_like(self.budget_template.values)
            formated_budget.flat[flat_indices] = budget_optimized
        else:
            formated_budget = budget_optimized.reshape(
                (self.period, len(self.budget_by_media))
            )

        return formated_budget

    def create_control_variable(self, model):
        """Create a control variable for optimization.

        Parameters
        ----------
        model : MMM
            The fitted model to be optimized.

        Returns
        -------
        xtensor
            The control variable for optimization.
        """
        campaign_index = pd.date_range(
            start=self.starting_date,
            periods=self.period,
            freq="W",
        )

        control_data = np.asarray(model["control_data"].eval(), dtype=float)
        n_controls = control_data.shape[1]
        control_names = list(model.coords.get("control", range(n_controls)))

        if self.mode == CampaignModes.CONSTANT:
            control_values = np.repeat(control_data[[-1], :], self.period, axis=0)
        else:
            if control_data.shape[0] >= self.period:
                control_values = control_data[-self.period :, :]
            else:
                control_values = np.repeat(control_data[[-1], :], self.period, axis=0)

        control_template = xr.DataArray(
            control_values,
            coords={"date": campaign_index, "control": control_names},
            dims=["date", "control"],
        )

        control_xt = as_xtensor(
            pt.as_tensor_variable(control_template.values),
            dims=control_template.dims,
            name="control_data",
        )

        return control_xt

    def _create_budget_template(self):
        """Create a budget template DataArray for optimization variable injection.

        Returns
        -------
        xarray.DataArray
            Budget template with dimensions ``date`` and ``media``.
        """
        if self.mode == CampaignModes.CONSTANT:
            index = pd.DatetimeIndex([self.starting_date])
        elif self.mode in [CampaignModes.PLAIN, CampaignModes.SPARSE]:
            index = pd.date_range(
                start=self.starting_date, periods=self.period, freq="W"
            )

        else:
            raise ValueError(
                f"Invalid campaign mode: {self.mode}. Expected one of: "
                + ", ".join(CampaignModes.__members__.keys())
            )

        budget = pd.DataFrame(
            self.budget_by_media,
            index=index,
        )

        if self.mode == CampaignModes.SPARSE:
            budget[self.last_campaign.loc[index] == 0] = 0

        budget_template = xr.DataArray(
            budget.values,
            coords={"media": list(budget.columns), "date": budget.index},
            dims=["date", "media"],
        )

        print(f"✅ Budget template created :\n\t{budget}")

        return budget_template

    def create_initial_flattened_budget(self):
        """Create the initial flattened budget vector for optimization.

        Returns
        -------
        np.ndarray
            Flattened budget vector, either 1D (constant mode) or 2D (plain/sparse mode).
        """
        if self.mode == CampaignModes.CONSTANT:
            x0 = np.array(list(self.budget_by_media.values()), dtype=float)
        elif self.mode == CampaignModes.SPARSE:
            flat_indices = np.flatnonzero(self.budget_template)
            x0 = self.budget_template.values.flatten()[flat_indices]
        else:
            x0 = self.budget_template.values.flatten()

        return x0.flatten()

    def create_optimization_variables(self, model):
        """Build optimization variables and objective from a budget input.

        Parameters
        ----------
        model : MMM
            The fitted model to be optimized.

        Returns
        -------
        tuple[pytensor.graph.basic.Variable, pytensor.graph.basic.Variable]
            A tuple ``(optimizable_target, optimizable_budget)`` where:

            - ``optimizable_target`` is the differentiable scalar objective
              (negative mean of ``total_media_contribution``).
            - ``optimizable_budget`` is the flattened optimization variable
              injected into the model in place of ``channel_data``.
        """

        # 1. Create control variable
        extra_replacements = {"control_data": self.create_control_variable(model)}

        # 2. Replace channel_data with optimization variable
        replace_variable = {
            CampaignModes.CONSTANT: replace_variable_by_repeated_optimization_variable,
            CampaignModes.PLAIN: replace_variable_by_optimization_variable,
            CampaignModes.SPARSE: replace_variable_by_sparse_optimization_variable,
        }
        optimizable_budget, optimizable_model = replace_variable[self.mode](
            model,
            "channel_data",
            self.budget_template,
            extra_replacements=extra_replacements,
        )

        return optimizable_budget, optimizable_model
