"""Module for modeling utilities"""

import numpy as np
import pandas as pd


def get_uncorrelated_controls_against_reference(
    dataframe: pd.DataFrame,
    control_columns: list[str],
    reference_column: str,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Get control variables that are mostly uncorrelated with a reference control variable.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataframe containing controls.
    control_columns : list[str]
        List of control variable names.
    reference_column : str
        Column used as reference for checking correlation.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        Updated dataframe with residual columns and the list of residualized
        control column names.
    """
    updated_dataframe = dataframe.copy()
    residualized_columns: list[str] = []

    for control_column in control_columns:
        if control_column == reference_column:
            continue

        valid_mask = (
            updated_dataframe[[control_column, reference_column]].notna().all(axis=1)
        )
        # will perform : y = beta0 + beta1 * x + epsilon
        x_ref = updated_dataframe.loc[valid_mask, reference_column].to_numpy(
            dtype=float
        )
        y_ctrl = updated_dataframe.loc[valid_mask, control_column].to_numpy(dtype=float)

        design_matrix = np.column_stack([np.ones(x_ref.shape[0]), x_ref])  # =[1, x_ref]
        coefficients, *_ = np.linalg.lstsq(design_matrix, y_ctrl, rcond=None)

        residual_column = f"{control_column} ⟂"
        updated_dataframe[residual_column] = np.nan
        fitted_values = (
            coefficients[0]
            + coefficients[1] * updated_dataframe.loc[valid_mask, reference_column]
        )
        updated_dataframe.loc[valid_mask, residual_column] = (
            updated_dataframe.loc[valid_mask, control_column] - fitted_values
        )

        residualized_columns.append(residual_column)

    return updated_dataframe, residualized_columns
