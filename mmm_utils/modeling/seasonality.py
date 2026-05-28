"""Seasonality modeling utilities for MMM."""

import numpy as np

PERIOD = 365.25 / 7


def fourier_features(
    n: int,
    order: int,
    fourier_dim: str = "fourier_mode",
) -> np.ndarray:
    """Generate Fourier seasonal basis functions.

    Parameters
    ----------
    n : int
        Number of time points.
    order : int
        Fourier order. The output has ``2 * order`` columns.
    fourier_dim : str, optional
        Kept for backward API compatibility.

    Returns
    -------
    np.ndarray
        Matrix of shape ``(n, 2 * order)`` containing sine/cosine terms.
    """

    del fourier_dim
    if order <= 0:
        return np.empty((n, 0), dtype=np.float64)

    time = np.arange(n, dtype=np.float64)[:, None]
    k = np.arange(1, order + 1, dtype=np.float64)[None, :]
    x = 2.0 * np.pi * k * time / PERIOD

    return np.concatenate([np.sin(x), np.cos(x)], axis=1)


if __name__ == "__main__":
    ORDER = 2
    features = fourier_features(10, ORDER)
    print(features)

    beta_season = np.random.normal(0.0, 0.5, size=(ORDER * 2,))
