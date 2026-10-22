"""
ROAS priors module for MMM models.

Lets the user express media knowledge as a plausible ROAS range per channel
(in original units: € of target per € spent) instead of a prior on the raw
media coefficient ``beta_media``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

from ..prior import PriorSpec


@dataclass
class RoasPriors:
    """ROAS prior specifications for a subset of media channels.

    Each channel listed in ``priors`` gets a LogNormal prior on its average
    ROAS (total contribution / total spend over the fitting period) such that
    the interval ``(a, b)`` holds a probability mass of ``coverage``.
    Channels absent from ``priors`` keep the default ``BetaPriors.media``
    prior on their coefficient.

    Parameters
    ----------
    priors : dict[str, tuple[float, float]], optional
        Mapping ``{media_name: (a, b)}`` with ``0 < a < b``, the plausible
        ROAS range of the channel in original units (€ / €).  An empty
        mapping (default) means no channel uses a ROAS prior.
    coverage : float, optional
        Probability mass of the central interval ``(a, b)``, in ``(0, 1)``.
        Defaults to ``0.9`` (i.e. ``a`` and ``b`` are the 5 % and 95 %
        quantiles).

    Raises
    ------
    ValueError
        If ``coverage`` is not in ``(0, 1)`` or if a range is not a pair of
        finite values with ``0 < a < b``.

    Examples
    --------
    >>> rp = RoasPriors({"TV": (1.0, 4.0)}, coverage=0.9)
    >>> rp.media_names
    ['TV']
    """

    priors: dict[str, tuple[float, float]] = field(default_factory=dict)
    coverage: float = 0.9

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        self._validate_coverage()
        self._validate_ranges()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_coverage(self) -> None:
        """Check that ``coverage`` is a probability in the open interval ``(0, 1)``.

        Raises
        ------
        ValueError
            If ``coverage`` is not in ``(0, 1)``.
        """
        if not 0.0 < self.coverage < 1.0:
            raise ValueError(f"coverage must be in (0, 1), got {self.coverage}")

    def _validate_ranges(self) -> None:
        """Check that every ROAS range is a pair of finite values with ``0 < a < b``.

        Raises
        ------
        ValueError
            If a range is malformed.
        """
        for name, bounds in self.priors.items():
            if len(bounds) != 2:
                raise ValueError(
                    f"ROAS range for '{name}' must be a pair (a, b), got {bounds!r}"
                )
            a, b = (float(v) for v in bounds)
            if not (np.isfinite(a) and np.isfinite(b)):
                raise ValueError(
                    f"ROAS range for '{name}' must be finite, got {bounds}"
                )
            if not 0.0 < a < b:
                raise ValueError(
                    f"ROAS range for '{name}' must satisfy 0 < a < b, got {bounds}"
                )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def media_names(self) -> list[str]:
        """Media channels that have a ROAS prior, in insertion order.

        Returns
        -------
        list[str]
            Channel names listed in ``priors``.
        """
        return list(self.priors)

    def lognormal_params(
        self, media_names: list[str] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert the ROAS ranges into LogNormal ``(mu, sigma)`` parameters.

        With ``z = Φ⁻¹((1 + coverage) / 2)``, the LogNormal whose central
        ``coverage`` interval is exactly ``(a, b)`` has
        ``mu = (ln a + ln b) / 2`` and ``sigma = (ln b - ln a) / (2 z)``.

        Parameters
        ----------
        media_names : list[str] | None, optional
            Channels (and order) for which to return the parameters.
            Defaults to :attr:`media_names`.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(mu, sigma)`` arrays aligned with ``media_names``.

        Raises
        ------
        ValueError
            If a requested channel has no ROAS prior.
        """
        names = self.media_names if media_names is None else media_names
        missing = [n for n in names if n not in self.priors]
        if missing:
            raise ValueError(f"No ROAS prior defined for channels: {missing}")

        bounds = np.array([self.priors[n] for n in names], dtype=np.float64)
        log_a, log_b = np.log(bounds[:, 0]), np.log(bounds[:, 1])
        z = norm.ppf((1.0 + self.coverage) / 2.0)

        mu = (log_a + log_b) / 2.0
        sigma = (log_b - log_a) / (2.0 * z)
        return mu, sigma

    def to_prior_spec(self, media_names: list[str] | None = None) -> PriorSpec:
        """Build the vectorized LogNormal :class:`PriorSpec` on the ROAS.

        Parameters
        ----------
        media_names : list[str] | None, optional
            Channels (and order) of the vectorized prior.
            Defaults to :attr:`media_names`.

        Returns
        -------
        PriorSpec
            ``PriorSpec("LogNormal", {"mu": mu, "sigma": sigma})`` with one
            entry per channel.
        """
        mu, sigma = self.lognormal_params(media_names)
        return PriorSpec("LogNormal", {"mu": mu, "sigma": sigma})
