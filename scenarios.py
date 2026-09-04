"""
Scenario analysis for portfolio risk.

Unlike VaR, stress testing does not assign a probability to an event. Instead,
it asks what the portfolio would lose if a specified market shock occurred.

The module supports:

    - security-specific shocks
    - uniform market shocks
    - cross-asset stress
    - correlation shocks
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from data import portfolio_returns


class ScenarioError(Exception):
    """Raised when scenario inputs are invalid."""


@dataclass(frozen=True)
class ScenarioResult:
    """Container for the result of a single stress scenario."""

    name: str
    portfolio_return: float
    pnl: float

    @property
    def loss(self) -> float:
        """Return the scenario loss as a positive dollar amount."""

        return -self.pnl


# ============================================================
# Security-Level Stress Testing
# ============================================================


def apply_shock(
    name: str,
    shocks: dict[str, float],
    weights: dict[str, float],
    portfolio_value: float,
) -> ScenarioResult:
    """
    Apply security-specific return shocks to a portfolio.

    Assets that are not explicitly included in ``shocks`` receive a
    0% return shock.

    Example:

        shocks = {
            "AAPL": -0.08,
            "MSFT": -0.08,
            "GOOGL": -0.10,
        }

    means AAPL, MSFT and GOOGL are shocked while all other assets
    remain unchanged.
    """

    if not weights:
        raise ScenarioError(
            "Portfolio weights cannot be empty."
        )

    if portfolio_value <= 0:
        raise ScenarioError(
            "Portfolio value must be positive."
        )

    # Validate weights.
    weight_sum = sum(weights.values())

    if not np.isclose(weight_sum, 1.0, atol=1e-8):
        raise ScenarioError(
            f"Portfolio weights must sum to 1.0. Got {weight_sum:.6f}."
        )

    # Validate shock values.
    for ticker, shock in shocks.items():

        if ticker not in weights:
            raise ScenarioError(
                f"Shock specified for asset '{ticker}', "
                f"but it is not in the portfolio."
            )

        if not np.isfinite(shock):
            raise ScenarioError(
                f"Shock for {ticker} must be finite."
            )

    tickers = list(weights.keys())

    # Assets not explicitly shocked receive a 0% return.
    shocked_returns = pd.DataFrame(
        [[shocks.get(ticker, 0.0) for ticker in tickers]],
        columns=tickers,
    )

    portfolio_return = float(
        portfolio_returns(
            shocked_returns,
            weights,
        ).iloc[0]
    )

    pnl = portfolio_return * portfolio_value

    return ScenarioResult(
        name=name,
        portfolio_return=portfolio_return,
        pnl=pnl,
    )


# ============================================================
# Uniform Stress
# ============================================================


def uniform_shock(
    name: str,
    tickers: list[str],
    shock: float,
    weights: dict[str, float],
    portfolio_value: float,
) -> ScenarioResult:
    """
    Apply an identical return shock to every portfolio holding.
    """

    if not np.isfinite(shock):
        raise ScenarioError(
            "Uniform shock must be finite."
        )

    return apply_shock(
        name,
        {ticker: shock for ticker in tickers},
        weights,
        portfolio_value,
    )


# ============================================================
# Covariance / Correlation Utilities
# ============================================================


def volatility_vector(
    covariance: pd.DataFrame,
) -> np.ndarray:
    """
    Extract individual security volatilities from covariance matrix Σ.
    """

    if covariance.empty:
        raise ScenarioError(
            "Covariance matrix cannot be empty."
        )

    diagonal = np.diag(
        covariance.to_numpy()
    )

    if np.any(diagonal < -1e-12):
        raise ScenarioError(
            "Covariance matrix contains negative variances."
        )

    # Protect against tiny negative floating-point values.
    diagonal = np.maximum(
        diagonal,
        0.0,
    )

    return np.sqrt(diagonal)


def correlation_matrix(
    covariance: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert covariance estimates into correlations.
    """

    vols = volatility_vector(
        covariance
    )

    if np.any(vols <= 0):
        raise ScenarioError(
            "Correlation cannot be calculated for zero-volatility assets."
        )

    corr = (
        covariance.to_numpy()
        / np.outer(vols, vols)
    )

    return pd.DataFrame(
        corr,
        index=covariance.index,
        columns=covariance.columns,
    )


def rebuild_covariance(
    volatilities: np.ndarray,
    correlations: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reconstruct covariance matrix Σ from σ and ρ.
    """

    if len(volatilities) != len(correlations):
        raise ScenarioError(
            "Volatility vector and correlation matrix dimensions do not match."
        )

    covariance = (
        np.outer(
            volatilities,
            volatilities,
        )
        * correlations.to_numpy()
    )

    return pd.DataFrame(
        covariance,
        index=correlations.index,
        columns=correlations.columns,
    )


# ============================================================
# Correlation Stress
# ============================================================


def correlation_stress(
    returns: pd.DataFrame,
    weights: dict[str, float],
    stressed_correlation: float = 0.85,
) -> dict:
    """
    Stress portfolio correlations while keeping individual asset
    volatilities unchanged.

    The function:

        1. Calculates the historical covariance matrix.
        2. Extracts individual asset volatilities.
        3. Replaces all pairwise correlations with the target value.
        4. Rebuilds the covariance matrix.
        5. Calculates base and stressed portfolio volatility.
        6. Calculates the corresponding parametric VaR.

    This isolates the effect of increased cross-asset dependence
    on portfolio risk.
    """

    if returns.empty:
        raise ScenarioError(
            "Return data cannot be empty."
        )

    if not weights:
        raise ScenarioError(
            "Portfolio weights cannot be empty."
        )

    if not -1 <= stressed_correlation <= 1:
        raise ScenarioError(
            "Stressed correlation must lie between -1 and 1."
        )

    weight_series = pd.Series(
        weights,
        dtype=float,
    )

    missing = [
        ticker
        for ticker in weight_series.index
        if ticker not in returns.columns
    ]

    if missing:
        raise ScenarioError(
            f"Return data is missing assets: {', '.join(missing)}"
        )

    weight_sum = weight_series.sum()

    if not np.isclose(
        weight_sum,
        1.0,
        atol=1e-8,
    ):
        raise ScenarioError(
            f"Portfolio weights must sum to 1.0. Got {weight_sum:.6f}."
        )

    asset_returns = returns[
        weight_series.index
    ].dropna()

    if len(asset_returns) < 2:
        raise ScenarioError(
            "At least two return observations are required."
        )

    # Historical covariance matrix.
    covariance = asset_returns.cov()

    # Historical individual volatilities.
    vols = volatility_vector(
        covariance
    )

    # Historical correlations.
    base_correlation = correlation_matrix(
        covariance
    )

    # Build stressed correlation matrix.
    stressed_values = np.full(
        base_correlation.shape,
        stressed_correlation,
        dtype=float,
    )

    # Assets remain perfectly correlated with themselves.
    np.fill_diagonal(
        stressed_values,
        1.0,
    )

    stressed_correlation_matrix = pd.DataFrame(
        stressed_values,
        index=base_correlation.index,
        columns=base_correlation.columns,
    )

    # Rebuild covariance using the same individual volatilities.
    stressed_covariance = rebuild_covariance(
        vols,
        stressed_correlation_matrix,
    )

    weights_array = weight_series.to_numpy()

    # Base portfolio variance:
    # w'Σw
    base_variance = float(
        weights_array
        @ covariance.to_numpy()
        @ weights_array
    )

    # Stressed portfolio variance:
    # w'Σ_stressed w
    stressed_variance = float(
        weights_array
        @ stressed_covariance.to_numpy()
        @ weights_array
    )

    base_volatility = np.sqrt(
        max(base_variance, 0.0)
    )

    stressed_volatility = np.sqrt(
        max(stressed_variance, 0.0)
    )

    volatility_change = (
        stressed_volatility / base_volatility - 1.0
        if base_volatility > 0
        else np.nan
    )

    # Parametric VaR under a zero-mean normal assumption.
    z_95 = stats.norm.ppf(0.95)
    z_99 = stats.norm.ppf(0.99)

    base_var_95 = (
        z_95
        * base_volatility
    )

    stressed_var_95 = (
        z_95
        * stressed_volatility
    )

    base_var_99 = (
        z_99
        * base_volatility
    )

    stressed_var_99 = (
        z_99
        * stressed_volatility
    )

    return {
        "base_covariance": covariance,
        "stressed_covariance": stressed_covariance,
        "base_correlation": base_correlation,
        "stressed_correlation": stressed_correlation_matrix,
        "base_volatility": float(base_volatility),
        "stressed_volatility": float(stressed_volatility),
        "volatility_change": float(volatility_change),
        "base_var_95": float(base_var_95 * 1_000_000),
        "stressed_var_95": float(stressed_var_95 * 1_000_000),
        "base_var_99": float(base_var_99 * 1_000_000),
        "stressed_var_99": float(stressed_var_99 * 1_000_000),
    }