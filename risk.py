"""
Portfolio risk measurement engine.

Provides:

- Portfolio statistics
- Historical VaR
- Parametric VaR
- Monte Carlo VaR
- Expected Shortfall
- Marginal / Component risk contributions
- Portfolio P&L conversion

The module is intentionally independent of configuration and data acquisition.
All portfolio-specific assumptions are passed into the functions explicitly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# ============================================================
# Exceptions
# ============================================================


class RiskError(Exception):
    """Raised when a risk calculation receives invalid inputs."""


# ============================================================
# Validation
# ============================================================


def _validate_confidence(confidence: float) -> None:
    """Validate a confidence level."""

    if not 0.0 < confidence < 1.0:
        raise RiskError(
            f"Confidence must lie between 0 and 1. "
            f"Received {confidence}."
        )


def _validate_portfolio_returns(
    returns: pd.Series,
) -> None:
    """Validate a portfolio return series."""

    if returns is None or len(returns) == 0:
        raise RiskError(
            "Cannot calculate risk from an empty return series."
        )

    values = returns.to_numpy(
        dtype=float
    )

    if not np.isfinite(values).all():
        raise RiskError(
            "Return series contains non-finite values."
        )


def _validate_portfolio_value(
    portfolio_value: float,
) -> None:
    """Validate portfolio value."""

    if portfolio_value <= 0:
        raise RiskError(
            "Portfolio value must be positive."
        )


# ============================================================
# Portfolio Statistics
# ============================================================


def calculate_portfolio_statistics(
    returns: pd.DataFrame,
    weights: dict,
) -> dict:
    """
    Calculate portfolio return and volatility statistics.

    Parameters
    ----------
    returns:
        DataFrame containing individual asset returns.

    weights:
        Dictionary mapping ticker to portfolio weight.
    """

    if returns is None or returns.empty:
        raise RiskError(
            "Returns DataFrame is empty."
        )

    if not weights:
        raise RiskError(
            "Portfolio weights cannot be empty."
        )

    weight_series = pd.Series(
        weights,
        dtype=float,
    )

    if not np.isclose(
        weight_series.sum(),
        1.0,
        atol=1e-8,
    ):
        raise RiskError(
            "Portfolio weights must sum to 1."
        )

    missing = [
        ticker
        for ticker in weight_series.index
        if ticker not in returns.columns
    ]

    if missing:
        raise RiskError(
            "Returns are missing portfolio ticker(s): "
            + ", ".join(missing)
        )

    portfolio_returns = (
        returns[weight_series.index]
        .mul(weight_series, axis=1)
        .sum(axis=1)
    )

    if not np.isfinite(
        portfolio_returns.to_numpy(
            dtype=float
        )
    ).all():
        raise RiskError(
            "Portfolio returns contain non-finite values."
        )

    daily_mean = float(
        portfolio_returns.mean()
    )

    daily_vol = float(
        portfolio_returns.std()
    )

    return {
        "portfolio_returns": portfolio_returns,
        "daily_mean": daily_mean,
        "daily_vol": daily_vol,
        "annualized_return": daily_mean * 252,
        "annualized_vol": daily_vol * np.sqrt(252),
    }


# ============================================================
# Covariance and Volatility
# ============================================================


def covariance_matrix(
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Estimate the sample covariance matrix.
    """

    if returns is None or returns.empty:
        raise RiskError(
            "Returns DataFrame is empty."
        )

    if len(returns) < 2:
        raise RiskError(
            "At least two observations are required."
        )

    return returns.cov()


def portfolio_volatility(
    covariance: pd.DataFrame,
    weights: np.ndarray,
) -> float:
    """
    Calculate portfolio volatility using:

        sigma_p = sqrt(w' Sigma w)
    """

    weights = np.asarray(
        weights,
        dtype=float,
    )

    sigma = covariance.to_numpy(
        dtype=float
    )

    if sigma.shape != (
        len(weights),
        len(weights),
    ):
        raise RiskError(
            "Covariance dimensions do not "
            "match portfolio weights."
        )

    variance = float(
        weights @ sigma @ weights
    )

    if variance < -1e-14:
        raise RiskError(
            "Calculated portfolio variance is negative."
        )

    return float(
        np.sqrt(
            max(variance, 0.0)
        )
    )


# ============================================================
# Historical VaR
# ============================================================


def historical_var(
    returns: pd.Series,
    confidence: float,
    portfolio_value: float,
) -> float:
    """
    Calculate Historical VaR from the empirical
    portfolio return distribution.
    """

    _validate_confidence(
        confidence
    )

    _validate_portfolio_returns(
        returns
    )

    _validate_portfolio_value(
        portfolio_value
    )

    threshold = np.quantile(
        returns.to_numpy(
            dtype=float
        ),
        1.0 - confidence,
    )

    return float(
        -threshold * portfolio_value
    )


def empirical_var(
    returns: pd.Series,
    confidence: float,
    portfolio_value: float,
) -> float:
    """
    Alias for historical_var.
    """

    return historical_var(
        returns,
        confidence,
        portfolio_value,
    )


# ============================================================
# Parametric VaR
# ============================================================


def parametric_var(
    mean_return: float,
    volatility: float,
    confidence: float,
    portfolio_value: float,
) -> float:
    """
    Calculate one-day Gaussian VaR.
    """

    _validate_confidence(
        confidence
    )

    _validate_portfolio_value(
        portfolio_value
    )

    if volatility < 0:
        raise RiskError(
            "Volatility cannot be negative."
        )

    if not np.isfinite(
        mean_return
    ):
        raise RiskError(
            "Mean return must be finite."
        )

    if not np.isfinite(
        volatility
    ):
        raise RiskError(
            "Volatility must be finite."
        )

    z_score = stats.norm.ppf(
        confidence
    )

    return float(
        (
            z_score * volatility
            - mean_return
        )
        * portfolio_value
    )


def parametric_var_zero_mean(
    volatility: float,
    confidence: float,
    portfolio_value: float,
) -> float:
    """
    Convenience wrapper for zero-mean
    parametric VaR.
    """

    return parametric_var(
        0.0,
        volatility,
        confidence,
        portfolio_value,
    )


# ============================================================
# Monte Carlo VaR
# ============================================================


def simulate_portfolio_returns(
    covariance: pd.DataFrame,
    weights: np.ndarray,
    mean: np.ndarray,
    simulations: int,
    seed: int,
) -> pd.Series:
    """
    Generate correlated Monte Carlo portfolio returns.
    """

    if simulations <= 0:
        raise RiskError(
            "Simulation count must be positive."
        )

    weights = np.asarray(
        weights,
        dtype=float,
    )

    mean = np.asarray(
        mean,
        dtype=float,
    )

    if len(mean) != len(weights):
        raise RiskError(
            "Mean vector and weights have "
            "different dimensions."
        )

    if covariance.shape != (
        len(weights),
        len(weights),
    ):
        raise RiskError(
            "Covariance dimensions do not "
            "match weights."
        )

    rng = np.random.default_rng(
        seed
    )

    simulated_assets = (
        rng.multivariate_normal(
            mean=mean,
            cov=covariance.to_numpy(
                dtype=float
            ),
            size=simulations,
        )
    )

    return pd.Series(
        simulated_assets @ weights,
        name="simulated_return",
    )


def monte_carlo_var(
    returns: pd.Series,
    confidence: float,
    portfolio_value: float,
    simulations: int,
    seed: int,
) -> float:
    """
    Monte Carlo VaR.

    The input is a portfolio return series.
    Its historical mean and volatility are used
    to simulate normally distributed one-day
    portfolio returns.
    """

    _validate_confidence(
        confidence
    )

    _validate_portfolio_returns(
        returns
    )

    _validate_portfolio_value(
        portfolio_value
    )

    if simulations <= 0:
        raise RiskError(
            "Simulation count must be positive."
        )

    volatility = float(
        returns.std()
    )

    if volatility < 0 or not np.isfinite(
        volatility
    ):
        raise RiskError(
            "Portfolio volatility must be finite "
            "and non-negative."
        )

    rng = np.random.default_rng(
        seed
    )

    simulated_returns = rng.normal(
        loc=float(returns.mean()),
        scale=volatility,
        size=simulations,
    )

    threshold = np.quantile(
        simulated_returns,
        1.0 - confidence,
    )

    return float(
        -threshold * portfolio_value
    )


# ============================================================
# Expected Shortfall
# ============================================================


def expected_shortfall(
    returns: pd.Series,
    confidence: float,
    portfolio_value: float,
) -> float:
    """
    Calculate historical Expected Shortfall.

    ES is the average loss in the empirical
    return tail beyond the VaR threshold.
    """

    _validate_confidence(
        confidence
    )

    _validate_portfolio_returns(
        returns
    )

    _validate_portfolio_value(
        portfolio_value
    )

    values = returns.to_numpy(
        dtype=float
    )

    threshold = np.quantile(
        values,
        1.0 - confidence,
    )

    tail = values[
        values <= threshold
    ]

    if len(tail) == 0:
        raise RiskError(
            "The empirical tail contains "
            "no observations."
        )

    return float(
        -tail.mean()
        * portfolio_value
    )


def empirical_expected_shortfall(
    returns: pd.Series,
    confidence: float,
    portfolio_value: float,
) -> float:
    """
    Alias for expected_shortfall.
    """

    return expected_shortfall(
        returns,
        confidence,
        portfolio_value,
    )


def parametric_expected_shortfall(
    volatility: float,
    confidence: float,
    portfolio_value: float,
) -> float:
    """
    Calculate Expected Shortfall under
    a normal return distribution.
    """

    _validate_confidence(
        confidence
    )

    _validate_portfolio_value(
        portfolio_value
    )

    if volatility < 0:
        raise RiskError(
            "Volatility cannot be negative."
        )

    if not np.isfinite(
        volatility
    ):
        raise RiskError(
            "Volatility must be finite."
        )

    z_score = stats.norm.ppf(
        confidence
    )

    multiplier = (
        stats.norm.pdf(z_score)
        / (1.0 - confidence)
    )

    return float(
        multiplier
        * volatility
        * portfolio_value
    )


# ============================================================
# Combined VaR Metrics
# ============================================================


def calculate_var_metrics(
    portfolio_returns: pd.Series,
    portfolio_value: float,
    confidence_levels: list[float],
    simulations: int,
    seed: int,
) -> dict:
    """
    Calculate Historical, Parametric,
    Monte Carlo VaR and Expected Shortfall.
    """

    if not confidence_levels:
        raise RiskError(
            "Confidence levels cannot be empty."
        )

    results = {}

    for confidence in confidence_levels:

        results[confidence] = {
            "historical": historical_var(
                portfolio_returns,
                confidence,
                portfolio_value,
            ),

            "parametric": parametric_var(
                float(
                    portfolio_returns.mean()
                ),
                float(
                    portfolio_returns.std()
                ),
                confidence,
                portfolio_value,
            ),

            "monte_carlo": monte_carlo_var(
                portfolio_returns,
                confidence,
                portfolio_value,
                simulations,
                seed,
            ),

            "expected_shortfall": expected_shortfall(
                portfolio_returns,
                confidence,
                portfolio_value,
            ),
        }

    return results


# ============================================================
# Risk Decomposition
# ============================================================


def marginal_risk_contribution(
    covariance: pd.DataFrame,
    weights: np.ndarray,
) -> np.ndarray:
    """
    Calculate:

        MRC_i = (Sigma w)_i / sigma_p
    """

    weights = np.asarray(
        weights,
        dtype=float,
    )

    sigma = covariance.to_numpy(
        dtype=float
    )

    portfolio_sigma = portfolio_volatility(
        covariance,
        weights,
    )

    if portfolio_sigma == 0:
        raise RiskError(
            "Cannot decompose a "
            "zero-volatility portfolio."
        )

    return (
        sigma @ weights
    ) / portfolio_sigma


def component_risk_contribution(
    covariance: pd.DataFrame,
    weights: np.ndarray,
) -> np.ndarray:
    """
    Calculate component risk contribution:

        CRC_i = w_i * MRC_i
    """

    weights = np.asarray(
        weights,
        dtype=float,
    )

    return (
        weights
        * marginal_risk_contribution(
            covariance,
            weights,
        )
    )


def risk_contribution_table(
    returns_or_covariance: pd.DataFrame,
    weights: dict | np.ndarray,
) -> pd.DataFrame:
    """
    Return security-level volatility contributions.

    Accepts either:

        risk_contribution_table(
            returns_dataframe,
            weights_dict
        )

    or:

        risk_contribution_table(
            covariance_matrix,
            weights_array
        )
    """

    if isinstance(
        weights,
        dict,
    ):

        weight_series = pd.Series(
            weights,
            dtype=float,
        )

        if not np.isclose(
            weight_series.sum(),
            1.0,
            atol=1e-8,
        ):
            raise RiskError(
                "Portfolio weights must sum to 1."
            )

        missing = [
            ticker
            for ticker in weight_series.index
            if ticker not in returns_or_covariance.columns
        ]

        if missing:
            raise RiskError(
                "Returns are missing portfolio ticker(s): "
                + ", ".join(missing)
            )

        covariance = (
            returns_or_covariance[
                weight_series.index
            ].cov()
        )

        weight_array = (
            weight_series.to_numpy(
                dtype=float
            )
        )

        labels = list(
            weight_series.index
        )

    else:

        covariance = (
            returns_or_covariance
        )

        weight_array = np.asarray(
            weights,
            dtype=float,
        )

        labels = list(
            covariance.index
        )

    marginal = (
        marginal_risk_contribution(
            covariance,
            weight_array,
        )
    )

    component = (
        component_risk_contribution(
            covariance,
            weight_array,
        )
    )

    total = float(
        component.sum()
    )

    if total != 0:

        contribution_pct = (
            component / total
        )

    else:

        contribution_pct = (
            np.zeros_like(component)
        )

    return pd.DataFrame(
        {
            "Weight": weight_array,

            "Marginal_Risk": marginal,

            "Component_Risk": component,

            "Risk_Contribution": (
                contribution_pct
            ),

            # Human-readable aliases.
            "Marginal Risk": marginal,

            "Component Risk": component,

            "Risk Contribution %": (
                contribution_pct
            ),
        },
        index=labels,
    )


# ============================================================
# P&L
# ============================================================


def to_pnl(
    returns: pd.Series,
    portfolio_value: float,
) -> pd.Series:
    """
    Convert percentage returns into
    dollar P&L.
    """

    _validate_portfolio_value(
        portfolio_value
    )

    _validate_portfolio_returns(
        returns
    )

    return returns * portfolio_value