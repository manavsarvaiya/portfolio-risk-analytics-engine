"""
Historical backtesting of VaR forecasts.

The backtesting module compares rolling one-day VaR estimates
against realized portfolio returns.

It reports:

    - VaR breaches
    - Exception rate
    - Kupiec unconditional coverage test
    - Christoffersen independence test

These diagnostics help distinguish a model that produces a
reasonable-looking risk number from one that actually behaves
appropriately out of sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# ============================================================
# Rolling Historical VaR
# ============================================================


def rolling_historical_var(
    returns: pd.Series,
    confidence: float,
    window: int,
    portfolio_value: float,
) -> pd.Series:
    """
    Generate one-step-ahead rolling Historical VaR forecasts.

    VaR is reported as a positive dollar loss.
    """

    if not 0 < confidence < 1:
        raise ValueError(
            "Confidence level must lie between 0 and 1."
        )

    if window <= 0:
        raise ValueError(
            "Backtesting window must be positive."
        )

    if portfolio_value <= 0:
        raise ValueError(
            "Portfolio value must be positive."
        )

    if returns is None or len(returns) == 0:
        raise ValueError(
            "Return series cannot be empty."
        )

    # Remove missing observations before
    # checking the usable sample size.
    returns = returns.dropna()

    if len(returns) <= window:
        raise ValueError(
            "Not enough observations for the "
            "requested backtesting window."
        )

    values = returns.to_numpy(
        dtype=float
    )

    if not np.isfinite(values).all():
        raise ValueError(
            "Return series contains non-finite values."
        )

    var = pd.Series(
        index=returns.index,
        dtype=float,
        name="VaR",
    )

    for i in range(
        window,
        len(returns),
    ):

        history = returns.iloc[
            i - window:i
        ]

        cutoff = np.quantile(
            history,
            1.0 - confidence,
        )

        # Positive dollar VaR.
        var.iloc[i] = (
            -cutoff
            * portfolio_value
        )

    return var


# ============================================================
# Exception Identification
# ============================================================


def identify_exceptions(
    returns: pd.Series,
    var: pd.Series,
    portfolio_value: float,
) -> pd.Series:
    """
    Identify days on which realized losses
    exceed forecast VaR.
    """

    if portfolio_value <= 0:
        raise ValueError(
            "Portfolio value must be positive."
        )

    realized_loss = (
        -returns
        * portfolio_value
    )

    exceptions = (
        realized_loss > var
    )

    return exceptions.fillna(
        False
    )


# ============================================================
# Kupiec Test
# ============================================================


def kupiec_test(
    exceptions: pd.Series,
    confidence: float,
) -> dict:
    """
    Perform the Kupiec unconditional
    coverage test.

    Null hypothesis:

        observed exception frequency
        equals expected frequency.
    """

    if not 0 < confidence < 1:
        raise ValueError(
            "Confidence level must lie between 0 and 1."
        )

    observations = len(
        exceptions
    )

    failures = int(
        exceptions.sum()
    )

    expected_rate = (
        1.0 - confidence
    )

    if observations == 0:

        return {
            "lr_stat": np.nan,
            "p_value": np.nan,
            "failures": 0,
            "observations": 0,
            "observed_rate": np.nan,
            "expected_rate": expected_rate,
        }

    observed_rate_raw = (
        failures
        / observations
    )

    eps = 1e-12

    observed_rate = np.clip(
        observed_rate_raw,
        eps,
        1 - eps,
    )

    likelihood_null = (
        (1 - expected_rate)
        ** (observations - failures)
        * expected_rate
        ** failures
    )

    likelihood_alt = (
        (1 - observed_rate)
        ** (observations - failures)
        * observed_rate
        ** failures
    )

    lr_stat = (
        -2
        * np.log(
            likelihood_null
            / likelihood_alt
        )
    )

    p_value = (
        1
        - stats.chi2.cdf(
            lr_stat,
            df=1,
        )
    )

    return {
        "lr_stat": float(
            lr_stat
        ),
        "p_value": float(
            p_value
        ),
        "failures": failures,
        "observations": observations,
        "observed_rate": float(
            observed_rate_raw
        ),
        "expected_rate": float(
            expected_rate
        ),
    }


# ============================================================
# Christoffersen Test
# ============================================================


def christoffersen_test(
    exceptions: pd.Series,
) -> dict:
    """
    Test whether VaR exceptions occur
    independently over time.

    The test uses a first-order Markov
    representation of exception states.
    """

    values = (
        exceptions
        .astype(int)
        .to_numpy()
    )

    if len(values) < 2:

        return {
            "lr_stat": np.nan,
            "p_value": np.nan,
        }

    n00 = 0
    n01 = 0
    n10 = 0
    n11 = 0

    for previous, current in zip(
        values[:-1],
        values[1:],
    ):

        if previous == 0 and current == 0:

            n00 += 1

        elif previous == 0 and current == 1:

            n01 += 1

        elif previous == 1 and current == 0:

            n10 += 1

        else:

            n11 += 1

    def safe_rate(
        numerator: int,
        denominator: int,
    ) -> float:
        """
        Safely calculate a probability.
        """

        if denominator == 0:
            return 0.5

        return np.clip(
            numerator / denominator,
            1e-12,
            1 - 1e-12,
        )

    pi01 = safe_rate(
        n01,
        n00 + n01,
    )

    pi11 = safe_rate(
        n11,
        n10 + n11,
    )

    total_exceptions = (
        n01 + n11
    )

    total_non_exceptions = (
        n00 + n10
    )

    pi = safe_rate(
        total_exceptions,
        (
            total_non_exceptions
            + total_exceptions
        ),
    )

    likelihood_independent = (
        (1 - pi)
        ** total_non_exceptions
        * pi
        ** total_exceptions
    )

    likelihood_dependent = (
        (1 - pi01)
        ** n00
        * pi01
        ** n01
        * (1 - pi11)
        ** n10
        * pi11
        ** n11
    )

    lr_stat = (
        -2
        * np.log(
            likelihood_independent
            / likelihood_dependent
        )
    )

    p_value = (
        1
        - stats.chi2.cdf(
            lr_stat,
            df=1,
        )
    )

    return {
        "lr_stat": float(
            lr_stat
        ),
        "p_value": float(
            p_value
        ),
    }


# ============================================================
# Complete Backtest
# ============================================================


def backtest_var(
    returns: pd.Series,
    confidence: float,
    window: int,
    portfolio_value: float,
) -> dict:
    """
    Run the complete rolling Historical
    VaR backtest.
    """

    var = rolling_historical_var(
        returns,
        confidence,
        window,
        portfolio_value,
    )

    exceptions = identify_exceptions(
        returns,
        var,
        portfolio_value,
    )

    valid = var.notna()

    realized = returns.loc[
        valid
    ]

    forecasts = var.loc[
        valid
    ]

    exception_series = exceptions.loc[
        valid
    ]

    kupiec = kupiec_test(
        exception_series,
        confidence,
    )

    christoffersen = (
        christoffersen_test(
            exception_series
        )
    )

    return {
        "var": forecasts,

        "var_series": forecasts,

        "realized_returns": realized,

        "exceptions": exception_series,

        "exception_rate": float(
            exception_series.mean()
        ),

        "expected_exception_rate": float(
            1.0 - confidence
        ),

        "kupiec": kupiec,

        "christoffersen": christoffersen,
    }


# ============================================================
# Public Wrapper
# ============================================================


def run_var_backtest(
    returns: pd.Series,
    confidence: float = 0.95,
    window: int = 252,
    portfolio_value: float | None = None,
) -> dict:
    """
    Public interface used by main.py.

    Portfolio value must be provided explicitly.
    """

    if portfolio_value is None:

        raise ValueError(
            "portfolio_value must be provided "
            "when running a VaR backtest."
        )

    return backtest_var(
        returns=returns,
        confidence=confidence,
        window=window,
        portfolio_value=portfolio_value,
    )