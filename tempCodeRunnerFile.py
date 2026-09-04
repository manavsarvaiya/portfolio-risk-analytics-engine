"""
Portfolio Risk Analytics Engine
--------------------------------
Main orchestration layer for:
    - Market data acquisition
    - Portfolio statistics
    - VaR and Expected Shortfall
    - Risk decomposition
    - Stress testing
    - Correlation stress testing
    - VaR backtesting
    - Visualization
"""

import numpy as np
import pandas as pd

from config import (
    WEIGHTS,
    PORTFOLIO_VALUE,
    CONFIDENCE_LEVELS,
    MC_SIMULATIONS,
    RANDOM_SEED,
    BACKTEST_WINDOW,
    EQUITY_STRESS,
    RATES_STRESS,
    CREDIT_STRESS,
    GOLD_STRESS,
    STRESSED_CORRELATION,
)

from data import load_market_data, calculate_returns

from risk import (
    historical_var,
    parametric_var,
    monte_carlo_var,
    expected_shortfall,
    risk_contribution_table,
)

from scenarios import apply_shock, correlation_stress

from backtesting import run_var_backtest

from plots import (
    plot_return_distribution,
    plot_var_comparison,
    plot_risk_contribution,
    plot_correlation_matrix,
    plot_stress_scenarios,
    plot_var_backtest,
)


# ============================================================
# Utility Functions
# ============================================================


def format_currency(value):
    """Format a numeric value as USD."""
    return f"${value:,.0f}"


def format_pct(value):
    """Format a decimal return as a percentage."""
    return f"{value * 100:.2f}%"


def print_section(title):
    """Print a formatted report section."""
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


# ============================================================
# Portfolio Statistics
# ============================================================


def calculate_portfolio_statistics(returns):
    """
    Calculate portfolio-level return and volatility statistics.
    """

    weights = pd.Series(WEIGHTS, dtype=float)

    # Ensure the return data contains the portfolio assets.
    missing = [ticker for ticker in weights.index if ticker not in returns.columns]

    if missing:
        raise ValueError(
            f"Return data is missing portfolio assets: {', '.join(missing)}"
        )

    portfolio_returns = (
        returns[weights.index]
        .mul(weights, axis=1)
        .sum(axis=1)
    )

    daily_mean = portfolio_returns.mean()
    daily_vol = portfolio_returns.std()

    annualized_return = daily_mean * 252
    annualized_vol = daily_vol * np.sqrt(252)

    return {
        "portfolio_returns": portfolio_returns,
        "daily_mean": daily_mean,
        "daily_vol": daily_vol,
        "annualized_return": annualized_return,
        "annualized_vol": annualized_vol,
    }


# ============================================================
# VaR Analysis
# ============================================================


def calculate_var_metrics(portfolio_returns):
    """
    Calculate VaR and Expected Shortfall using three approaches.
    """

    results = {}

    for confidence in CONFIDENCE_LEVELS:

        historical = historical_var(
            portfolio_returns,
            confidence,
            PORTFOLIO_VALUE,
        )

        parametric = parametric_var(
            portfolio_returns.mean(),
            portfolio_returns.std(),
            confidence,
            PORTFOLIO_VALUE,
        )

        monte_carlo = monte_carlo_var(
            portfolio_returns,
            confidence,
            PORTFOLIO_VALUE,
            MC_SIMULATIONS,
            RANDOM_SEED,
        )

        es = expected_shortfall(
            portfolio_returns,
            confidence,
            PORTFOLIO_VALUE,
        )

        results[confidence] = {
            "historical": historical,
            "parametric": parametric,
            "monte_carlo": monte_carlo,
            "expected_shortfall": es,
        }

    return results


# ============================================================
# Stress Testing
# ============================================================


def calculate_stress_results():

    scenarios = {
        "Equity Shock": EQUITY_STRESS,
        "Rates Shock": RATES_STRESS,
        "Credit Shock": CREDIT_STRESS,
        "Gold Shock": GOLD_STRESS,
    }

    results = {}

    for name, shocks in scenarios.items():

        result = apply_shock(
            name,
            shocks,
            WEIGHTS,
            PORTFOLIO_VALUE,
        )

        results[name] = {
            "return": result.portfolio_return,
            "loss": result.loss,
        }

    return results


# ============================================================
# Main Analysis
# ============================================================


def run_analysis(generate_plots=True):

    print_section("WEIGHTS RISK ANALYTICS ENGINE")

    print(f"Portfolio Value: {format_currency(PORTFOLIO_VALUE)}")
    print(f"Number of Assets: {len(WEIGHTS)}")
    print("")

    print("Portfolio Allocation:")

    for ticker, weight in WEIGHTS.items():
        print(f"  {ticker:<8} {weight:.1%}")

    # --------------------------------------------------------
    # Market Data
    # --------------------------------------------------------

    print_section("MARKET DATA")

    prices = load_market_data(list(WEIGHTS.keys()))

    returns = calculate_returns(prices)

    print(f"Observations: {len(prices):,}")
    print(f"Return Observations: {len(returns):,}")
    print(f"Start Date: {prices.index[0].date()}")
    print(f"End Date:   {prices.index[-1].date()}")

    # --------------------------------------------------------
    # Portfolio Statistics
    # --------------------------------------------------------

    statistics = calculate_portfolio_statistics(returns)

    portfolio_returns = statistics["portfolio_returns"]

    print_section("PORTFOLIO RISK")

    print(
        f"Daily Mean Return:       "
        f"{format_pct(statistics['daily_mean'])}"
    )

    print(
        f"Daily Volatility:        "
        f"{format_pct(statistics['daily_vol'])}"
    )

    print(
        f"Annualized Return:       "
        f"{format_pct(statistics['annualized_return'])}"
    )

    print(
        f"Annualized Volatility:   "
        f"{format_pct(statistics['annualized_vol'])}"
    )

    # --------------------------------------------------------
    # VaR
    # --------------------------------------------------------

    var_results = calculate_var_metrics(portfolio_returns)

    print_section("VALUE AT RISK & EXPECTED SHORTFALL")

    for confidence, metrics in var_results.items():

        print(f"\nConfidence Level: {confidence:.0%}")

        print(
            f"  Historical VaR:       "
            f"{format_currency(metrics['historical'])}"
        )

        print(
            f"  Parametric VaR:       "
            f"{format_currency(metrics['parametric'])}"
        )

        print(
            f"  Monte Carlo VaR:      "
            f"{format_currency(metrics['monte_carlo'])}"
        )

        print(
            f"  Expected Shortfall:   "
            f"{format_currency(metrics['expected_shortfall'])}"
        )

    # --------------------------------------------------------
    # Risk Contribution
    # --------------------------------------------------------

    print_section("RISK CONTRIBUTION")

    risk_table = risk_contribution_table(
        returns,
        WEIGHTS,
    )

    print(
        risk_table[
            [
                "Weight",
                "Marginal_Risk",
                "Component_Risk",
                "Risk_Contribution",
            ]
        ].to_string(
            formatters={
                "Weight": "{:.2%}".format,
                "Marginal_Risk": "{:.6f}".format,
                "Component_Risk": "{:.6f}".format,
                "Risk_Contribution": "{:.2%}".format,
            }
        )
    )

    # --------------------------------------------------------
    # Stress Testing
    # --------------------------------------------------------

    stress_results = calculate_stress_results()

    print_section("STRESS TESTING")

    for scenario, result in stress_results.items():

        print(
            f"{scenario:<20}"
            f" Return: {format_pct(result['return']):>10}"
            f" | Loss: {format_currency(result['loss']):>12}"
        )

    # --------------------------------------------------------
    # Correlation Stress
    # --------------------------------------------------------

    print_section("CORRELATION STRESS")

    corr_result = correlation_stress(
        returns,
        WEIGHTS,
        stressed_correlation=STRESSED_CORRELATION,
    )

    print(
        f"Base Portfolio Volatility:       "
        f"{format_pct(corr_result['base_volatility'])}"
    )

    print(
        f"Stressed Portfolio Volatility:   "
        f"{format_pct(corr_result['stressed_volatility'])}"
    )

    print(
        f"Volatility Increase:              "
        f"{format_pct(corr_result['volatility_change'])}"
    )

    print(
        f"95% Parametric VaR Increase:      "
        f"{format_currency(corr_result['base_var_95'])}"
        f" -> "
        f"{format_currency(corr_result['stressed_var_95'])}"
    )

    print(
        f"99% Parametric VaR Increase:      "
        f"{format_currency(corr_result['base_var_99'])}"
        f" -> "
        f"{format_currency(corr_result['stressed_var_99'])}"
    )

    # --------------------------------------------------------
    # VaR Backtesting
    # --------------------------------------------------------

    print_section("VAR BACKTESTING")

    backtest = run_var_backtest(
        portfolio_returns,
        confidence=0.95,
        window=BACKTEST_WINDOW,
        portfolio_value=PORTFOLIO_VALUE,
    )

    print(f"Backtesting Window: {BACKTEST_WINDOW} days")

    print(
        f"Observed Exceptions: "
        f"{backtest['exceptions'].sum()}"
    )

    print(
        f"Expected Exception Rate: "
        f"{1 - 0.95:.2%}"
    )

    print(
        f"Observed Exception Rate: "
        f"{backtest['exception_rate']:.2%}"
    )

    print(
        f"Kupiec p-value: "
        f"{backtest['kupiec']['p_value']:.4f}"
    )

    print(
        f"Christoffersen p-value: "
        f"{backtest['christoffersen']['p_value']:.4f}"
    )

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------

    if generate_plots:

        print_section("GENERATING VISUALIZATIONS")

        plot_return_distribution(
            portfolio_returns,
            var_results[0.95]["historical"],
            PORTFOLIO_VALUE,
        )

        plot_var_comparison(
            var_results
        )

        plot_risk_contribution(
            risk_table
        )

        plot_correlation_matrix(
            returns
        )

        plot_stress_scenarios(
            stress_results,
            var_results[0.99]["historical"],
        )

        plot_var_backtest(
            portfolio_returns,
            backtest,
        )

        print("Plots saved to the plots/ directory.")

    print_section("ANALYSIS COMPLETE")

    return {
        "prices": prices,
        "returns": returns,
        "portfolio_returns": portfolio_returns,
        "statistics": statistics,
        "var_results": var_results,
        "risk_contribution": risk_table,
        "stress_results": stress_results,
        "correlation_stress": corr_result,
        "backtest": backtest,
    }


# ============================================================
# Entry Point
# ============================================================


if __name__ == "__main__":
    run_analysis()