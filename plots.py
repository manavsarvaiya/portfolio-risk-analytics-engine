"""
Portfolio Risk Visualization Module
------------------------------------
Creates publication-style charts for:
    - Return distributions
    - VaR model comparison
    - Risk contribution
    - Correlation structure
    - Stress scenarios
    - VaR backtesting
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

PLOT_DIR = Path("plots")
PLOT_DIR.mkdir(exist_ok=True)


def save_plot(filename):
    """
    Save figure to the project's plots directory.
    """
    path = PLOT_DIR / filename

    plt.tight_layout()
    plt.savefig(
        path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close()

    return path


# ============================================================
# 1. Return Distribution
# ============================================================

def plot_return_distribution(
    portfolio_returns,
    var_value,
    portfolio_value
):
    """
    Plot portfolio return distribution with VaR threshold.
    """

    losses = -portfolio_returns * portfolio_value

    plt.figure(figsize=(10, 6))

    plt.hist(
        losses,
        bins=70,
        density=True,
        alpha=0.75
    )

    plt.axvline(
        var_value,
        linestyle="--",
        linewidth=2,
        label=f"95% VaR = ${var_value:,.0f}"
    )

    plt.xlabel("Daily Portfolio Loss ($)")
    plt.ylabel("Density")
    plt.title("Portfolio Loss Distribution and 95% VaR")
    plt.legend()

    return save_plot("01_return_distribution.png")


# ============================================================
# 2. VaR Method Comparison
# ============================================================

def plot_var_comparison(var_results):
    """
    Compare Historical, Parametric and Monte Carlo VaR.
    """

    confidence_levels = list(var_results.keys())

    historical = [
        var_results[c]["historical"]
        for c in confidence_levels
    ]

    parametric = [
        var_results[c]["parametric"]
        for c in confidence_levels
    ]

    monte_carlo = [
        var_results[c]["monte_carlo"]
        for c in confidence_levels
    ]

    labels = [
        f"{int(c * 100)}%"
        for c in confidence_levels
    ]

    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(10, 6))

    plt.bar(
        x - width,
        historical,
        width,
        label="Historical"
    )

    plt.bar(
        x,
        parametric,
        width,
        label="Parametric"
    )

    plt.bar(
        x + width,
        monte_carlo,
        width,
        label="Monte Carlo"
    )

    plt.xticks(x, labels)
    plt.ylabel("VaR ($)")
    plt.xlabel("Confidence Level")
    plt.title("VaR Model Comparison")
    plt.legend()

    return save_plot("02_var_method_comparison.png")


# ============================================================
# 3. Risk Contribution
# ============================================================

def plot_risk_contribution(risk_table):
    """
    Plot percentage contribution of each asset
    to total portfolio volatility.
    """

    data = risk_table.copy()

    if "Risk_Contribution" in data.columns:
        contribution = data["Risk_Contribution"]
    else:
        contribution = data["Component_Risk"] / data["Component_Risk"].sum()

    contribution = contribution.sort_values()

    plt.figure(figsize=(10, 6))

    plt.barh(
        contribution.index,
        contribution.values * 100
    )

    plt.xlabel("Contribution to Portfolio Risk (%)")
    plt.ylabel("Asset")
    plt.title("Portfolio Risk Contribution")

    for i, value in enumerate(contribution.values * 100):
        plt.text(
            value,
            i,
            f" {value:.1f}%",
            va="center"
        )

    return save_plot("03_risk_contribution.png")


# ============================================================
# 4. Correlation Matrix
# ============================================================

def plot_correlation_matrix(returns):
    """
    Display asset return correlations.
    """

    correlation = returns.corr()

    plt.figure(figsize=(9, 7))

    image = plt.imshow(
        correlation,
        interpolation="nearest",
        aspect="auto"
    )

    plt.colorbar(image, label="Correlation")

    plt.xticks(
        range(len(correlation.columns)),
        correlation.columns,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        range(len(correlation.index)),
        correlation.index
    )

    plt.title("Asset Return Correlation Matrix")

    # Add values inside cells
    for i in range(len(correlation)):
        for j in range(len(correlation.columns)):
            plt.text(
                j,
                i,
                f"{correlation.iloc[i, j]:.2f}",
                ha="center",
                va="center"
            )

    return save_plot("04_correlation_matrix.png")


# ============================================================
# 5. Stress Scenarios
# ============================================================

def plot_stress_scenarios(
    stress_results,
    var_99
):
    """
    Compare scenario losses with 99% Historical VaR.
    """

    names = list(stress_results.keys())

    losses = [
        abs(stress_results[name]["loss"])
        for name in names
    ]

    x = np.arange(len(names))

    plt.figure(figsize=(11, 6))

    bars = plt.bar(
        x,
        losses
    )

    plt.axhline(
        var_99,
        linestyle="--",
        linewidth=2,
        label=f"99% VaR = ${var_99:,.0f}"
    )

    plt.xticks(
        x,
        names,
        rotation=25,
        ha="right"
    )

    plt.ylabel("Loss ($)")
    plt.title("Stress Scenario Losses vs 99% VaR")
    plt.legend()

    for bar, loss in zip(bars, losses):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"${loss:,.0f}",
            ha="center",
            va="bottom"
        )

    return save_plot("05_stress_scenarios.png")


# ============================================================
# 6. VaR Backtest
# ============================================================

def plot_var_backtest(
    portfolio_returns,
    backtest
):
    """
    Plot rolling VaR against realized portfolio losses.
    """

    var_series = backtest["var_series"]

    aligned_returns = portfolio_returns.loc[
        var_series.index
    ]

    realized_losses = -aligned_returns

    plt.figure(figsize=(12, 6))

    plt.plot(
        realized_losses.index,
        realized_losses.values,
        linewidth=1,
        label="Realized Loss"
    )

    plt.plot(
        var_series.index,
        var_series.values,
        linewidth=1.5,
        label="95% VaR"
    )

    exceptions = realized_losses > var_series

    plt.scatter(
        realized_losses.index[exceptions],
        realized_losses[exceptions],
        s=25,
        label="VaR Breach"
    )

    plt.xlabel("Date")
    plt.ylabel("Loss / VaR")
    plt.title("Historical VaR Backtest")
    plt.legend()

    return save_plot("06_var_backtest.png")


# ============================================================
# Optional: Portfolio Cumulative Return
# ============================================================

def plot_cumulative_returns(portfolio_returns):
    """
    Plot cumulative portfolio performance.
    """

    cumulative = (1 + portfolio_returns).cumprod()

    plt.figure(figsize=(11, 6))

    plt.plot(
        cumulative.index,
        cumulative.values,
        linewidth=1.5
    )

    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.title("Cumulative Portfolio Return")

    return save_plot("07_cumulative_returns.png")