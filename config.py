"""
Central configuration for the Portfolio Risk Analytics Engine.

All portfolio assumptions, market-data settings, simulation controls,
backtesting parameters, stress parameters, and output paths are defined here.

The portfolio is intentionally fixed so that every risk model, stress scenario,
risk decomposition, and backtest evaluates the same portfolio consistently.
"""

from pathlib import Path


# ============================================================
# Portfolio Definition
# ============================================================

TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "TLT",
    "HYG",
    "GLD",
]


DEFAULT_WEIGHTS = {
    "AAPL": 0.20,
    "MSFT": 0.15,
    "GOOGL": 0.15,
    "TLT": 0.20,
    "HYG": 0.15,
    "GLD": 0.15,
}


DEFAULT_PORTFOLIO_VALUE = 1_000_000.0


# ============================================================
# Market Data
# ============================================================

START_DATE = "2021-08-30"

# yfinance treats the end date as exclusive.
END_DATE = "2026-08-29"

PRICE_FIELD = "Close"


# ============================================================
# Risk Model
# ============================================================

CONFIDENCE_LEVELS = [
    0.90,
    0.95,
    0.99,
    0.995,
]


PARAMETRIC_ZERO_MEAN = True


# ============================================================
# Monte Carlo
# ============================================================

DEFAULT_MC_SIMULATIONS = 100_000

RANDOM_SEED = 42

MC_USE_SAMPLE_MEAN = True


# ============================================================
# Backtesting
# ============================================================

DEFAULT_BACKTEST_WINDOW = 252

MIN_BACKTEST_OBSERVATIONS = 252


# ============================================================
# Stress Testing
# ============================================================

EQUITY_STRESS = {
    "AAPL": -0.08,
    "MSFT": -0.08,
    "GOOGL": -0.10,
}


RATES_STRESS = {
    "TLT": -0.08,
}


CREDIT_STRESS = {
    "HYG": -0.10,
}


GOLD_STRESS = {
    "GLD": -0.05,
}


BROAD_MARKET_STRESS = -0.10


DEFAULT_STRESSED_CORRELATION = 0.85


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"

PLOTS_DIR = PROJECT_ROOT / "plots"

PRICE_CACHE = DATA_DIR / "prices.csv"