"""
Market-data utilities for the Portfolio Risk Analytics Engine.

Responsibilities:
- Load the cached price history from data/prices.csv.
- Optionally download missing/fresh data from Yahoo Finance.
- Validate ticker coverage and data quality.
- Calculate simple daily returns.
- Aggregate asset returns into portfolio returns when needed.
"""

from pathlib import Path
import time

import numpy as np
import pandas as pd

from config import (
    TICKERS,
    START_DATE,
    END_DATE,
    PRICE_FIELD,
    PRICE_CACHE,
)


def _clean_prices(prices, tickers):
    """Clean and validate a price DataFrame."""
    if prices is None or prices.empty:
        raise ValueError("No market-price data was returned.")

    prices = prices.copy()

    # Normalize index to DatetimeIndex.
    prices.index = pd.to_datetime(prices.index)
    prices = prices[~prices.index.duplicated(keep="last")]
    prices = prices.sort_index()

    missing = [ticker for ticker in tickers if ticker not in prices.columns]
    if missing:
        raise ValueError(
            f"Price data is missing requested ticker(s): {', '.join(missing)}."
        )

    prices = prices[tickers].apply(pd.to_numeric, errors="coerce")

    all_nan = [ticker for ticker in tickers if prices[ticker].isna().all()]
    if all_nan:
        raise ValueError(
            f"Ticker(s) contain no usable price observations: {', '.join(all_nan)}."
        )

    prices = prices.dropna(how="any")

    if prices.empty:
        raise ValueError("No common observations remain after removing missing values.")

    return prices


def _read_cache(tickers):
    """Read the local price cache if it exists."""
    if not Path(PRICE_CACHE).exists():
        return None

    try:
        cached = pd.read_csv(PRICE_CACHE, index_col=0, parse_dates=True)
    except Exception as exc:
        raise ValueError(
            f"Could not read cached price file: {PRICE_CACHE}"
        ) from exc

    # Support a cache where the date column is explicitly named Date.
    if "Date" in cached.columns:
        cached["Date"] = pd.to_datetime(cached["Date"])
        cached = cached.set_index("Date")

    return _clean_prices(cached, tickers)


def _download_prices(tickers):
    """Download price history from Yahoo Finance."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "yfinance is required to download market data. "
            "Install it with: python3 -m pip install --break-system-packages yfinance"
        ) from exc

    last_error = None

    for attempt in range(3):
        try:
            data = yf.download(
                tickers=tickers,
                start=START_DATE,
                end=END_DATE,
                auto_adjust=False,
                progress=False,
                threads=False,
            )

            if data is None or data.empty:
                raise ValueError("Yahoo Finance returned no data.")

            # yfinance returns a MultiIndex when multiple tickers are requested.
            if isinstance(data.columns, pd.MultiIndex):
                if PRICE_FIELD not in data.columns.get_level_values(0):
                    raise ValueError(
                        f"Requested price field '{PRICE_FIELD}' was not returned."
                    )
                prices = data[PRICE_FIELD].copy()
            else:
                # Single-ticker fallback.
                ticker = tickers[0]
                column = PRICE_FIELD if PRICE_FIELD in data.columns else ticker
                prices = data[[column]].copy()
                prices.columns = [ticker]

            prices = _clean_prices(prices, tickers)

            Path(PRICE_CACHE).parent.mkdir(parents=True, exist_ok=True)
            prices.to_csv(PRICE_CACHE)

            return prices

        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(1)

    raise RuntimeError(
        f"Failed to download market data after 3 attempts: {last_error}"
    ) from last_error


def load_market_data(tickers=None, refresh=False):
    """
    Load market prices for the requested tickers.

    By default, the local cache is used so the project can run reproducibly
    without network access. Set refresh=True to download fresh data.
    """
    requested = list(tickers) if tickers is not None else list(TICKERS)

    unknown = [ticker for ticker in requested if ticker not in TICKERS]
    if unknown:
        raise ValueError(
            f"Ticker(s) not defined in config.py: {', '.join(unknown)}."
        )

    if not refresh:
        cached = _read_cache(requested)
        if cached is not None:
            return cached

    return _download_prices(requested)


def calculate_returns(prices):
    """Calculate simple daily percentage returns."""
    if prices is None or prices.empty:
        raise ValueError("Cannot calculate returns from an empty price DataFrame.")

    returns = prices.pct_change().dropna(how="any")

    if returns.empty:
        raise ValueError("Price data does not contain enough observations for returns.")

    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("Calculated returns contain non-finite values.")

    return returns


def portfolio_returns(returns, weights=None):
    """
    Aggregate asset returns into portfolio returns using a weighted sum.

    If weights are not supplied, equal/default portfolio weights are intentionally
    not stored in this market-data module. Pass the live portfolio weights from
    the dashboard or another calling module.
    """
    if weights is None:
        raise ValueError(
            "Portfolio weights must be provided to portfolio_returns()."
        )

    weight_series = pd.Series(weights, dtype=float)

    missing = [
        ticker for ticker in weight_series.index
        if ticker not in returns.columns
    ]
    if missing:
        raise ValueError(
            f"Returns are missing portfolio ticker(s): {', '.join(missing)}."
        )

    weight_series = weight_series.reindex(returns.columns).fillna(0.0)

    if not np.isclose(weight_series.sum(), 1.0):
        raise ValueError(
            f"Portfolio weights must sum to 1.0, but sum to "
            f"{weight_series.sum():.6f}."
        )

    return returns.mul(weight_series, axis=1).sum(axis=1)
