"""
Portfolio Risk Analytics Dashboard
-----------------------------------
Interactive Streamlit dashboard for:
    - Portfolio allocation
    - Portfolio volatility
    - VaR and Expected Shortfall
    - Risk contribution
    - Correlation analysis
    - Stress testing
    - VaR backtesting
"""

import streamlit as st
import pandas as pd
import numpy as np

from config import (
    TICKERS,
    DEFAULT_WEIGHTS,
    DEFAULT_PORTFOLIO_VALUE,
    CONFIDENCE_LEVELS,
    DEFAULT_MC_SIMULATIONS,
    RANDOM_SEED,
    DEFAULT_BACKTEST_WINDOW,
    EQUITY_STRESS,
    RATES_STRESS,
    CREDIT_STRESS,
    GOLD_STRESS,
    DEFAULT_STRESSED_CORRELATION,
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


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Portfolio Risk Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Custom CSS
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #777777;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    .section-title {
        font-size: 1.5rem;
        font-weight: 650;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Header
# ============================================================

st.markdown(
    '<div class="main-title">Portfolio Risk Analytics Engine</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Market risk measurement, stress testing and VaR model validation"
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# Sidebar
# ============================================================

st.sidebar.title("Risk Controls")

st.sidebar.metric(
    "Portfolio Value",
    f"${DEFAULT_PORTFOLIO_VALUE:,.0f}",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Portfolio Allocation")

# Fixed portfolio allocation from config.py.
# The portfolio is intentionally non-customizable so the dashboard uses
# one consistent portfolio throughout all risk calculations.
weights = DEFAULT_WEIGHTS.copy()

allocation_df = pd.DataFrame(
    {
        "Asset": list(weights.keys()),
        "Weight": list(weights.values()),
    }
)

st.sidebar.dataframe(
    allocation_df.style.format({"Weight": "{:.0%}"}),
    hide_index=True,
    width="stretch",
)

st.sidebar.markdown("---")

confidence = st.sidebar.selectbox(
    "VaR Confidence Level",
    CONFIDENCE_LEVELS,
    index=0,
    format_func=lambda x: f"{x:.0%}",
)


# ============================================================
# Load Data
# ============================================================

@st.cache_data
def get_data():
    prices = load_market_data(TICKERS)
    returns = calculate_returns(prices)
    return prices, returns


with st.spinner("Loading market data..."):
    prices, returns = get_data()


# ============================================================
# Portfolio Returns
# ============================================================

weight_series = pd.Series(weights, dtype=float)

portfolio_returns = (
    returns[weight_series.index]
    .mul(weight_series, axis=1)
    .sum(axis=1)
)


# ============================================================
# Portfolio Statistics
# ============================================================

daily_mean = portfolio_returns.mean()
daily_vol = portfolio_returns.std()
annualized_vol = daily_vol * np.sqrt(252)
annualized_return = daily_mean * 252


# ============================================================
# VaR
# ============================================================

historical = historical_var(
    portfolio_returns,
    confidence,
    DEFAULT_PORTFOLIO_VALUE,
)

parametric = parametric_var(
    portfolio_returns.mean(),
    portfolio_returns.std(),
    confidence,
    DEFAULT_PORTFOLIO_VALUE,
)

monte_carlo = monte_carlo_var(
    portfolio_returns,
    confidence,
    DEFAULT_PORTFOLIO_VALUE,
    DEFAULT_MC_SIMULATIONS,
    RANDOM_SEED,
)

es = expected_shortfall(
    portfolio_returns,
    confidence,
    DEFAULT_PORTFOLIO_VALUE,
)


# ============================================================
# KPI SECTION
# ============================================================

st.markdown(
    '<div class="section-title">Portfolio Risk Overview</div>',
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Daily Volatility",
        f"{daily_vol:.2%}",
    )

with col2:
    st.metric(
        "Annualized Volatility",
        f"{annualized_vol:.2%}",
    )

with col3:
    st.metric(
        f"{confidence:.0%} Historical VaR",
        f"${historical:,.0f}",
    )

with col4:
    st.metric(
        f"{confidence:.0%} Expected Shortfall",
        f"${es:,.0f}",
    )


# ============================================================
# Portfolio Information
# ============================================================

st.markdown(
    '<div class="section-title">Portfolio Composition</div>',
    unsafe_allow_html=True,
)

col1, col2 = st.columns([1, 2])

with col1:
    st.dataframe(
        allocation_df.style.format({"Weight": "{:.1%}"}),
        hide_index=True,
        width="stretch",
    )

with col2:
    st.bar_chart(
        allocation_df.set_index("Asset")["Weight"]
    )


# ============================================================
# Portfolio Statistics
# ============================================================

st.markdown(
    '<div class="section-title">Portfolio Performance Statistics</div>',
    unsafe_allow_html=True,
)

stats_df = pd.DataFrame(
    {
        "Metric": [
            "Daily Mean Return",
            "Annualized Return",
            "Daily Volatility",
            "Annualized Volatility",
        ],
        "Value": [
            daily_mean,
            annualized_return,
            daily_vol,
            annualized_vol,
        ],
    }
)

stats_display = stats_df.copy()
stats_display["Value"] = stats_display["Value"].map(
    lambda x: f"{x:.2%}"
)

st.dataframe(
    stats_display,
    hide_index=True,
    width="stretch",
)


# ============================================================
# VaR Model Comparison
# ============================================================

st.markdown(
    '<div class="section-title">VaR Model Comparison</div>',
    unsafe_allow_html=True,
)

var_comparison = pd.DataFrame(
    {
        "Model": [
            "Historical",
            "Parametric",
            "Monte Carlo",
            "Expected Shortfall",
        ],
        "Risk ($)": [
            historical,
            parametric,
            monte_carlo,
            es,
        ],
    }
)

st.bar_chart(
    var_comparison.set_index("Model")
)

st.dataframe(
    var_comparison.style.format({"Risk ($)": "${:,.0f}"}),
    hide_index=True,
    width="stretch",
)


# ============================================================
# Risk Contribution
# ============================================================

st.markdown(
    '<div class="section-title">Risk Contribution</div>',
    unsafe_allow_html=True,
)

risk_table = risk_contribution_table(
    returns,
    weights,
)

display_table = risk_table.copy()

if "Weight" in display_table.columns:
    display_table["Weight"] = display_table["Weight"].map(
        lambda x: f"{x:.2%}"
    )

if "Risk_Contribution" in display_table.columns:
    display_table["Risk_Contribution"] = display_table[
        "Risk_Contribution"
    ].map(
        lambda x: f"{x:.2%}"
    )

st.dataframe(
    display_table,
    width="stretch",
)


# ============================================================
# Correlation Analysis
# ============================================================

st.markdown(
    '<div class="section-title">Correlation Structure</div>',
    unsafe_allow_html=True,
)

correlation = returns.corr()

st.dataframe(
    correlation.style.format("{:.2f}"),
    width="stretch",
)


# ============================================================
# Correlation Stress
# ============================================================

st.markdown(
    '<div class="section-title">Correlation Stress Test</div>',
    unsafe_allow_html=True,
)

correlation_result = correlation_stress(
    returns,
    weights,
    stressed_correlation=DEFAULT_STRESSED_CORRELATION,
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Base Volatility",
        f"{correlation_result['base_volatility']:.2%}",
    )

with col2:
    st.metric(
        "Stressed Volatility",
        f"{correlation_result['stressed_volatility']:.2%}",
    )

with col3:
    st.metric(
        "Volatility Increase",
        f"{correlation_result['volatility_change']:.2%}",
    )

st.write(
    f"Correlation shock: all pairwise correlations are set to "
    f"{DEFAULT_STRESSED_CORRELATION:.2f} while individual asset volatilities "
    "are held constant."
)

corr_var_df = pd.DataFrame(
    {
        "Confidence": ["95%", "99%"],
        "Base Parametric VaR": [
            correlation_result["base_var_95"],
            correlation_result["base_var_99"],
        ],
        "Stressed Parametric VaR": [
            correlation_result["stressed_var_95"],
            correlation_result["stressed_var_99"],
        ],
    }
)

st.dataframe(
    corr_var_df.style.format(
        {
            "Base Parametric VaR": "${:,.0f}",
            "Stressed Parametric VaR": "${:,.0f}",
        }
    ),
    hide_index=True,
    width="stretch",
)


# ============================================================
# Stress Testing
# ============================================================

st.markdown(
    '<div class="section-title">Scenario Stress Testing</div>',
    unsafe_allow_html=True,
)

scenario_definitions = {
    "Equity Shock": EQUITY_STRESS,
    "Rates Shock": RATES_STRESS,
    "Credit Shock": CREDIT_STRESS,
    "Gold Shock": GOLD_STRESS,
    "Combined Shock": {
        "AAPL": -0.08,
        "MSFT": -0.08,
        "GOOGL": -0.08,
        "TLT": -0.06,
        "HYG": -0.07,
        "GLD": -0.05,
    },
}

stress_rows = []

for name, shocks in scenario_definitions.items():

    result = apply_shock(
        name,
        shocks,
        weights,
        DEFAULT_PORTFOLIO_VALUE,
    )

    stress_rows.append(
        {
            "Scenario": name,
            "Portfolio Return": result.portfolio_return,
            "Loss": result.loss,
        }
    )

stress_df = pd.DataFrame(stress_rows)

display_stress = stress_df.copy()

display_stress["Portfolio Return"] = display_stress[
    "Portfolio Return"
].map(
    lambda x: f"{x:.2%}"
)

display_stress["Loss"] = display_stress[
    "Loss"
].map(
    lambda x: f"${x:,.0f}"
)

st.dataframe(
    display_stress,
    hide_index=True,
    width="stretch",
)

st.bar_chart(
    stress_df.set_index("Scenario")["Loss"]
)


# ============================================================
# VaR Backtesting
# ============================================================

st.markdown(
    '<div class="section-title">VaR Backtesting</div>',
    unsafe_allow_html=True,
)

backtest = run_var_backtest(
    portfolio_returns,
    confidence=confidence,
    window=DEFAULT_BACKTEST_WINDOW,
    portfolio_value=DEFAULT_PORTFOLIO_VALUE,
)

exception_count = int(backtest["exceptions"].sum())

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Backtest Window",
        f"{DEFAULT_BACKTEST_WINDOW} days",
    )

with col2:
    st.metric(
        "VaR Exceptions",
        f"{exception_count}",
    )

with col3:
    st.metric(
        "Observed Exception Rate",
        f"{backtest['exception_rate']:.2%}",
    )

with col4:
    st.metric(
        "Expected Exception Rate",
        f"{1 - confidence:.2%}",
    )


# ============================================================
# Statistical Validation
# ============================================================

st.markdown(
    '<div class="section-title">Statistical Diagnostics</div>',
    unsafe_allow_html=True,
)

test_results = pd.DataFrame(
    {
        "Test": [
            "Kupiec Unconditional Coverage",
            "Christoffersen Independence",
        ],
        "Statistic": [
            backtest["kupiec"]["lr_stat"],
            backtest["christoffersen"]["lr_stat"],
        ],
        "p-value": [
            backtest["kupiec"]["p_value"],
            backtest["christoffersen"]["p_value"],
        ],
    }
)

st.dataframe(
    test_results.style.format(
        {
            "Statistic": "{:.4f}",
            "p-value": "{:.4f}",
        }
    ),
    hide_index=True,
    width="stretch",
)


# ============================================================
# VaR Forecast Chart
# ============================================================

st.markdown(
    '<div class="section-title">Rolling VaR Backtest</div>',
    unsafe_allow_html=True,
)

backtest_chart = pd.DataFrame(
    {
        "Realized Return": backtest["realized_returns"] * DEFAULT_PORTFOLIO_VALUE,
        "VaR Threshold": -backtest["var_series"],
    }
)

st.line_chart(backtest_chart)


# ============================================================
# Portfolio Return Chart
# ============================================================

st.markdown(
    '<div class="section-title">Portfolio Return History</div>',
    unsafe_allow_html=True,
)

cumulative = (1 + portfolio_returns).cumprod()

st.line_chart(cumulative)


# ============================================================
# Footer
# ============================================================

st.markdown("---")

st.caption(
    "Portfolio Risk Analytics Engine | "
    "Historical, Parametric and Monte Carlo VaR | "
    "Expected Shortfall | Stress Testing | VaR Backtesting"
)
