# Portfolio Risk Analytics Engine

A modular Python risk engine for a diversified multi-asset portfolio. It measures **1-day Value-at-Risk (VaR)** using three approaches, adds **Expected Shortfall**, decomposes portfolio risk, performs **scenario and correlation stress testing**, and validates Historical VaR through **rolling backtesting**.

The project also includes a Streamlit dashboard for visualizing the portfolio's risk profile and model diagnostics.

---

## Quick Start

Requires **Python 3.10 or newer**.

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the command-line risk engine:

```bash
python3 main.py
```

Run the interactive dashboard:

```bash
streamlit run app.py
```

Market prices are cached in `data/prices.csv` so the analysis can be reproduced without repeatedly downloading the data.

---

## The Portfolio

The engine analyzes a fixed **$1,000,000 multi-asset portfolio**.

| Asset | Ticker | Weight | Exposure |
|---|---|---|---|
| Apple | AAPL | 20% | Equity |
| Microsoft | MSFT | 15% | Equity |
| Alphabet | GOOGL | 15% | Equity |
| iShares 20+ Year Treasury Bond ETF | TLT | 20% | Interest Rate / Duration |
| iShares iBoxx $ High Yield Corporate Bond ETF | HYG | 15% | Credit |
| SPDR Gold Shares | GLD | 15% | Commodity |
| **Total** | | **100%** | |

### Portfolio by Risk Factor

| Risk Factor | Allocation |
|---|---|
| Equities | 50% |
| Interest Rates | 20% |
| Credit | 15% |
| Gold / Commodity | 15% |
| **Total** | **100%** |

### Why These Weights?

The weights are **intentionally designed rather than optimized**.

The objective of this project is to study portfolio risk rather than construct a minimum-volatility or maximum-Sharpe portfolio. The allocation therefore creates exposure to several distinct risk factors:

- **50% equities** — primary source of equity-market risk
- **20% TLT** — interest-rate and duration exposure
- **15% HYG** — credit and spread risk
- **15% GLD** — commodity exposure and diversification

This creates a controlled portfolio for studying how volatility, correlation, risk contribution and stress scenarios interact.

> **The weights are not presented as an optimal investment allocation.**
> They are a deliberately constructed test portfolio for the risk engine.

This distinction also allows the project to demonstrate an important risk management principle:

> **Portfolio weight is not the same thing as portfolio risk contribution.**

A 20% position does not necessarily contribute 20% of portfolio volatility. Its contribution depends on both its own volatility and its correlation with the rest of the portfolio.

---

## Data

Daily historical prices are obtained from Yahoo Finance.

| Parameter | Value |
|---|---|
| Start Date | 2021-08-30 |
| End Date | 2026-08-28 |
| Price Field | Close |
| Portfolio Value | $1,000,000 |
| Price Observations | 1,255 |
| Return Observations | 1,254 |
| Trading Days per Year | 252 |
| Backtesting Window | 252 days |
| Monte Carlo Simulations | 100,000 |
| Random Seed | 42 |

The data module:

1. Checks the local price cache
2. Validates ticker coverage
3. Cleans and aligns the price series
4. Calculates simple daily returns
5. Aggregates individual asset returns into portfolio returns

The cache is stored at `data/prices.csv`.

---

## What VaR Means

At a confidence level `c`, 1-day VaR estimates the loss threshold that should only be exceeded approximately `1-c` of the time under the model.

For example:

> A 95% 1-day VaR of $25,000 means that the model estimates only about a 5% probability of losing more than $25,000 on a single day.

VaR does **not** tell us how severe losses become once the VaR threshold is breached. That is why the project also calculates Expected Shortfall.

---

## Part A — VaR and Expected Shortfall

The engine calculates risk using three different VaR methodologies.

| Method | Distribution | Main Assumption |
|---|---|---|
| **Historical** | Observed portfolio returns | Future resembles historical sample |
| **Parametric** | Normal distribution | Returns are approximately Gaussian |
| **Monte Carlo** | 100,000 simulated returns | Simulated returns follow estimated distribution |

All three methods produce **1-day dollar VaR**. Expected Shortfall is calculated from the empirical tail of the portfolio return distribution.

### 1. Historical VaR

Historical VaR uses the actual observed portfolio return distribution.

```math
VaR_c = -Q_{1-c}(R_p)V
```

where:

- `R_p` is the portfolio return
- `Q_{1-c}` is the lower-tail empirical quantile
- `V` is portfolio value

The advantage is that no normality assumption is required. The limitation is that the model cannot observe a type of event that does not exist in the historical sample.

### 2. Parametric VaR

Parametric VaR assumes normally distributed returns.

```math
VaR_c = (z_c \sigma_p - \mu_p)V
```

where:

- `z_c` is the normal quantile
- `\sigma_p` is portfolio volatility
- `\mu_p` is mean portfolio return
- `V` is portfolio value

This approach is computationally simple, but normality can become problematic in the extreme tail.

### 3. Monte Carlo VaR

The Monte Carlo engine generates 100,000 simulated one-day portfolio returns using the historical portfolio mean and volatility.

A fixed random seed is used:

```python
RANDOM_SEED = 42
```

This makes the simulation reproducible across runs.

Monte Carlo provides a flexible simulation framework, but in this implementation the simulated portfolio return distribution remains Gaussian. Therefore:

> **Monte Carlo does not automatically mean a more realistic risk model.**

It inherits the assumptions used to generate the simulations.

### Expected Shortfall

Expected Shortfall answers the question VaR leaves open:

> **How severe are losses once the VaR threshold has already been breached?**

For historical Expected Shortfall, the model averages returns in the tail below the VaR threshold. This makes ES particularly useful when comparing portfolios with similar VaR but different tail-loss behavior.

---

## Part B — Risk Decomposition

The project decomposes total portfolio volatility into asset-level contributions.

Portfolio volatility is calculated as:

```math
\sigma_p = \sqrt{w^\top \Sigma w}
```

where:

- `w` = portfolio weights
- `\Sigma` = covariance matrix

**Marginal Risk Contribution:**

```math
MRC_i = \frac{(\Sigma w)_i}{\sigma_p}
```

**Component Risk:**

```math
CRC_i = w_i MRC_i
```

**Risk Contribution:**

```math
RC_i = \frac{CRC_i}{\sum_j CRC_j}
```

This allows the model to distinguish between **Capital Allocation** and **Risk Allocation**. That distinction is important in portfolio risk management — an asset can represent a relatively small portion of capital while contributing a disproportionately large amount of portfolio risk.

---

## Part C — Correlation Analysis

The project calculates the historical correlation matrix across all six assets. Portfolio risk is affected not only by individual asset volatility but also by cross-asset dependence.

The portfolio variance can be expressed as:

```math
\sigma_p^2 = \sum_i w_i^2 \sigma_i^2 + 2 \sum_{i < j} w_i w_j \sigma_i \sigma_j \rho_{ij}
```

The second term captures the effect of correlations. This means diversification can disappear even when the individual assets themselves do not become more volatile.

---

## Part D — Correlation Stress Test

The correlation stress test isolates the effect of increased cross-asset dependence.

The procedure is:

1. Estimate the historical covariance matrix
2. Extract each asset's historical volatility
3. Convert covariance into correlations
4. Replace all pairwise correlations with **0.85**
5. Keep individual asset volatilities unchanged
6. Rebuild the covariance matrix
7. Recalculate portfolio volatility
8. Recalculate parametric VaR

The important feature is that **individual asset volatility is held constant**. Therefore, any increase in portfolio risk comes entirely from the change in correlation.

### Interpretation

If portfolio volatility rises from the base portfolio volatility to the stressed portfolio volatility without changing individual asset volatility, the additional risk comes purely from the loss of diversification.

> **No asset needs to become individually riskier for the portfolio to become significantly riskier. Assets simply need to start moving together.**

This is one of the key risk-management insights demonstrated by the project.

---

## Part E — Scenario Stress Testing

VaR estimates risk under a statistical model. Stress testing asks a different question:

> **What happens if a specific market shock occurs?**

The project uses deterministic scenario shocks.

| Scenario | Shocked Assets |
|---|---|
| Equity Shock | AAPL, MSFT, GOOGL |
| Rates Shock | TLT |
| Credit Shock | HYG |
| Gold Shock | GLD |
| Combined Shock | All portfolio assets |

### Current Shock Assumptions

| Asset | Shock |
|---|---|
| AAPL | -8% |
| MSFT | -8% |
| GOOGL | -10% |
| TLT | -8% |
| HYG | -10% |
| GLD | -5% |

### Combined Scenario Shocks

| Asset | Combined Shock |
|---|---|
| AAPL | -8% |
| MSFT | -8% |
| GOOGL | -8% |
| TLT | -6% |
| HYG | -7% |
| GLD | -5% |

These are **illustrative stress assumptions**, not forecasts or probability estimates.

---

## Part F — VaR Backtesting

A risk model should not only produce a number — its forecasts should be evaluated against realized market outcomes. The project therefore performs a rolling Historical VaR backtest.

The current implementation uses **252 trading days** to calculate the Historical VaR forecast. The following day's realized portfolio loss is then compared with the forecast.

A VaR exception occurs when:

```math
Realized\ Loss > VaR
```

At 95% confidence, the expected exception rate is:

```math
1 - 0.95 = 5\%
```

The dashboard reports:

- Number of VaR exceptions
- Expected exception rate
- Observed exception rate
- Rolling VaR forecasts

---

## Statistical Validation

The project uses two statistical diagnostics.

### Kupiec Unconditional Coverage Test

The Kupiec test evaluates whether the observed frequency of VaR exceptions is consistent with the expected frequency.

**Null hypothesis:**

```math
H_0: p_{observed} = p_{expected}
```

A low p-value indicates evidence that the VaR model may be incorrectly calibrated.

### Christoffersen Independence Test

Correct exception frequency is not sufficient. A model could generate approximately the correct number of exceptions while still producing them in clusters.

The Christoffersen test evaluates whether exceptions are independent over time. This provides another layer of model validation beyond simply counting breaches.

---

## Results

The command-line engine reports results for four confidence levels:

- 90%
- 95%
- 99%
- 99.5%

The dashboard allows the user to select the confidence level used for the displayed VaR and backtesting analysis.

The complete command-line output includes:

- Portfolio statistics
- Historical VaR
- Parametric VaR
- Monte Carlo VaR
- Expected Shortfall
- Risk contribution
- Stress-test losses
- Correlation stress
- VaR exceptions
- Kupiec p-value
- Christoffersen p-value

### Results Table

After running `python3 main.py`, the core results can be summarized here:

| Method | VaR 90% | VaR 95% | VaR 99% | VaR 99.5% |
|---|---|---|---|---|
| Historical | $10,708 | $14,426 | $21,855 | $24,383 |
| Parametric | $11,040 | $14,301 | $20,417 | $22,656 |
| Monte Carlo | $11,113 | $14,422 | $20,571 | $22,866 |
| Expected Shortfall | $15,892 | $19,444 | $26,820 | $30,099 |

Portfolio statistics from the underlying run:

| Statistic | Value |
|---|---|
| Daily Mean Return | 0.05% |
| Daily Volatility | 0.90% |
| Annualized Return | 11.65% |
| Annualized Volatility | 14.25% |

### Risk Contribution

| Asset | Weight | Marginal Risk | Component Risk | Risk Contribution |
|---|---|---|---|---|
| AAPL | 20.00% | 0.014047 | 0.002809 | 31.30% |
| MSFT | 15.00% | 0.013280 | 0.001992 | 22.19% |
| GOOGL | 15.00% | 0.015675 | 0.002351 | 26.20% |
| TLT | 20.00% | 0.003454 | 0.000691 | 7.70% |
| HYG | 15.00% | 0.003564 | 0.000535 | 5.96% |
| GLD | 15.00% | 0.003981 | 0.000597 | 6.65% |

Equities (AAPL + MSFT + GOOGL) make up 50% of capital but roughly 80% of portfolio risk — a clear illustration that capital allocation and risk allocation are not the same thing.

### Stress Test Losses

| Scenario | Portfolio Return | Loss |
|---|---|---|
| Equity Shock | -4.30% | $43,000 |
| Rates Shock | -1.60% | $16,000 |
| Credit Shock | -1.50% | $15,000 |
| Gold Shock | -0.75% | $7,500 |
| Combined Shock | -7.00% | $70,000 |

### Correlation Stress

| Metric | Base | Stressed |
|---|---|---|
| Portfolio Volatility | 0.90% | 1.28% |
| 95% Parametric VaR | $14,763 | $21,113 |
| 99% Parametric VaR | $20,879 | $29,860 |

Volatility increased **43.01%** purely from raising pairwise correlations to 0.85, with each asset's individual volatility held constant.

### VaR Backtesting

| Metric | Value |
|---|---|
| Backtesting Window | 252 days |
| Observed Exceptions | 48 |
| Expected Exception Rate | 5.00% |
| Observed Exception Rate | 4.79% |
| Kupiec p-value | 0.7593 |
| Christoffersen p-value | 0.0296 |

The observed exception rate (4.79%) is close to the expected 5%, and the Kupiec test does not reject correct calibration (p = 0.7593). However, the Christoffersen test's low p-value (0.0296) suggests exceptions are not fully independent over time — they show some clustering rather than occurring uniformly at random.

---

## Design Decisions

### Simple Returns

The project uses simple daily returns:

```math
r_t = \frac{P_t}{P_{t-1}} - 1
```

This allows portfolio returns to be aggregated directly:

```math
r_p = \sum_i w_i r_i
```

This is consistent with the weighted portfolio construction used throughout the engine.

### One Portfolio Aggregation Method

Portfolio returns are constructed using a weighted sum of individual asset returns. The same portfolio representation is used across the risk calculations and scenario framework. This reduces the possibility of different parts of the project using inconsistent portfolio definitions.

### Fixed Portfolio Weights

The portfolio weights are stored centrally in `config.py`. The dashboard does not allow users to modify the allocation. This is intentional.

A fixed portfolio means:

- Every risk model analyzes the same portfolio
- Stress scenarios use the same exposures
- Risk contribution remains directly comparable
- Backtesting evaluates a consistent portfolio
- Results remain reproducible

The portfolio therefore acts as a **controlled experiment for portfolio risk**.

### Correlations Are Stressed Explicitly

Rather than editing covariance values arbitrarily, the correlation stress framework separates:

```
Covariance = Volatility × Correlation × Volatility
```

The stress test changes the correlation matrix while preserving the individual asset volatilities. This makes the interpretation of the stress scenario explicit.

### Monte Carlo Uses a Fixed Seed

The simulation uses `RANDOM_SEED = 42` so the results are reproducible.

### Market Data Is Cached

The project stores downloaded prices locally. This makes the analysis:

- Faster on subsequent runs
- Less dependent on network availability
- Easier to reproduce

---

## Error Handling

The project validates inputs throughout the risk engine. Examples include:

- Missing market data
- Missing tickers
- Invalid confidence levels
- Non-finite returns
- Invalid portfolio weights
- Invalid portfolio values
- Invalid Monte Carlo simulation counts
- Invalid stress shocks
- Invalid correlation assumptions
- Insufficient observations for backtesting

The objective is to fail with a meaningful error rather than silently produce an invalid risk estimate.

---

## Project Structure

```
.
├── app.py
├── main.py
├── config.py
├── data.py
├── risk.py
├── scenarios.py
├── backtesting.py
├── plots.py
├── requirements.txt
├── README.md
│
├── data/
│   └── prices.csv
│
└── plots/
    ├── 01_return_distribution.png
    ├── 02_var_method_comparison.png
    ├── 03_risk_contribution.png
    ├── 04_correlation_matrix.png
    ├── 05_stress_scenarios.png
    ├── 06_var_backtest.png
    └── 07_cumulative_returns.png
```

### Module Responsibilities

| File | Responsibility |
|---|---|
| `config.py` | Portfolio, dates, model parameters and stress assumptions |
| `data.py` | Market data loading, cleaning and return calculation |
| `risk.py` | VaR, Expected Shortfall, volatility and risk decomposition |
| `scenarios.py` | Security shocks and correlation stress |
| `backtesting.py` | Rolling Historical VaR and statistical tests |
| `plots.py` | Matplotlib risk visualizations |
| `main.py` | Command-line orchestration |
| `app.py` | Streamlit dashboard |

The analytical modules are separated from the presentation layer so that the risk calculations can be tested and reused independently of the dashboard.

---

## Visualizations

The project generates charts covering:

- **Portfolio Loss Distribution** — shows the empirical portfolio loss distribution and VaR threshold
- **VaR Method Comparison** — compares Historical, Parametric and Monte Carlo VaR across confidence levels
- **Risk Contribution** — shows how each asset contributes to total portfolio volatility
- **Correlation Matrix** — shows the historical dependence structure across portfolio assets
- **Stress Scenarios** — compares losses generated by the predefined stress scenarios
- **VaR Backtest** — compares realized losses against rolling Historical VaR forecasts and identifies VaR breaches

---

## Assumptions and Limitations

| Assumption | Applies To | Limitation |
|---|---|---|
| Returns are normally distributed | Parametric / Monte Carlo | May underestimate fat-tail risk |
| Historical sample is informative about the future | Historical VaR | Cannot capture events absent from the sample |
| Portfolio weights remain fixed | Portfolio analysis | Real portfolios may be rebalanced or drift |
| Correlations are estimated from history | Portfolio risk | Correlations can change substantially during stress |
| Stress shocks are deterministic | Scenario analysis | They describe consequences, not probabilities |
| Position can be exited within one day | 1-day VaR | Does not capture liquidity or market-impact risk |
| Historical observations are equally weighted | Historical VaR | Recent market conditions are not given additional weight |
| 252-day rolling window | VaR backtesting | Results depend on the selected window |

The project is therefore an **analytical risk framework**, not a production risk-management system.

---

## Key Risk Insights

The project is designed to demonstrate several principles that are central to portfolio risk management:

**1. Diversification depends on correlation**
Owning different securities does not guarantee diversification. If assets become highly correlated, portfolio risk can increase substantially.

**2. Capital allocation and risk allocation are different**
A 15% position does not necessarily contribute 15% of portfolio risk. Volatility and correlation determine the actual contribution.

**3. Different VaR models can produce different answers**
Historical, Parametric and Monte Carlo approaches use different assumptions and therefore can produce different estimates for the same portfolio.

**4. VaR does not describe tail severity**
Expected Shortfall complements VaR by measuring the average loss beyond the VaR threshold.

**5. Stress testing complements statistical risk measures**
VaR asks: *"How large is a loss at a specified statistical confidence level?"*
Stress testing asks: *"What happens if a particular adverse event occurs?"*
Both perspectives are necessary for understanding portfolio resilience.

**6. Risk models need validation**
A risk estimate should not simply be accepted because the number looks reasonable. Rolling backtesting and statistical tests provide evidence about whether the model behaves as expected.

---

## Technologies

- Python
- NumPy
- Pandas
- SciPy
- Matplotlib
- yfinance
- Streamlit

---

## Disclaimer

This project is for **educational, research, and quantitative risk-modeling purposes only**.

The portfolio allocation, stress scenarios, assumptions and risk estimates do not constitute investment advice or recommendations.

---

## Author

**Manav Prakash**
