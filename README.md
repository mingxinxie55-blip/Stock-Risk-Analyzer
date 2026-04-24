# Stock Risk & Return Analyzer

Interactive stock risk-return analysis tool for **ACC102 (Track 4)**, built with **Python + WRDS + SQL + Streamlit**.

This README contains the full project documentation (design, technical implementation, user guide, and submission notes) in one place.

## 1) Project Overview

- **Course**: ACC102 AI-Driven Data Analytics
- **Track**: Track 4 Interactive Analytics Tool (Streamlit)
- **Project Name**: Stock Risk & Return Analyzer
- **Data Source**: WRDS (CSMAR) + SQL queries
- **Target Users**: learners, analysts, coursework demo audience
- **Goal**: single-stock risk/return analysis and multi-stock horizontal comparison

## 2) Required Functional Scope

### 2.1 Data Access (WRDS + SQL)

- Connect to WRDS database
- Run parameterized SQL via `db.raw_sql()`
- Support SQL clauses (`SELECT`, `FROM`, `WHERE`, `LIMIT`)
- Retrieve fields: stock code, trade date, close, volume, adjusted close
- Convert query result to Pandas `DataFrame`
- Cache query output to local CSV files under `data/`

### 2.2 Data Processing (Pandas)

- Daily return
- Cumulative return / normalized NAV
- CAGR (annualized compound return)
- Annualized volatility
- Maximum drawdown
- Sharpe ratio
- Beta and Alpha (relative to benchmark)
- Win rate (positive daily return ratio)

### 2.3 Single-Stock Analysis (Visualization)

- Normalized NAV line chart
- Daily return histogram
- Rolling volatility curve
- Historical drawdown curve
- KPI cards (return, volatility, drawdown, Sharpe, Beta, Alpha)

### 2.4 Multi-Stock Comparison (Core Highlight)

- Overlay NAV chart
- Annual return bar chart
- Volatility vs max drawdown comparison bar chart
- Risk-return quadrant scatter plot
- Six-dimension radar chart (CAGR, Volatility, Drawdown, Sharpe, Beta, Alpha)
- Rolling volatility overlay chart (30D annualized volatility)
- Return correlation heatmap
- Multi-metric dimension heatmap (normalized)
- Full metrics comparison table

### 2.5 UI Interactions (Clear Buttons/Controls)

- Multi-select stock symbols from default pool (25+ symbols)
- Add custom stock symbols
- Select period (1Y, 3Y, 5Y, custom)
- Run analysis
- Refresh WRDS data
- Toggle mode (single vs multi stock)
- Export results to Excel
- View metric definitions
- Help panel

## 3) Technical Implementation

### 3.1 Core SQL Example

```sql
SELECT
    stkcd,
    accper,
    clsprc,
    dnshrtrd,
    adjcls
FROM csmar.wrds_csmar_price
WHERE stkcd = %s
  AND accper >= %s
  AND accper <= %s
ORDER BY accper
LIMIT %s
```

### 3.2 Data Processing Formulas

- `return = adj_close.pct_change()`
- `nav = (1 + return).cumprod()`
- `cum_return = nav - 1`
- `rolling_vol_30d = return.rolling(30).std() * sqrt(252)`
- `drawdown = nav / nav.cummax() - 1`

Indicators:

- `CAGR = nav[-1]^(252/N) - 1`
- `Annual Volatility = std(return) * sqrt(252)`
- `Sharpe = mean(return) / std(return) * sqrt(252)` (risk-free rate assumed 0)
- `Win Rate = P(return > 0)`
- `Beta = Cov(r_stock, r_bench) / Var(r_bench)`
- `Alpha = (E[r_stock] - beta * E[r_bench]) * 252`

### 3.3 Engineering Notes

- App state managed with `st.session_state`
- Local cache strategy in `data/*.csv`
- Excel export to `outputs/stock_analysis_YYYY-MM-DD.xlsx`
- Fallback synthetic dataset when WRDS is unavailable (ensures demo reliability)

## 4) Codebase Structure

```text
Stock-Risk-Analyzer/
├─ README.md
├─ app.py
├─ requirements.txt
├─ data/
├─ outputs/
└─ analysis.ipynb
```

Module responsibilities:

- `app.py`: single-file implementation containing Streamlit UI, WRDS/SQL data access, data processing, visualization functions, and Excel export
- `analysis.ipynb`: coursework notebook appendix with code snapshots and explanations

## 5) Installation and Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL shown in the terminal.

## 6) User Guide

1. Input a stock symbol (example: `600519`)
2. Select symbols from the default list (supports multi-select, 25+ built-in symbols)
3. (Optional) Add a custom symbol via **Add Custom Symbol**
4. Select time range
5. Choose **Single Stock** or **Multi Stock** mode
6. (Optional) Input benchmark symbol for Beta/Alpha
7. Click **Run Analysis**
8. Review charts and metric table
9. Click **Export to Excel**

FAQ:

- **No WRDS account available?** The app still runs using fallback mock data.
- **Where is the exported file?** `outputs/stock_analysis_YYYY-MM-DD.xlsx`

## 7) Submission Checklist

Required deliverables:

- `README.md`
- `app.py`
- `requirements.txt`
- `analysis.ipynb`

Requirement mapping:

- WRDS + SQL: `app.py` (integrated data access layer)
- Data metrics: `app.py` (integrated processing functions)
- Streamlit interactions: `app.py`
- Visualization set: `app.py` (integrated chart functions)
- Export feature: `app.py`

Recommended demo flow:

1. Show single-stock analysis
2. Switch to multi-stock comparison
3. Export Excel file
4. Explain WRDS fallback design
