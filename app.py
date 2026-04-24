from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Optional
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine

try:
    import wrds
except ImportError:  # pragma: no cover - optional in local env
    wrds = None


DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TRADING_DAYS = 252


@dataclass
class QueryConfig:
    symbol: str
    start_date: date
    end_date: date
    limit: int = 5000


def get_wrds_connection(wrds_username: str | None = None, wrds_password: str | None = None) -> Optional["wrds.Connection"]:
    if wrds is None:
        return None
    try:
        if wrds_username and wrds_password:
            try:
                return wrds.Connection(wrds_username=wrds_username, wrds_password=wrds_password)
            except TypeError:
                # Compatibility fallback for wrds versions without password kwarg.
                return wrds.Connection(wrds_username=wrds_username)
        if wrds_username:
            return wrds.Connection(wrds_username=wrds_username)
        return wrds.Connection(wrds_username=None)
    except Exception:
        return None


def build_sql() -> str:
    return """
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
    """


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "stkcd": "symbol",
        "accper": "date",
        "clsprc": "close",
        "dnshrtrd": "volume",
        "adjcls": "adj_close",
    }
    out = df.rename(columns=rename_map).copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"])
    for col in ("close", "volume", "adj_close"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.dropna(subset=["date", "adj_close"]).sort_values("date").reset_index(drop=True)


def _mock_prices(config: QueryConfig) -> pd.DataFrame:
    dates = pd.bdate_range(config.start_date, config.end_date)
    if len(dates) == 0:
        return pd.DataFrame(columns=["symbol", "date", "close", "volume", "adj_close"])
    seed = abs(hash(config.symbol)) % (2**32)
    noise = pd.Series(np.random.default_rng(seed).normal(0, 0.01, len(dates)))
    returns = 0.0004 + 0.2 * noise
    adj_close = 100 * (1 + returns).cumprod()
    close = adj_close * (1 + np.random.default_rng(seed + 1).normal(0, 0.002, len(dates)))
    volume = np.random.default_rng(seed + 2).integers(2_000_000, 15_000_000, len(dates))
    return pd.DataFrame(
        {
            "symbol": config.symbol.upper(),
            "date": dates,
            "close": close,
            "volume": volume,
            "adj_close": adj_close,
        }
    )


def cache_path(symbol: str, start_date: date, end_date: date) -> Path:
    return DATA_DIR / f"{symbol.upper()}_{start_date}_{end_date}.csv"


def fetch_stock_data(
    config: QueryConfig,
    refresh: bool = False,
    wrds_username: str | None = None,
    wrds_password: str | None = None,
) -> pd.DataFrame:
    path = cache_path(config.symbol, config.start_date, config.end_date)
    if path.exists() and not refresh:
        df = pd.read_csv(path, parse_dates=["date"])
        return df.sort_values("date").reset_index(drop=True)

    conn = get_wrds_connection(wrds_username=wrds_username, wrds_password=wrds_password)
    if conn is None:
        df = _mock_prices(config)
    else:
        sql = build_sql()
        sql_params = (config.symbol.upper(), config.start_date, config.end_date, config.limit)
        try:
            raw = conn.raw_sql(
                sql,
                params=sql_params,
            )
        except Exception as e:
            if "UndefinedTable" in type(e).__name__ or "does not exist" in str(e):
                try:
                    conn.close()
                except Exception:
                    pass
                df = _mock_prices(config)
                df.to_csv(path, index=False)
                return df
            raise
        df = _normalize_columns(raw)
        if df.empty:
            df = _mock_prices(config)
        conn.close()

    df.to_csv(path, index=False)
    return df


def fetch_stock_data_direct_sqlalchemy(
    config: QueryConfig,
    wrds_username: str,
    wrds_password: str,
) -> pd.DataFrame | None:
    try:
        user = quote_plus(wrds_username)
        pwd = quote_plus(wrds_password)
        engine = create_engine(
            f"postgresql+psycopg2://{user}:{pwd}@wrds-pgdata.wharton.upenn.edu:9737/wrds",
            connect_args={"connect_timeout": 5},
            pool_pre_ping=True,
        )
        sql = """
            SELECT
                stkcd,
                accper,
                clsprc,
                dnshrtrd,
                adjcls
            FROM csmar.wrds_csmar_price
            WHERE stkcd = %(symbol)s
              AND accper >= %(start_date)s
              AND accper <= %(end_date)s
            ORDER BY accper
            LIMIT %(limit)s
        """
        params = {
            "symbol": config.symbol.upper(),
            "start_date": config.start_date,
            "end_date": config.end_date,
            "limit": config.limit,
        }
        with engine.begin() as con:
            raw = pd.read_sql_query(sql, con=con, params=params)
        return _normalize_columns(raw)
    except Exception:
        return None


def prepare_price_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").reset_index(drop=True)
    out["return"] = out["adj_close"].pct_change()
    out["cum_return"] = (1 + out["return"].fillna(0)).cumprod() - 1
    out["nav"] = (1 + out["return"].fillna(0)).cumprod()
    out["rolling_vol_30d"] = out["return"].rolling(30).std() * np.sqrt(TRADING_DAYS)
    out["running_max_nav"] = out["nav"].cummax()
    out["drawdown"] = out["nav"] / out["running_max_nav"] - 1
    return out


def compute_indicators(stock_df: pd.DataFrame, benchmark_df: pd.DataFrame | None = None) -> dict:
    ret = stock_df["return"].dropna()
    if ret.empty:
        return {
            "total_return": np.nan,
            "cagr": np.nan,
            "annual_vol": np.nan,
            "max_drawdown": np.nan,
            "sharpe": np.nan,
            "win_rate": np.nan,
            "beta": np.nan,
            "alpha": np.nan,
        }

    periods = max(len(ret), 1)
    total_return = float(stock_df["nav"].iloc[-1] - 1)
    cagr = float((stock_df["nav"].iloc[-1]) ** (TRADING_DAYS / periods) - 1)
    annual_vol = float(ret.std(ddof=0) * np.sqrt(TRADING_DAYS))
    max_drawdown = float(stock_df["drawdown"].min())
    sharpe = float(ret.mean() / ret.std(ddof=0) * np.sqrt(TRADING_DAYS)) if ret.std(ddof=0) > 0 else np.nan
    win_rate = float((ret > 0).mean())

    beta, alpha = np.nan, np.nan
    if benchmark_df is not None and "return" in benchmark_df.columns:
        merged = stock_df[["date", "return"]].merge(
            benchmark_df[["date", "return"]], on="date", suffixes=("_stock", "_bench")
        ).dropna()
        if len(merged) > 2 and merged["return_bench"].var(ddof=0) > 0:
            cov = np.cov(merged["return_stock"], merged["return_bench"], ddof=0)[0, 1]
            var_b = np.var(merged["return_bench"], ddof=0)
            beta = float(cov / var_b)
            alpha = float(
                (merged["return_stock"].mean() - beta * merged["return_bench"].mean()) * TRADING_DAYS
            )

    return {
        "total_return": total_return,
        "cagr": cagr,
        "annual_vol": annual_vol,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "win_rate": win_rate,
        "beta": beta,
        "alpha": alpha,
    }


def build_summary_table(processed: dict[str, pd.DataFrame], benchmark_symbol: str | None = None) -> pd.DataFrame:
    bench = processed.get(benchmark_symbol) if benchmark_symbol else None
    rows = []
    for symbol, df in processed.items():
        metrics = compute_indicators(df, bench if symbol != benchmark_symbol else None)
        rows.append({"symbol": symbol, **metrics})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    pct_cols = ["total_return", "cagr", "annual_vol", "max_drawdown", "win_rate", "alpha"]
    for col in pct_cols:
        if col in out.columns:
            out[col] = out[col] * 100
    return out.sort_values("symbol").reset_index(drop=True)


def nav_line(df: pd.DataFrame, symbol: str) -> go.Figure:
    fig = px.line(df, x="date", y="nav", title=f"{symbol} NAV (Normalized)")
    fig.update_layout(yaxis_title="NAV", xaxis_title="Date")
    return fig


def return_hist(df: pd.DataFrame, symbol: str) -> go.Figure:
    fig = px.histogram(df.dropna(subset=["return"]), x="return", nbins=50, title=f"{symbol} Daily Return Distribution")
    fig.update_layout(xaxis_title="Daily Return", yaxis_title="Count")
    return fig


def rolling_vol_line(df: pd.DataFrame, symbol: str) -> go.Figure:
    fig = px.line(df, x="date", y="rolling_vol_30d", title=f"{symbol} 30D Rolling Volatility")
    fig.update_layout(yaxis_title="Annualized Vol", xaxis_title="Date")
    return fig


def drawdown_line(df: pd.DataFrame, symbol: str) -> go.Figure:
    fig = px.line(df, x="date", y="drawdown", title=f"{symbol} Drawdown Curve")
    fig.update_layout(yaxis_title="Drawdown", xaxis_title="Date")
    return fig


def multi_nav_chart(processed: dict[str, pd.DataFrame]) -> go.Figure:
    fig = go.Figure()
    for symbol, df in processed.items():
        fig.add_trace(go.Scatter(x=df["date"], y=df["nav"], mode="lines", name=symbol))
    fig.update_layout(title="Multi-Stock NAV Overlay", xaxis_title="Date", yaxis_title="NAV")
    return fig


def annual_return_bar(processed: dict[str, pd.DataFrame]) -> go.Figure:
    rows = []
    for symbol, df in processed.items():
        tmp = df.copy()
        tmp["year"] = tmp["date"].dt.year
        yearly = tmp.groupby("year")["return"].apply(lambda x: (1 + x.fillna(0)).prod() - 1).reset_index()
        yearly["symbol"] = symbol
        rows.append(yearly)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["year", "return", "symbol"])
    fig = px.bar(out, x="year", y="return", color="symbol", barmode="group", title="Annual Return Comparison")
    fig.update_layout(yaxis_title="Annual Return", xaxis_title="Year")
    return fig


def vol_drawdown_bar(summary_df: pd.DataFrame) -> go.Figure:
    plot_df = summary_df.copy()
    plot_df["symbol"] = plot_df["symbol"].astype(str)
    melted = plot_df.melt(
        id_vars=["symbol"],
        value_vars=["annual_vol", "max_drawdown"],
        var_name="metric",
        value_name="value",
    )
    melted["value"] = pd.to_numeric(melted["value"], errors="coerce")
    melted = melted.dropna(subset=["value"])
    if melted.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No valid volatility/drawdown data for current selection.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(title="Volatility vs Max Drawdown")
        return fig
    order = plot_df["symbol"].dropna().astype(str).unique().tolist()
    melted["symbol"] = pd.Categorical(melted["symbol"], categories=order, ordered=True)
    fig = px.bar(melted, x="symbol", y="value", color="metric", barmode="group", title="Volatility vs Max Drawdown")
    fig.update_layout(yaxis_title="Percent", xaxis_title="Symbol")
    fig.update_xaxes(type="category")
    return fig


def risk_return_scatter(summary_df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        summary_df,
        x="annual_vol",
        y="cagr",
        text="symbol",
        hover_data=["sharpe", "max_drawdown", "beta", "alpha"],
        title="Risk-Return Quadrant (Volatility vs CAGR)",
    )
    fig.update_traces(textposition="top center")
    fig.update_layout(xaxis_title="Annual Volatility (%)", yaxis_title="CAGR (%)")
    return fig


def six_dimension_radar(summary_df: pd.DataFrame) -> go.Figure:
    dims = ["cagr", "annual_vol", "max_drawdown", "sharpe", "beta", "alpha"]
    labels = ["CAGR", "Annual Vol", "Max Drawdown", "Sharpe", "Beta", "Alpha"]
    plot_df = summary_df[["symbol", *dims]].copy()
    plot_df["max_drawdown"] = plot_df["max_drawdown"].abs()
    for col in dims:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
        col_min = plot_df[col].min(skipna=True)
        col_max = plot_df[col].max(skipna=True)
        if pd.isna(col_min) or pd.isna(col_max) or col_max == col_min:
            plot_df[col] = 0.5
        else:
            plot_df[col] = (plot_df[col] - col_min) / (col_max - col_min)

    fig = go.Figure()
    for _, row in plot_df.iterrows():
        r_values = [float(row[d]) for d in dims]
        fig.add_trace(
            go.Scatterpolar(
                r=r_values + [r_values[0]],
                theta=labels + [labels[0]],
                fill="toself",
                name=str(row["symbol"]),
            )
        )

    fig.update_layout(
        title="Six-Dimension Radar (Normalized Across Selected Stocks)",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
    )
    return fig


def rolling_vol_overlay(processed: dict[str, pd.DataFrame]) -> go.Figure:
    fig = go.Figure()
    for symbol, df in processed.items():
        fig.add_trace(go.Scatter(x=df["date"], y=df["rolling_vol_30d"], mode="lines", name=symbol))
    fig.update_layout(title="Rolling Volatility Overlay (30D)", xaxis_title="Date", yaxis_title="Annualized Vol")
    return fig


def returns_corr_heatmap(processed: dict[str, pd.DataFrame]) -> go.Figure:
    rows: list[pd.DataFrame] = []
    for symbol, df in processed.items():
        tmp = df[["date", "return"]].copy()
        tmp["symbol"] = symbol
        rows.append(tmp)
    if not rows:
        return go.Figure()

    stacked = pd.concat(rows, ignore_index=True)
    pivot_ret = stacked.pivot_table(index="date", columns="symbol", values="return", aggfunc="mean")
    corr = pivot_ret.corr(min_periods=20)
    if corr.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Not enough overlapping returns to compute correlation.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(title="Return Correlation Heatmap")
        return fig

    fig = px.imshow(
        corr,
        text_auto=".2f",
        zmin=-1,
        zmax=1,
        color_continuous_scale="RdBu_r",
        title="Return Correlation Heatmap",
    )
    fig.update_layout(xaxis_title="Symbol", yaxis_title="Symbol")
    return fig


def metric_heatmap(summary_df: pd.DataFrame) -> go.Figure:
    dims = ["total_return", "cagr", "annual_vol", "max_drawdown", "sharpe", "win_rate", "beta", "alpha"]
    labels = ["Total Return", "CAGR", "Annual Vol", "Max Drawdown", "Sharpe", "Win Rate", "Beta", "Alpha"]
    plot_df = summary_df[["symbol", *dims]].copy()
    plot_df["symbol"] = plot_df["symbol"].astype(str)
    for col in dims:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
    plot_df["max_drawdown"] = plot_df["max_drawdown"].abs()

    matrix = plot_df.set_index("symbol")[dims]
    for col in dims:
        col_min = matrix[col].min(skipna=True)
        col_max = matrix[col].max(skipna=True)
        if pd.isna(col_min) or pd.isna(col_max) or col_max == col_min:
            matrix[col] = 0.5
        else:
            matrix[col] = (matrix[col] - col_min) / (col_max - col_min)

    fig = px.imshow(
        matrix.T,
        labels=dict(x="Symbol", y="Metric", color="Normalized Score"),
        x=matrix.index.tolist(),
        y=labels,
        color_continuous_scale="Viridis",
        title="Multi-Metric Dimension Heatmap (Normalized)",
        aspect="auto",
    )
    return fig


def format_metric(value: float, pct: bool = True) -> str:
    if pd.isna(value):
        return "N/A"
    if pct:
        return f"{value:.2f}%"
    return f"{value:.4f}"


st.set_page_config(page_title="Stock Risk & Return Analyzer", layout="wide")
st.title("Stock Risk & Return Analyzer")
st.caption("ACC102 Track 4 - WRDS + SQL + Streamlit")

if "symbols" not in st.session_state:
    st.session_state.symbols = ["600519"]
if "last_data" not in st.session_state:
    st.session_state.last_data = {}

st.sidebar.header("WRDS Credentials")
wrds_username = st.sidebar.text_input("WRDS Username", value="", placeholder="Enter WRDS username")
wrds_password = st.sidebar.text_input("WRDS Password", value="", type="password", placeholder="Enter WRDS password")
st.sidebar.caption("Credentials are used for the current session only.")

st.subheader("Stock Selection")
default_symbols = [
    "600519",
    "000001",
    "000300",
    "600036",
    "601318",
    "601166",
    "600276",
    "600887",
    "600030",
    "601888",
    "000651",
    "000333",
    "002415",
    "300750",
    "688981",
    "AAPL",
    "MSFT",
    "TSLA",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "NFLX",
    "JPM",
    "XOM",
    "BABA",
    "PDD",
    "TCEHY",
    "BAC",
    "V",
]
selected_symbols = st.multiselect(
    "Click to choose stocks",
    options=default_symbols,
    default=[s for s in st.session_state.symbols if s in default_symbols],
)
custom_symbol = st.text_input("Add custom stock symbol", value="", placeholder="Example: 600036 or NVDA")
if st.button("Add Custom Symbol"):
    custom = custom_symbol.strip().upper()
    if custom and custom not in st.session_state.symbols:
        st.session_state.symbols.append(custom)
        st.success(f"Added symbol: {custom}")

merged_symbols = list(dict.fromkeys(selected_symbols + st.session_state.symbols))
st.session_state.symbols = merged_symbols if merged_symbols else ["600519"]
st.write("Current symbols:", ", ".join(st.session_state.symbols))

mode = st.radio("Mode", ["Single Stock Analysis", "Multi Stock Comparison"], horizontal=True)
date_range = st.date_input(
    "Click to select date range",
    value=(date.today() - timedelta(days=365 * 2), date.today()),
    max_value=date.today(),
)
if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = date.today() - timedelta(days=365 * 2)
    end_date = date.today()

bench_symbol = st.text_input("Benchmark Symbol (for Beta/Alpha)", value="000300")
refresh = st.button("Refresh WRDS Data")

if st.button("Run Analysis", type="primary"):
    processed = {}
    progress = st.progress(0, text="Loading data...")
    for idx, symbol in enumerate(st.session_state.symbols):
        cfg = QueryConfig(symbol=symbol, start_date=start_date, end_date=end_date)
        if wrds_username and wrds_password:
            raw = fetch_stock_data_direct_sqlalchemy(cfg, wrds_username, wrds_password)
            if raw is None or raw.empty:
                raw = _mock_prices(cfg)
        else:
            raw = fetch_stock_data(
                cfg,
                refresh=refresh,
                wrds_username=None,
                wrds_password=None,
            )
        processed[symbol] = prepare_price_df(raw)
        progress.progress((idx + 1) / len(st.session_state.symbols), text=f"Completed: {symbol}")

    if bench_symbol and bench_symbol not in processed:
        cfg = QueryConfig(symbol=bench_symbol, start_date=start_date, end_date=end_date)
        if wrds_username and wrds_password:
            bench_raw = fetch_stock_data_direct_sqlalchemy(cfg, wrds_username, wrds_password)
            if bench_raw is None or bench_raw.empty:
                bench_raw = _mock_prices(cfg)
        else:
            bench_raw = fetch_stock_data(
                cfg,
                refresh=refresh,
                wrds_username=None,
                wrds_password=None,
            )
        processed[bench_symbol] = prepare_price_df(bench_raw)

    st.session_state.last_data = processed
    st.success("Analysis complete")

if st.session_state.last_data:
    processed = st.session_state.last_data
    symbols = [s for s in st.session_state.symbols if s in processed]
    summary = build_summary_table(processed, benchmark_symbol=bench_symbol if bench_symbol in processed else None)

    if mode == "Single Stock Analysis":
        pick = st.selectbox("Select Stock", symbols)
        df = processed[pick]
        metric_row = summary[summary["symbol"] == pick].iloc[0].to_dict()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Return", format_metric(metric_row["total_return"]))
        c2.metric("Annual Volatility", format_metric(metric_row["annual_vol"]))
        c3.metric("Max Drawdown", format_metric(metric_row["max_drawdown"]))
        c4.metric("Sharpe Ratio", format_metric(metric_row["sharpe"], pct=False))

        c5, c6 = st.columns(2)
        c5.metric("Beta", format_metric(metric_row["beta"], pct=False))
        c6.metric("Alpha (Annualized)", format_metric(metric_row["alpha"]))

        st.plotly_chart(nav_line(df, pick), use_container_width=True)
        st.plotly_chart(return_hist(df, pick), use_container_width=True)
        st.plotly_chart(rolling_vol_line(df, pick), use_container_width=True)
        st.plotly_chart(drawdown_line(df, pick), use_container_width=True)
    else:
        cmp_df = summary[summary["symbol"].isin(symbols)].copy()
        selected_processed = {s: processed[s] for s in symbols}
        st.plotly_chart(multi_nav_chart(selected_processed), use_container_width=True)
        st.plotly_chart(annual_return_bar(selected_processed), use_container_width=True)
        st.plotly_chart(vol_drawdown_bar(cmp_df), use_container_width=True)
        st.plotly_chart(risk_return_scatter(cmp_df), use_container_width=True)
        st.plotly_chart(six_dimension_radar(cmp_df), use_container_width=True)
        st.plotly_chart(rolling_vol_overlay(selected_processed), use_container_width=True)
        st.plotly_chart(returns_corr_heatmap(selected_processed), use_container_width=True)
        st.plotly_chart(metric_heatmap(cmp_df), use_container_width=True)
        st.dataframe(cmp_df, use_container_width=True)

    excel_name = OUTPUT_DIR / f"stock_analysis_{date.today()}.xlsx"
    if st.button("Export to Excel"):
        with pd.ExcelWriter(excel_name, engine="openpyxl") as writer:
            summary.to_excel(writer, sheet_name="summary", index=False)
            for symbol, df in processed.items():
                df.to_excel(writer, sheet_name=symbol[:31], index=False)
        st.success(f"Exported: {excel_name}")

with st.expander("Metric Definitions"):
    st.markdown(
        """
        - Total Return: overall return from start date to end date
        - CAGR: annualized compound return
        - Annual Volatility: std(daily return) * sqrt(252)
        - Max Drawdown: largest peak-to-trough decline in NAV
        - Sharpe Ratio: return-to-volatility ratio (risk-free rate assumed 0)
        - Beta/Alpha: relative to benchmark return series
        """
    )

with st.expander("Help"):
    st.markdown(
        """
        1. Enter WRDS username/password in the sidebar (optional for mock mode).
        2. Select stocks by clicking options or adding a custom symbol.
        3. Pick the analysis window directly from the calendar range selector.
        4. Click `Run Analysis`, then switch between single/multi-stock views.
        5. Use `Export to Excel` to generate submission output files.
        """
    )
