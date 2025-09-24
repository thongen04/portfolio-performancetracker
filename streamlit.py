import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Portfolio Performance Tracker", layout="centered")


def set_background_url(url: str):
    st.markdown(f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image: url("{url}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        [data-testid="stHeader"] {{
            background: rgba(0,0,0,0);
        }}
        .block-container {{
            background: rgba(255,255,255,0.92);
            border-radius: 12px;
            padding: 1rem 1.2rem;
        }}
        </style>
    """, unsafe_allow_html=True)

set_background_url("https://media.istockphoto.com/id/1300120223/photo/digital-data-financial-investment-trends-financial-business-diagram-with-charts-and-stock.jpg?s=612x612&w=0&k=20&c=gQCwdSMiohFpR6XU-8v20sImgOhZEh4ajlr6RmVCXDA=")

st.markdown(
    "<h1 style='text-align: center; color: #2c3e50;'>📊 Portfolio Tracker Dashboard</h1>",
    unsafe_allow_html=True
)

# Inputs
tickers_input = st.text_input("Enter stock tickers (comma separated):", "AAPL, MSFT, GOOGL")
shares_input = st.text_input("Enter shares (comma separated):", "10, 5, 8")
colA, colB, colC = st.columns(3)
with colA:
    start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
with colB:
    end_date = st.date_input("End Date", pd.to_datetime("2024-12-31"))
with colC:
    risk_free = st.number_input("Risk-free rate (%)", 0.0, 10.0, 1.0)

benchmark_choice = st.selectbox(
    "Choose a benchmark:",
    ["^GSPC (S&P 500)", "^KLSE (FBMKLCI)", "SPY (ETF)", "None"]
)

if st.button("Run Analysis", type="primary"):
    try:
        # Parse inputs
        tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]
        shares  = [int(s.strip()) for s in shares_input.split(",") if s.strip()]

        if len(tickers) == 0:
            st.error("Please enter at least one ticker.")
            st.stop()
        if len(tickers) != len(shares):
            st.error("The number of tickers and shares must match.")
            st.stop()

        # Download adjusted close prices
        data = yf.download(tickers, start=start_date, end=end_date)["Close"]
        if isinstance(data, pd.Series):   # single ticker -> DataFrame
            data = data.to_frame(name=tickers[0])
        data = data.dropna(how="all")     # remove empty rows if any

        # ---- Prices chart (responsive) ----
        st.subheader("Prices")
        norm_prices = data / data.iloc[0]
        st.line_chart(norm_prices, use_container_width=True)

        # ---- Portfolio series ----
        weights  = pd.Series(shares, index=norm_prices.columns)
        alloc    = norm_prices * weights
        port_val = alloc.sum(axis=1)

        # ---- Portfolio metrics ----
        daily_ret  = port_val.pct_change().dropna()
        ann_return = daily_ret.mean() * 252
        ann_vol    = daily_ret.std() * np.sqrt(252)
        sharpe     = (ann_return - (risk_free/100.0)) / ann_vol if ann_vol > 0 else np.nan

        st.subheader("📊 Portfolio Metrics")
        c1, c2, c3 = st.columns(3)
        c1.metric("Annual Return",     f"{ann_return*100:.2f}%")
        c2.metric("Annual Volatility", f"{ann_vol*100:.2f}%")
        c3.metric("Sharpe Ratio",      f"{sharpe:.2f}")

        # ---- Benchmark (optional, fully responsive) ----
        if "None" not in benchmark_choice:
            bench_symbol = benchmark_choice.split()[0]
            bench = yf.download(bench_symbol, start=start_date, end=end_date)["Close"]
            # Handle DataFrame/Series cases safely
            if isinstance(bench, pd.DataFrame):
                bench = bench.squeeze("columns")
            if bench.empty or bench.dropna().empty:
                st.warning(f"No data for benchmark {bench_symbol}.")
                st.stop()

            bench_norm = bench / bench.iloc[0]

            # Align dates (intersection) for a clean comparison
            idx = port_val.index.intersection(bench_norm.index)
            series_port = (port_val / port_val.loc[idx[0]]).loc[idx]
            series_bench = bench_norm.loc[idx]

            st.subheader("Portfolio vs Benchmark (Normalised)")
            combined = pd.concat(
                [series_port.rename("Portfolio"), series_bench.rename(bench_symbol)],
                axis=1
            ).dropna()
            st.line_chart(combined, use_container_width=True)

            # Benchmark metrics (based on its own series)
            bench_ret        = series_bench.pct_change().dropna()
            bench_ann_return = bench_ret.mean() * 252
            bench_ann_vol    = bench_ret.std() * np.sqrt(252)
            bench_sharpe     = (bench_ann_return - (risk_free/100.0)) / bench_ann_vol if bench_ann_vol > 0 else np.nan

            st.subheader(f"📊 Benchmark ({benchmark_choice}) Metrics")
            b1, b2, b3 = st.columns(3)
            b1.metric("Annual Return",     f"{bench_ann_return*100:.2f}%")
            b2.metric("Annual Volatility", f"{bench_ann_vol*100:.2f}%")
            b3.metric("Sharpe Ratio",      f"{bench_sharpe:.2f}")
        else:
            # Portfolio-only view (responsive)
            st.subheader("Portfolio Value (Normalised)")
            st.line_chart((port_val / port_val.iloc[0]).to_frame("Portfolio"),
                          use_container_width=True)

    except Exception as e:
        st.error(f"Something went wrong: {e}")