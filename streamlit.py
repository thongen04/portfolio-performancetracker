import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io    
import zipfile

st.set_page_config(page_title="Portfolio Performance Tracker", layout="centered")

def set_background_url(url: str):
    st.markdown(f"""
        <style>
        /* Dark overlay on background */
        [data-testid="stAppViewContainer"] {{
             background: linear-gradient(rgba(0,0,0,0.55), rgba(0,0,0,0.55)),
                        url("{url}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        /* Main content container */
        .block-container {{
            background: rgba(255,255,255,0.92);
            border-radius: 12px;
            padding: 1rem 1.2rem;
        }}
        /* === Fonts === */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600&family=Roboto:wght@400&display=swap');
        h1, h2, h3, h4, h5, h6, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
            font-family: 'Poppins', sans-serif !important;
            font-weight: 600;
        }}
        html, body, [class*="css"], p, div, span, label {{
            font-family: 'Roboto', sans-serif !important;
            font-weight: 400;
        }}
        div[data-testid="metric-container"] > label {{
            font-family: 'Poppins', sans-serif !important;
            font-weight: 600;
            color: #2c3e50;
        }}
        </style>
    """, unsafe_allow_html=True)

set_background_url("https://media.istockphoto.com/id/1300120223/photo/digital-data-financial-investment-trends-financial-business-diagram-with-charts-and-stock.jpg?s=612x612&w=0&k=20&c=gQCwdSMiohFpR6XU-8v20sImgOhZEh4ajlr6RmVCXDA=")

st.markdown(
    "<h1 style='text-align: center;'>📊 Portfolio Tracker Dashboard</h1>",
    unsafe_allow_html=True
)

def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()

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
        tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]
        shares  = [int(s.strip()) for s in shares_input.split(",") if s.strip()]

        if len(tickers) == 0:
            st.error("Please enter at least one ticker.")
            st.stop()
        if len(tickers) != len(shares):
            st.error("The number of tickers and shares must match.")
            st.stop()

        data = yf.download(tickers, start=start_date, end=end_date)["Close"]
        if isinstance(data, pd.Series):
            data = data.to_frame(name=tickers[0])
        data = data.dropna(how="all")

        st.subheader("Prices")
        norm_prices = data / data.iloc[0]
        st.line_chart(norm_prices, use_container_width=True)

        weights  = pd.Series(shares, index=norm_prices.columns)
        alloc    = norm_prices * weights
        port_val = alloc.sum(axis=1)

        daily_ret  = port_val.pct_change().dropna()
        ann_return = daily_ret.mean() * 252
        ann_vol    = daily_ret.std() * np.sqrt(252)
        sharpe     = (ann_return - (risk_free/100.0)) / ann_vol if ann_vol > 0 else np.nan

        st.subheader("📊 Portfolio Metrics")
        c1, c2, c3 = st.columns(3)
        c1.metric("Annual Return",     f"{ann_return*100:.2f}%")
        c2.metric("Annual Volatility", f"{ann_vol*100:.2f}%")
        c3.metric("Sharpe Ratio",      f"{sharpe:.2f}")

        bench_symbol = None
        combined = None
        bench_norm = None
        if "None" not in benchmark_choice:
            bench_symbol = benchmark_choice.split()[0]
            bench = yf.download(bench_symbol, start=start_date, end=end_date)["Close"]
            if isinstance(bench, pd.DataFrame):
                bench = bench.squeeze("columns")
            if bench.empty or bench.dropna().empty:
                st.warning(f"No data for benchmark {bench_symbol}.")
                st.stop()

            bench_norm = bench / bench.iloc[0]

            idx = port_val.index.intersection(bench_norm.index)
            series_port = (port_val / port_val.loc[idx[0]]).loc[idx]
            series_bench = bench_norm.loc[idx]

            st.subheader("Portfolio vs Benchmark (Normalised)")
            combined = pd.concat(
                [series_port.rename("Portfolio"), series_bench.rename(bench_symbol)],
                axis=1
            ).dropna()
            st.line_chart(combined, use_container_width=True)

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
            st.subheader("Portfolio Value (Normalised)")
            st.line_chart((port_val / port_val.iloc[0]).to_frame("Portfolio"),
                          use_container_width=True)

        st.subheader("⬇️ Export: metrics CSV + chart images")

        fig_prices, ax = plt.subplots(figsize=(8, 4))
        norm_prices.plot(ax=ax)
        ax.set_title("Normalised Prices (base = 1.0)")
        ax.set_xlabel("Date"); ax.set_ylabel("Index")
        ax.legend(loc="best", fontsize=8)
        prices_png = fig_to_png_bytes(fig_prices)
        plt.close(fig_prices)

        if combined is not None:
            fig_comp, axc = plt.subplots(figsize=(8, 4))
            combined.plot(ax=axc)
            axc.set_title("Portfolio vs Benchmark (Normalised)")
            axc.set_xlabel("Date"); axc.set_ylabel("Index")
            axc.legend(loc="best", fontsize=8)
            comp_png = fig_to_png_bytes(fig_comp)
            comp_name = "comparison_chart.png"
            plt.close(fig_comp)
        else:
            fig_port, axp = plt.subplots(figsize=(8, 4))
            (port_val / port_val.iloc[0]).rename("Portfolio").plot(ax=axp)
            axp.set_title("Portfolio (Normalised)")
            axp.set_xlabel("Date"); axp.set_ylabel("Index")
            axp.legend(loc="best", fontsize=8)
            comp_png = fig_to_png_bytes(fig_port)
            comp_name = "portfolio_chart.png"
            plt.close(fig_port)

        results = {
            "Portfolio Annual Return": ann_return,
            "Portfolio Annual Volatility": ann_vol,
            "Portfolio Sharpe Ratio": sharpe,
        }
        if bench_symbol is not None:
            results.update({
                f"{bench_symbol} Annual Return": bench_ann_return,
                f"{bench_symbol} Annual Volatility": bench_ann_vol,
                f"{bench_symbol} Sharpe Ratio": bench_sharpe,
            })
        results_df = pd.DataFrame(list(results.items()), columns=["Metric", "Value"])

        fig_tbl, axt = plt.subplots(figsize=(6, 0.6 + 0.35*len(results_df)))
        axt.axis("off")
        table = axt.table(
            cellText=[[m, f"{v*100:.2f}%"] if ("Return" in m or "Volatility" in m) else [m, ("N/A" if pd.isna(v) else f"{v:.2f}")]
                      for m, v in results_df.values],
            colLabels=["Metric", "Value"],
            loc="center",
            cellLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.2)
        axt.set_title("Portfolio Metrics", pad=10)
        metrics_png = fig_to_png_bytes(fig_tbl)
        plt.close(fig_tbl)

        st.download_button(
            "Download metrics.csv",
            results_df.to_csv(index=False).encode("utf-8"),
            "metrics.csv",
            "text/csv"
        )

        st.download_button("Download prices_chart.png", prices_png, "prices_chart.png", "image/png")
        st.download_button(f"Download {comp_name}", comp_png, comp_name, "image/png")
        st.download_button("Download metrics_table.png", metrics_png, "metrics_table.png", "image/png")

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("metrics.csv", results_df.to_csv(index=False))
            zf.writestr("prices_chart.png", prices_png)
            zf.writestr(comp_name, comp_png)
            zf.writestr("metrics_table.png", metrics_png)
        zip_buf.seek(0)
        st.download_button("Download ALL (ZIP)", zip_buf.getvalue(), "portfolio_export.zip", "application/zip")

    except Exception as e:
        st.error(f"Something went wrong: {e}")
