import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, window=14):
    delta = series.diff()
    gain  = (delta.clip(lower=0)).ewm(alpha=1/window, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/window, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))
start_date, end_date = '2020-01-01', '2024-12-31'
rf_annual = 0.01                     
bench_symbol = 'SPY'                   
tickers = ['AAPL', 'MSFT', 'GOOGL']
shares  = [10, 6, 2]
frames = []
for t in tickers:
    try:
        h = yf.Ticker(t).history(start=start_date, end=end_date, auto_adjust=True)
        if not h.empty and 'Close' in h.columns:
            frames.append(h[['Close']].rename(columns={'Close': t}))
        else:
            print(f"⚠ No close prices for {t}")
    except Exception as e:
        print(f"⚠ Failed {t}: {e}")

if not frames:
    raise RuntimeError("No data downloaded. Try different tickers or a shorter date range.")

prices = pd.concat(frames, axis=1).dropna(how='all')
print("☑ Downloaded columns:", list(prices.columns))
norm = prices / prices.iloc[0]                     
share_series = pd.Series(shares, index=norm.columns)
money_paths = norm * share_series
port_val = money_paths.sum(axis=1)    
daily_ret  = port_val.pct_change().dropna()
annual_ret = (1 + daily_ret.mean())**252 - 1
annual_vol = daily_ret.std() * sqrt(252)
sharpe     = (annual_ret - rf_annual) / annual_vol if annual_vol != 0 else float('nan')

print("\n----- Portfolio Metrics -----")
print(f"Annual Return: {annual_ret:.2%}")
print(f"Annual Volatility: {annual_vol:.2%}")
print(f"Sharpe Ratio: {sharpe:.2f}")
plt.figure(figsize=(10,5))
(port_val / port_val.iloc[0]).plot(label="Portfolio")
plt.title("Portfolio Value Over Time (Normalised)")
plt.xlabel("Date"); plt.ylabel("Normalised Value"); plt.legend()
plt.tight_layout()
plt.show()
bh = yf.Ticker(bench_symbol).history(start=start_date, end=end_date, auto_adjust=True)
if not bh.empty and 'Close' in bh.columns:
    bench_norm = bh['Close'] / bh['Close'].iloc[0]
    port_norm  = port_val / port_val.iloc[0]
    aligned = pd.concat([port_norm, bench_norm], axis=1).dropna()
    aligned.columns = ['Portfolio', bench_symbol]

    plt.figure(figsize=(10,5))
    aligned.plot()
    plt.title(f"Portfolio vs {bench_symbol} (Normalised)")
    plt.xlabel("Date"); plt.ylabel("Normalised Value"); plt.legend()
    plt.tight_layout()
    plt.show()

    port_cum  = aligned['Portfolio'].iloc[-1] / aligned['Portfolio'].iloc[0] - 1
    bench_cum = aligned[bench_symbol].iloc[-1] / aligned[bench_symbol].iloc[0] - 1
    print(f"\nCumulative Return — Portfolio: {port_cum:.2%}, {bench_symbol}: {bench_cum:.2%}")
    print(f"Out/Under-performance vs {bench_symbol}: {(port_cum - bench_cum):+.2%}")
else:
    print(f"⚠ Benchmark {bench_symbol} had no data.")
detail = pd.DataFrame({"date": port_val.index, "portfolio_value": port_val.values}) 

metric = pd.DataFrame({
    "annual_return":    [annual_ret],
    "annual_volatility":[annual_vol],
    "sharpe_ratio":     [sharpe],
})

print(detail)
print(metric)
indicator_symbol = tickers[0]
hist = yf.Ticker(indicator_symbol).history(start=start_date, end=end_date, auto_adjust=True)

if not hist.empty and {'Close','Volume'}.issubset(hist.columns):
    close = hist['Close']; vol = hist['Volume']

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper = sma20 + 2*std20
    lower = sma20 - 2*std20

    macd_line = ema(close, 12) - ema(close, 26)
    signal    = ema(macd_line, 9)
    macd_hist = macd_line - signal

    rsi14 = rsi(close, 14)

    fig = plt.figure(figsize=(12,9))
    gs = fig.add_gridspec(4, 1, hspace=0.05)
    ax_price = fig.add_subplot(gs[0, 0])
    ax_vol   = fig.add_subplot(gs[1, 0], sharex=ax_price)
    ax_macd  = fig.add_subplot(gs[2, 0], sharex=ax_price)
    ax_rsi   = fig.add_subplot(gs[3, 0], sharex=ax_price)

    ax_price.plot(close, label='Close')
    ax_price.plot(sma20, label='SMA20')
    ax_price.plot(upper, label='BB Upper', linestyle='--')
    ax_price.plot(lower, label='BB Lower', linestyle='--')
    ax_price.set_title(f"{indicator_symbol} — Price with Bollinger Bands")
    ax_price.legend(loc='upper left')

    ax_vol.bar(vol.index, vol.values)
    ax_vol.set_ylabel("Volume")

    ax_macd.plot(macd_line, label='MACD')
    ax_macd.plot(signal, label='Signal')
    ax_macd.bar(macd_hist.index, macd_hist.values)
    ax_macd.legend(loc='upper left')
    ax_macd.set_ylabel("MACD")

    ax_rsi.plot(rsi14, label='RSI (14)')
    ax_rsi.axhline(70, linestyle='--'); ax_rsi.axhline(30, linestyle='--')
    ax_rsi.legend(loc='upper left'); ax_rsi.set_ylabel("RSI")

    plt.setp(ax_price.get_xticklabels(), visible=False)
    plt.setp(ax_vol.get_xticklabels(),   visible=False)
    plt.setp(ax_macd.get_xticklabels(),  visible=False)
    plt.tight_layout()
    plt.show()
else:
    print(f"⚠ Not enough data to plot indicators for {indicator_symbol}.")

