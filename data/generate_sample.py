"""
data/generate_sample.py
Generates synthetic ETF prices and macro data with injected market regimes.
"""
import pandas as pd
import numpy as np
import os

def generate_data():
    print("Generating synthetic market data...")
    os.makedirs("data/sample", exist_ok=True)

    # Date range: 2010 to end of 2025
    dates = pd.date_range(start="2010-01-01", end="2025-12-31", freq="B")
    n_days = len(dates)

    # 1. Macro Indicators (Random Walk with Trends)
    macro = pd.DataFrame(index=dates)
    macro["GDPC1"] = np.linspace(15000, 28000, n_days) + np.random.normal(0, 50, n_days)
    macro["CPIAUCSL"] = np.linspace(210, 310, n_days) + np.random.normal(0, 1, n_days)
    macro["FEDFUNDS"] = np.sin(np.linspace(0, 10, n_days)) * 2 + 2.5 + np.random.normal(0, 0.1, n_days)
    macro["UNRATE"] = np.cos(np.linspace(0, 8, n_days)) * 2 + 5 + np.random.normal(0, 0.2, n_days)

    # 2. ETF Prices (GBM with Regimes)
    tickers = ["SPY", "QQQ", "TLT", "IEF", "GLD", "XLF", "XLV", "XLK"]
    prices = pd.DataFrame(index=dates, columns=tickers)
    prices.iloc[0] = 100.0  # Start all ETFs at $100

    # Base daily drifts and volatilities
    drifts = {"SPY": 0.0003, "QQQ": 0.0004, "TLT": 0.0001, "IEF": 0.00005, "GLD": 0.0002, "XLF": 0.00025, "XLV": 0.00025, "XLK": 0.0004}
    vols = {"SPY": 0.01, "QQQ": 0.015, "TLT": 0.008, "IEF": 0.004, "GLD": 0.009, "XLF": 0.012, "XLV": 0.01, "XLK": 0.015}

    for i in range(1, n_days):
        current_date = dates[i]
        drift_mod, vol_mod = 1.0, 1.0

        # Inject Regimes
        if pd.to_datetime("2020-02-15") <= current_date <= pd.to_datetime("2020-04-15"):
            drift_mod, vol_mod = -15.0, 5.0  # COVID Crash
        elif pd.to_datetime("2022-01-01") <= current_date <= pd.to_datetime("2022-10-31"):
            drift_mod, vol_mod = -3.0, 2.0   # 2022 Bear Market
        elif pd.to_datetime("2023-11-01") <= current_date <= pd.to_datetime("2024-12-31"):
            drift_mod, vol_mod = 3.0, 1.5    # AI Bull Run (Benefits all, especially Tech)

        for ticker in tickers:
            # Bonds and Gold act as slight safe havens during crashes
            if drift_mod < 0 and ticker in ["TLT", "IEF", "GLD"]:
                actual_drift = drifts[ticker] * 2.0
            # Tech and SPY get the AI boost
            elif drift_mod > 1 and ticker not in ["QQQ", "XLK", "SPY"]:
                actual_drift = drifts[ticker]
            else:
                actual_drift = drifts[ticker] * drift_mod

            actual_vol = vols[ticker] * vol_mod
            shock = np.random.normal(actual_drift, actual_vol)
            prices.iloc[i, prices.columns.get_loc(ticker)] = prices.iloc[i-1, prices.columns.get_loc(ticker)] * np.exp(shock)

    # Save to disk
    prices.to_csv("data/sample/prices.csv")
    macro.to_csv("data/sample/macro.csv")
    print(f"Success! {n_days} days of regime-injected data saved to data/sample/")

if __name__ == "__main__":
    generate_data()