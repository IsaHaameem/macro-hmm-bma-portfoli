"""
data/preprocess.py
Transforms raw/sample prices into log-returns and normalizes macro data.
Splits data into Train (2010-2019), Val (2020-2022), and Test (2023-2025).
"""
import pandas as pd
import numpy as np
import os
import sys

# Add the parent directory (root) to the Python path so it can find config.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
def run_preprocessing():
    print("Loading raw data...")
    
    # 3-Tier Fallback Strategy
    if config.USE_SAMPLE_DATA or not os.path.exists("data/raw/prices.csv"):
        print("Using sample synthetic data...")
        prices = pd.read_csv("data/sample/prices.csv", index_col=0, parse_dates=True)
        macro = pd.read_csv("data/sample/macro.csv", index_col=0, parse_dates=True)
    else:
        print("Using live fetched data...")
        prices = pd.read_csv("data/raw/prices.csv", index_col=0, parse_dates=True)
        macro = pd.read_csv("data/raw/macro.csv", index_col=0, parse_dates=True)

    print("Calculating log returns...")
    returns = np.log(prices / prices.shift(1)).dropna()

    print(f"Applying rolling Z-score (window={config.ROLLING_NORM_WINDOW}) to macro...")
    macro_mean = macro.rolling(window=config.ROLLING_NORM_WINDOW).mean()
    macro_std = macro.rolling(window=config.ROLLING_NORM_WINDOW).std()
    macro_z = ((macro - macro_mean) / macro_std).dropna()

    # Align dates
    common_dates = returns.index.intersection(macro_z.index)
    returns = returns.loc[common_dates]
    macro_z = macro_z.loc[common_dates]

    print("Segmenting Train / Val / Test splits...")
    train_idx = (common_dates >= "2010-01-01") & (common_dates <= "2019-12-31")
    val_idx = (common_dates >= "2020-01-01") & (common_dates <= "2022-12-31")
    test_idx = (common_dates >= "2023-01-01") & (common_dates <= "2025-12-31")

    data_splits = {
        "returns_train": returns.loc[train_idx],
        "macro_train": macro_z.loc[train_idx],
        "returns_val": returns.loc[val_idx],
        "macro_val": macro_z.loc[val_idx],
        "returns_test": returns.loc[test_idx],
        "macro_test": macro_z.loc[test_idx],
        "full_returns": returns,
        "full_macro": macro_z
    }

    # Save to processed folder for dashboard usage
    os.makedirs("data/processed", exist_ok=True)
    returns.to_csv("data/processed/returns.csv")
    macro_z.to_csv("data/processed/macro_z.csv")
    print("Preprocessing complete!")
    
    return data_splits

if __name__ == "__main__":
    _ = run_preprocessing()