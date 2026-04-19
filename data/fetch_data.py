"""
data/fetch_data.py
Downloads live market data from Yahoo Finance and the St. Louis Fed (FRED).
Aligns quarterly/monthly macro data to daily trading days via forward-filling.
"""
import os
import requests
import pandas as pd
import yfinance as yf

# Project specific assets
TICKERS = ["SPY", "QQQ", "TLT", "IEF", "GLD", "XLF", "XLV", "XLK"]
MACRO_SERIES = ["GDPC1", "CPIAUCSL", "FEDFUNDS", "UNRATE"]
START_DATE = "2010-01-01"
END_DATE = "2025-12-31"

def fetch_fred_data(api_key):
    print("Fetching Macro data from FRED...")
    macro_df = pd.DataFrame()
    
    for series in MACRO_SERIES:
        url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series}&api_key={api_key}&file_type=json&observation_start={START_DATE}&observation_end={END_DATE}"
        response = requests.get(url)
        
        if response.status_code != 200:
            print(f"Failed to fetch {series} from FRED. Check your API key.")
            continue
            
        data = response.json()["observations"]
        
        # Extract dates and values, ignoring missing "." values
        dates = [obs["date"] for obs in data if obs["value"] != "."]
        values = [float(obs["value"]) for obs in data if obs["value"] != "."]
        
        series_df = pd.DataFrame({"Date": dates, series: values})
        series_df["Date"] = pd.to_datetime(series_df["Date"])
        series_df.set_index("Date", inplace=True)
        
        if macro_df.empty:
            macro_df = series_df
        else:
            macro_df = macro_df.join(series_df, how="outer")
            
    return macro_df

def fetch_live_data():
    # 1. Fetch ETF Prices (Sequential Loop)
    print(f"Fetching daily close prices for {len(TICKERS)} ETFs from Yahoo Finance...")
    prices = pd.DataFrame()
    
    for ticker in TICKERS:
        print(f" -> Downloading {ticker}...")
        df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
        if not df.empty:
            # Handle both old and new yfinance column structures
            close_data = df["Close"]
            if isinstance(close_data, pd.DataFrame):
                prices[ticker] = close_data.iloc[:, 0]
            else:
                prices[ticker] = close_data
        else:
            print(f"    WARNING: {ticker} failed to download!")
            
    # Strip timezones from yfinance dates so they match FRED dates
    if not prices.empty:
        prices.index = pd.to_datetime(prices.index).tz_localize(None)
    
    # 2. Fetch Macro Data
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        print("\nERROR: FRED_API_KEY environment variable not found!")
        print("Please set it in your terminal before running. Falling back to synthetic data.")
        return False
        
    macro = fetch_fred_data(api_key)
    
    if macro.empty:
        return False

    print("Aligning and forward-filling macro data to match trading days...")
    # Join macro data onto the trading days index
    aligned_data = pd.DataFrame(index=prices.index).join(macro, how="left")
    
    # Forward fill (e.g., Q1 GDP applies to all days until Q2 is reported)
    aligned_data = aligned_data.ffill().bfill()

    # Save to raw directory
    os.makedirs("data/raw", exist_ok=True)
    prices.to_csv("data/raw/prices.csv")
    aligned_data.to_csv("data/raw/macro.csv")
    
    print("Success! Live data saved to data/raw/")
    return True

# THIS IS THE CRITICAL BLOCK THAT WAS MISSING
if __name__ == "__main__":
    fetch_live_data()