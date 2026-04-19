"""
backtest/metrics.py
Financial performance metrics hardened against NaN/Inf values.
"""
import numpy as np
import pandas as pd

def calculate_cagr(cumulative_returns, years):
    """Compound Annual Growth Rate"""
    if cumulative_returns.iloc[-1] <= 0:
        return 0.0
    return (cumulative_returns.iloc[-1] ** (1 / years)) - 1

def calculate_sharpe(returns, risk_free_rate=0.0):
    """Sharpe Ratio (Risk-adjusted return)"""
    mean_ret = returns.mean() * 252 - risk_free_rate
    vol = returns.std() * np.sqrt(252)
    if vol < 1e-6 or pd.isna(vol):
        return 0.0
    return mean_ret / vol

def calculate_sortino(returns, risk_free_rate=0.0):
    """Sortino Ratio (Downside risk-adjusted return)"""
    mean_ret = returns.mean() * 252 - risk_free_rate
    downside_returns = returns[returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(252)
    if downside_vol < 1e-6 or pd.isna(downside_vol):
        return 0.0
    return mean_ret / downside_vol

def calculate_mdd(cumulative_returns):
    """Maximum Drawdown"""
    rolling_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    return drawdown.min()

def calculate_all_metrics(returns_df):
    """Calculates all KPIs for a DataFrame of strategy returns."""
    metrics = {}
    
    # FIX: Directly get the length of the DataFrame instead of treating it like a dict
    years = len(returns_df) / 252.0
    
    # pandas DataFrames support .items() natively (yielding column_name, series)
    for name, rets in returns_df.items():
        cum_rets = (1 + rets).cumprod()
        metrics[name] = {
            "CAGR": calculate_cagr(cum_rets, years),
            "Sharpe": calculate_sharpe(rets),
            "Sortino": calculate_sortino(rets),
            "MDD": calculate_mdd(cum_rets),
            "Calmar": calculate_cagr(cum_rets, years) / abs(calculate_mdd(cum_rets)) if abs(calculate_mdd(cum_rets)) > 1e-6 else 0.0,
            "Volatility": rets.std() * np.sqrt(252)
        }
        
    return pd.DataFrame(metrics).T