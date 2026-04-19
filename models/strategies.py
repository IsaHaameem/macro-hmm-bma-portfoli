"""
models/strategies.py
Generates daily target allocation weights for three distinct market strategies.
"""
import numpy as np
import pandas as pd

class StrategyEngine:
    @staticmethod
    def momentum(returns, window=21):
        """Buys assets with the highest recent returns."""
        mom = returns.rolling(window).mean()
        # ReLU activation (only buy positive momentum) and normalize
        weights = mom.apply(lambda x: np.maximum(x, 0) / (np.sum(np.maximum(x, 0)) + 1e-6), axis=1)
        return weights.fillna(1.0 / returns.shape[1])

    @staticmethod
    def mean_reversion(returns, window=5):
        """Buys assets that have recently dropped (buying the dip)."""
        rev = -returns.rolling(window).mean()
        # ReLU activation and normalize
        weights = rev.apply(lambda x: np.maximum(x, 0) / (np.sum(np.maximum(x, 0)) + 1e-6), axis=1)
        return weights.fillna(1.0 / returns.shape[1])

    @staticmethod
    def low_volatility(returns, window=21):
        """Allocates more capital to assets with lower variance."""
        vol = returns.rolling(window).std()
        inv_vol = 1.0 / (vol + 1e-6)
        # Normalize inverses so lowest vol gets highest weight
        weights = inv_vol.div(inv_vol.sum(axis=1), axis=0)
        return weights.fillna(1.0 / returns.shape[1])
        
    @classmethod
    def get_all_weights(cls, returns):
        print("Generating strategy weight matrices...")
        return {
            "Momentum": cls.momentum(returns),
            "MeanReversion": cls.mean_reversion(returns),
            "LowVolatility": cls.low_volatility(returns)
        }