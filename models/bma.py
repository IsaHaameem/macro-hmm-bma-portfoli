"""
models/bma.py
Bayesian Model Averaging (BMA) Engine.
Evaluates strategy performance using rolling MSE and generates softmax posteriors.
"""
import numpy as np
import pandas as pd
import config

class BMAEngine:
    def __init__(self, window=20, bma_lambda=config.BMA_LAMBDA):
        self.window = window
        self.bma_lambda = bma_lambda

    def fit(self, returns_df):
        """BMA is purely dynamic based on rolling inference, but we keep fit() for pipeline consistency."""
        print("Initializing BMA Engine...")
        pass

    def get_posteriors(self, returns_df, strategy_weights):
        """
        Calculates BMA posteriors using softmax(-lambda * MSE) over a rolling window.
        """
        print("Calculating BMA strategy posteriors...")
        
        # 1. Calculate daily returns for each strategy
        strat_returns = pd.DataFrame(index=returns_df.index)
        for name, weight_df in strategy_weights.items():
            # Shift weights by 1 day to prevent look-ahead bias
            shifted_weights = weight_df.shift(1).fillna(0)
            strat_returns[name] = (shifted_weights * returns_df).sum(axis=1)

        # 2. Define the "Ideal" return as the maximum return achieved by any strategy that day
        ideal_returns = strat_returns.max(axis=1)

        # 3. Calculate Rolling MSE for each strategy against the ideal return
        rolling_mse = pd.DataFrame(index=returns_df.index)
        for name in strategy_weights.keys():
            sq_error = (strat_returns[name] - ideal_returns) ** 2
            # Fill initial empty window with a high error penalty
            rolling_mse[name] = sq_error.rolling(self.window).mean().fillna(1.0) 

        # 4. Apply Softmax(-lambda * MSE)
        scaled_mse = -self.bma_lambda * rolling_mse
        
        # Subtract max for numerical stability before exponentiating
        scaled_mse_stable = scaled_mse.subtract(scaled_mse.max(axis=1), axis=0)
        exp_mse = np.exp(scaled_mse_stable)
        
        # Normalize to create probabilities (posteriors)
        posteriors = exp_mse.div(exp_mse.sum(axis=1), axis=0)

        # Fallback to equal weighting for the first few days
        return posteriors.fillna(1.0 / len(strategy_weights))

    def get_blended_weights(self, strategy_weights, posteriors):
        """
        Multiplies the base strategy weights by their daily BMA posterior probabilities.
        """
        blended = None
        for name in strategy_weights.keys():
            weighted_strat = strategy_weights[name].multiply(posteriors[name], axis=0)
            if blended is None:
                blended = weighted_strat
            else:
                blended += weighted_strat

        # Ensure weights sum to 1.0
        return blended.div(blended.sum(axis=1), axis=0).fillna(0)