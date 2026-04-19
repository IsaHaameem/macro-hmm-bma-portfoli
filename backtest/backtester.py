"""
backtest/backtester.py
Walk-forward evaluation engine comparing 7 strategy baselines.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from models.ppo_agent import predict_ppo

class Backtester:
    def __init__(self, returns, macro, hmm_posteriors, bma_weights, bma_posteriors, strategy_weights):
        self.returns = returns
        self.macro = macro
        self.hmm = hmm_posteriors
        self.bma_weights = bma_weights
        self.bma_posteriors = bma_posteriors  # <-- Added this
        self.strat_weights = strategy_weights
        self.assets = returns.columns
        self.dates = returns.index

    def run_baselines(self):
        print("Running walk-forward backtest across all baselines...")
        portfolio_returns = {}

        # 1. Equal Weight (1/N)
        eq_weights = pd.DataFrame(1.0 / len(self.assets), index=self.dates, columns=self.assets)
        portfolio_returns["Equal Weight"] = (eq_weights.shift(1) * self.returns).sum(axis=1)

        # 2. Buy & Hold SPY
        spy_weights = pd.DataFrame(0.0, index=self.dates, columns=self.assets)
        if "SPY" in self.assets:
            spy_weights["SPY"] = 1.0
        portfolio_returns["Buy & Hold SPY"] = (spy_weights.shift(1) * self.returns).sum(axis=1)

        # 3. 60/40 (SPY / TLT)
        sixty_forty = pd.DataFrame(0.0, index=self.dates, columns=self.assets)
        if "SPY" in self.assets and "TLT" in self.assets:
            sixty_forty["SPY"] = 0.6
            sixty_forty["TLT"] = 0.4
        portfolio_returns["60/40 Portfolio"] = (sixty_forty.shift(1) * self.returns).sum(axis=1)

        # 4. HMM-Only (Hard switch based on highest probability state)
        hmm_only_returns = pd.Series(0.0, index=self.dates)
        top_states = self.hmm.idxmax(axis=1)
        mom_ret = (self.strat_weights["Momentum"].shift(1) * self.returns).sum(axis=1)
        lv_ret = (self.strat_weights["LowVolatility"].shift(1) * self.returns).sum(axis=1)
        mr_ret = (self.strat_weights["MeanReversion"].shift(1) * self.returns).sum(axis=1)

        hmm_only_returns[top_states == "State_0"] = mom_ret[top_states == "State_0"]
        hmm_only_returns[top_states == "State_1"] = lv_ret[top_states == "State_1"]
        hmm_only_returns[top_states == "State_2"] = mr_ret[top_states == "State_2"]
        portfolio_returns["HMM-Only"] = hmm_only_returns

        # 5. Full System (Macro-HMM-BMA) without RL
        portfolio_returns["Macro-HMM-BMA"] = (self.bma_weights.shift(1) * self.returns).sum(axis=1)

        # 6. Markowitz MVO (Monthly Rebalance to maximize Sharpe)
        print("Calculating Markowitz MVO (This takes ~5 seconds)...")
        mvo_weights = pd.DataFrame(0.0, index=self.dates, columns=self.assets)
        init_guess = np.array([1.0 / len(self.assets)] * len(self.assets))
        bounds = tuple((0.0, 1.0) for _ in range(len(self.assets)))
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})

        last_weights = init_guess
        for i in range(len(self.dates)):
            if i >= 60 and i % 21 == 0:  # Rebalance every 21 trading days
                window_returns = self.returns.iloc[i-60:i]
                mu = window_returns.mean() * 252
                cov = window_returns.cov() * 252
                
                def neg_sharpe(w):
                    ret = np.sum(w * mu)
                    vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
                    return -ret / vol if vol > 1e-6 else 0.0

                res = minimize(neg_sharpe, last_weights, method='SLSQP', bounds=bounds, constraints=constraints)
                if res.success: last_weights = res.x
            mvo_weights.iloc[i] = last_weights

        portfolio_returns["Markowitz MVO"] = (mvo_weights.shift(1) * self.returns).sum(axis=1)

        return pd.DataFrame(portfolio_returns)

    def run_full_evaluation(self):
        """Runs baselines + PPO Agent predictions"""
        results_df = self.run_baselines()
        
        # 7. PPO Agent Allocation
        # 7. PPO Agent Allocation
        ppo_allocations = predict_ppo(
            self.returns, self.macro, self.hmm, 
            self.bma_posteriors, self.strat_weights  # <-- Changed from bma_weights
        )
        
        mom_ret = (self.strat_weights["Momentum"].shift(1) * self.returns).sum(axis=1)
        mr_ret = (self.strat_weights["MeanReversion"].shift(1) * self.returns).sum(axis=1)
        lv_ret = (self.strat_weights["LowVolatility"].shift(1) * self.returns).sum(axis=1)
        
        strat_returns = pd.DataFrame({"Momentum": mom_ret, "MeanReversion": mr_ret, "LowVolatility": lv_ret})
        
        results_df["PPO Agent Allocation"] = (ppo_allocations.shift(1) * strat_returns).sum(axis=1)
        
        return results_df