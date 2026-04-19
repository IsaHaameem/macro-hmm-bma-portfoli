"""
models/ppo_env.py
Custom Gymnasium environment for Portfolio Allocation.
State: 18-dim (8 returns, 4 macro, 3 regime, 3 BMA).
Action: 3-dim (allocation weights to Momentum, MeanReversion, LowVol).
Reward: Rolling Sharpe Ratio - (0.001 * turnover).
"""
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

class PortfolioEnv(gym.Env):
    def __init__(self, returns, macro, hmm_posteriors, bma_posteriors, strategy_returns):
        super(PortfolioEnv, self).__init__()
        
        # Align all data to the same dates
        self.dates = returns.index
        self.returns = returns.values
        self.macro = macro.values
        self.hmm = hmm_posteriors.values
        self.bma = bma_posteriors.values
        self.strat_returns = strategy_returns.values
        
        self.n_steps = len(self.dates)
        
        # Action Space: Weights for [Momentum, MeanReversion, LowVolatility]
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # Observation Space: 8 returns + 4 macro + 3 regime + 3 BMA = 18 dimensions
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32)
        
        # Trackers
        self.current_step = 0
        self.portfolio_returns = []
        self.prev_weights = np.array([1/3, 1/3, 1/3])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.portfolio_returns = []
        self.prev_weights = np.array([1/3, 1/3, 1/3])
        return self._get_obs(), {}

    def _get_obs(self):
        # FIX: Clamp the index to prevent Out-Of-Bounds on the final termination step
        idx = min(self.current_step, self.n_steps - 1)
        
        # Concatenate features for the current day
        obs = np.concatenate([
            self.returns[idx],
            self.macro[idx],
            self.hmm[idx],
            self.bma[idx]
        ]).astype(np.float32)
        return obs

    def step(self, action):
        # 1. Normalize action to sum to 1.0 (softmax behavior)
        action = np.clip(action, 1e-5, 1.0)
        weights = action / np.sum(action)
        
        # 2. Calculate Portfolio Return for the day
        day_return = np.sum(weights * self.strat_returns[self.current_step])
        self.portfolio_returns.append(day_return)
        
        # 3. Calculate Turnover Penalty (0.001 = 10 bps per trade)
        turnover = np.sum(np.abs(weights - self.prev_weights))
        penalty = 0.001 * turnover
        self.prev_weights = weights
        
        # 4. Calculate Reward (Rolling Sharpe approx over last 60 days)
        if len(self.portfolio_returns) > 60:
            recent_returns = np.array(self.portfolio_returns[-60:])
            mean_ret = np.mean(recent_returns)
            std_ret = np.std(recent_returns) + 1e-6
            sharpe = (mean_ret / std_ret) * np.sqrt(252)
        else:
            sharpe = 0.0 # Not enough history yet
            
        reward = sharpe - penalty
        
        # 5. Advance Time
        self.current_step += 1
        terminated = self.current_step >= self.n_steps - 1
        truncated = False
        
        info = {"portfolio_return": day_return, "weights": weights}
        
        if terminated:
            return self._get_obs(), reward, terminated, truncated, info
            
        return self._get_obs(), reward, terminated, truncated, info