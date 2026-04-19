"""
models/ppo_agent.py
Handles training and inference of the Stable-Baselines3 PPO agent.
"""
import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
from models.ppo_env import PortfolioEnv
import config

def get_strategy_returns(returns, strategy_weights):
    """Helper to calculate daily returns for each strategy."""
    strat_returns = pd.DataFrame(index=returns.index)
    for name, weights in strategy_weights.items():
        shifted_weights = weights.shift(1).fillna(0)
        strat_returns[name] = (shifted_weights * returns).sum(axis=1)
    return strat_returns

def train_ppo(returns, macro, hmm_posteriors, bma_posteriors, strategy_weights, total_timesteps=config.PPO_TOTAL_TIMESTEPS):
    print(f"Initializing PPO Environment with {torch.get_num_threads()} threads...")
    
    strat_returns = get_strategy_returns(returns, strategy_weights)
    
    # Wrap env for Stable-Baselines3
    env = DummyVecEnv([lambda: PortfolioEnv(returns, macro, hmm_posteriors, bma_posteriors, strat_returns)])
    
    # 3x256 Architecture per spec
    policy_kwargs = dict(net_arch=dict(pi=[256, 256, 256], vf=[256, 256, 256]))
    
    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, learning_rate=3e-4)
    
    print(f"Training PPO Agent for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps)
    
    os.makedirs("models/saved", exist_ok=True)
    model.save("models/saved/ppo_portfolio_agent")
    print("Training complete! Model saved to models/saved/ppo_portfolio_agent.zip")
    return model

def predict_ppo(returns, macro, hmm_posteriors, bma_posteriors, strategy_weights):
    if config.USE_PRETRAINED and os.path.exists("models/saved/ppo_portfolio_agent.zip"):
        print("Loading Pre-trained PPO Agent...")
        model = PPO.load("models/saved/ppo_portfolio_agent")
    else:
        print("No pre-trained PPO model found. Falling back to default BMA equal-weight initial state.")
        # Fallback logic handles standard BMA weights if PPO isn't ready
        return pd.DataFrame(1/3, index=returns.index, columns=strategy_weights.keys())

    strat_returns = get_strategy_returns(returns, strategy_weights)
    env = PortfolioEnv(returns, macro, hmm_posteriors, bma_posteriors, strat_returns)
    
    obs, _ = env.reset()
    actions = []
    
    print("Generating PPO Allocations...")
    for _ in range(len(returns)):
        action, _states = model.predict(obs, deterministic=True)
        # Normalize action to weights
        action = np.clip(action, 1e-5, 1.0)
        weights = action / np.sum(action)
        actions.append(weights)
        obs, _, _, _, _ = env.step(action)
        
    return pd.DataFrame(actions, index=returns.index, columns=strategy_weights.keys())