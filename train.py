"""
train.py
Executes the Reinforcement Learning training loop for the PPO Agent.
Run this script once to generate the saved model.
"""
import config
from data.preprocess import run_preprocessing
from models.hmm import MacroConditionedHMM
from models.strategies import StrategyEngine
from models.bma import BMAEngine
from models.ppo_agent import train_ppo

def main():
    print("Starting PPO Training Pipeline...")
    
    # Temporarily override config to force training behavior
    config.USE_PRETRAINED = False
    
    # 1. Load Data (Train split only)
    data = run_preprocessing()
    returns_train = data["returns_train"]
    macro_train = data["macro_train"]
    
    # 2. Fit HMM
    hmm = MacroConditionedHMM()
    hmm.fit(returns_train, macro_train)
    hmm_posteriors = hmm.predict_proba(returns_train, macro_train)
    
    # 3. Generate Base Strategy Weights
    strat_weights = StrategyEngine.get_all_weights(returns_train)
    
    # 4. Generate BMA Posteriors
    bma = BMAEngine()
    bma_posteriors = bma.get_posteriors(returns_train, strat_weights)
    
    # 5. Train the PPO Agent
    # (By default this runs for 500,000 steps as set in config.py)
    train_ppo(returns_train, macro_train, hmm_posteriors, bma_posteriors, strat_weights)

if __name__ == "__main__":
    main()