"""
models/hmm.py
Macro-Conditioned Hidden Markov Model for Regime Detection.
Blends price-based HMM posteriors with macro-based logistic regression posteriors.
"""
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.linear_model import LogisticRegression
import config

class MacroConditionedHMM:
    def __init__(self, alpha=0.7):
        self.alpha = alpha
        self.n_states = config.HMM_N_STATES
        
        # Base HMM: Diagonal covariance is numerically stable
        self.hmm = GaussianHMM(
            n_components=self.n_states,
            covariance_type=config.HMM_COV_TYPE,
            min_covar=config.HMM_MIN_COVAR,
            random_state=42,
            n_iter=100
        )
        
        # Macro Blend Layer
        self.macro_model = LogisticRegression(max_iter=1000, random_state=42)
    def fit(self, returns_df, macro_df):
        print("Training Macro-Conditioned HMM...")
        
        # 1. Fit Base HMM on SPY returns (Market Proxy)
        spy_returns = returns_df[['SPY']].values
        self.hmm.fit(spy_returns)
        
        # 2. Extract latent states (Viterbi path) to use as targets for the macro model
        hidden_states = self.hmm.predict(spy_returns)
        
        # 3. Fit Logistic Regression mapping Macro conditions to HMM states
        self.macro_model.fit(macro_df.values, hidden_states)
        print("HMM and Macro Blend Layer successfully trained.")

    def predict_proba(self, returns_df, macro_df):
        """
        Returns blended posteriors: [P(State 0), P(State 1), P(State 2)] per day.
        """
        spy_returns = returns_df[['SPY']].values
        
        # Get posteriors from HMM (Guaranteed to be N x n_states)
        hmm_probs = self.hmm.predict_proba(spy_returns)
        
        # Get raw posteriors from Macro model
        raw_macro_probs = self.macro_model.predict_proba(macro_df.values)
        
        # FIX: Force macro_probs to be exactly (N x n_states), mapping only the classes it learned.
        # This handles the case where the training set didn't contain all 3 HMM states.
        macro_probs = np.zeros((len(macro_df), self.n_states))
        for i, cls in enumerate(self.macro_model.classes_):
            macro_probs[:, cls] = raw_macro_probs[:, i]
        
        # Alpha blending (Default: 70% Price action, 30% Macro conditions)
        blended_probs = (self.alpha * hmm_probs) + ((1 - self.alpha) * macro_probs)
        
        # Return as DataFrame for easy alignment
        return pd.DataFrame(blended_probs, index=returns_df.index, columns=[f"State_{i}" for i in range(self.n_states)])