"""
config.py
Global configuration, hyperparameters, and runtime flags for Macro-HMM-BMA pipeline.
"""

# Runtime Flags
USE_PRETRAINED = True       # Load saved PPO model or fall back to BMA weights
USE_SAMPLE_DATA = False     # Force sample CSVs even if processed data exists

# Bayesian Model Averaging (BMA) Params
BMA_LAMBDA = 10.0           # BMA sharpness (higher = winner-takes-all)

# Hidden Markov Model (HMM) Params
HMM_N_STATES = 3            # Bull / Bear / Volatile
HMM_COV_TYPE = "diag"       # Diagonal covariance (numerically stable)
HMM_MIN_COVAR = 1e-3        # Variance floor — prevents degenerate HMM states

# Proximal Policy Optimization (PPO) Params
PPO_TOTAL_TIMESTEPS = 500_000 # PPO training budget (~30 min on CPU)

# Preprocessing Params
ROLLING_NORM_WINDOW = 252   # Z-score window (no look-ahead bias)