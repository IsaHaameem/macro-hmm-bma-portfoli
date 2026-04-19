
```markdown
# Macroeconomic-Conditioned Regime-Aware Portfolio Management  
### Hidden Markov Models + Bayesian Model Averaging + Reinforcement Learning  
**Final Year Project — B.Tech / MSc (2025)**

---

## 📌 Overview
This project develops a **regime-aware portfolio management system** that adapts dynamically to changing market conditions.

Traditional portfolio optimization assumes market stationarity. This system instead models financial markets as **latent regimes** (e.g., Bull, Bear, High Volatility) influenced by macroeconomic factors.

A multi-layered AI pipeline is designed to:
- detect market regimes from macro + price data  
- dynamically select investment strategies  
- learn optimal asset allocation policies  

The objective is to **maximize risk-adjusted returns (Sharpe Ratio)** while maintaining robustness across market cycles.

---

## 🧠 System Architecture

The pipeline integrates probabilistic modeling, ensemble learning, and reinforcement learning:

### 1. Regime Detection
- Gaussian Hidden Markov Model (HMM)
- Augmented with Logistic Regression for macro conditioning  
- Inputs: GDP, CPI, Interest Rates  
- Output: Regime probabilities (soft state assignments)

### 2. Strategy Layer
Base strategies:
- Momentum  
- Mean Reversion  
- Low Volatility  

### 3. Bayesian Model Averaging (BMA)
- Dynamically re-weights strategies  
- Based on rolling prediction error (MSE)  
- Uses softmax weighting for smooth transitions  

### 4. Reinforcement Learning (PPO)
- Learns optimal allocation across strategies  
- State space: 18-dimensional feature vector  
- Objective: maximize Sharpe Ratio under uncertainty  

---

## 🏗 Data Flow

```

Market Data (yfinance) + Macro Data (FRED)
│
▼
Feature Engineering
(Log Returns + Rolling Z-Scores)
│
├──► Macro-Conditioned HMM
│       → Regime Probabilities
│
├──► Strategy Engine
│       → Strategy Returns
│
├──► BMA Layer
│       → Dynamic Strategy Weights
│
└──► PPO Agent
→ Optimal Allocation Policy
│
▼
Streamlit Dashboard
(Backtesting + Simulation + Analytics)

````

---

## 🛠 Tech Stack

**Modeling & ML**
- `hmmlearn`, `scikit-learn`
- `stable-baselines3`, `PyTorch`

**Data & Analytics**
- `pandas`, `numpy`, `scipy`

**Visualization & UI**
- `streamlit`, `plotly`

**Data Sources**
- `yfinance` (market data)  
- `FRED API` (macroeconomic indicators)

---

## ⚙️ Setup

### 1. Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
````

### 2. API Configuration

```powershell
$env:FRED_API_KEY = "your_api_key"
```

---

## 🚀 Execution Pipeline

Run sequentially:

### Step 1 — Data Collection

```powershell
python data/fetch_data.py
```

### Step 2 — Preprocessing

```powershell
python data/preprocess.py
```

### Step 3 — Model Training (PPO)

```powershell
python train.py
```

### Step 4 — Dashboard

```powershell
streamlit run app.py
```

---

## 🕹 Key Features

* **Macro Stress Testing**
  Simulate inflation shocks or rate hikes and observe regime transitions

* **Dynamic Allocation Simulation**
  Generate portfolio weights based on current macro conditions

* **Trade Execution Sandbox**
  Simulated trades with configurable capital and risk

* **Ablation Analysis**
  Quantifies contribution of:

  * HMM (regime awareness)
  * BMA (strategy selection)
  * PPO (policy optimization)

---

## 📊 Benchmark Comparison

Evaluated against:

1. Equal Weight (1/N)
2. 60/40 Portfolio (SPY/TLT)
3. Buy & Hold (SPY)
4. Markowitz Mean-Variance Optimization
5. HMM-only Switching
6. HMM + BMA (No RL)
7. PPO-Based Allocation (Final Model)

---

## 🎯 Key Contributions

* Introduces **macro-conditioned regime detection**
* Combines **probabilistic + ensemble + RL approaches**
* Demonstrates **non-stationary market adaptation**
* Provides an **interactive decision-support system**

---

## 📚 Motivation

Financial markets exhibit **regime shifts and structural breaks**, making static models ineffective.

This project frames portfolio management as a **sequential decision-making problem under uncertainty**, enabling:

* proactive risk reduction during downturns
* increased exposure during favorable regimes
* adaptive strategy selection

---

## ⚠️ Disclaimer

This project is for academic purposes only.
It does not constitute financial advice.

---

## 👤 Author

**[Your Name]**
B.Tech Computer Science

**Supervisor:** [Name]
**Institution:** SRM Institute of Science and Technology

