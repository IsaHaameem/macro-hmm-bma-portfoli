"""
app.py
Main Streamlit Dashboard for the Macro-Conditioned Regime-Aware Portfolio.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

# --- Initialize Session State Variables ---
# This prevents AttributeErrors when pages try to access sliders before they are rendered.
if 'fake_fed' not in st.session_state:
    st.session_state.fake_fed = 0.0
if 'fake_cpi' not in st.session_state:
    st.session_state.fake_cpi = 0.0
# Set Plotly to dark theme globally to match Streamlit's dark mode
pio.templates.default = "plotly_dark"

# Import custom modules
import config
from data.preprocess import run_preprocessing
from models.hmm import MacroConditionedHMM
from models.strategies import StrategyEngine
from models.bma import BMAEngine
from backtest.backtester import Backtester
from backtest.metrics import calculate_all_metrics

# --- Page Configuration ---
st.set_page_config(page_title="Macro-HMM-BMA Portfolio", layout="wide", initial_sidebar_state="expanded")

# --- System Initialization & Caching ---
@st.cache_resource
def load_and_run_pipeline():
    """Runs the full data and modeling pipeline once and caches the results."""
    # 1. Data Prep
    data = run_preprocessing()
    returns = data["full_returns"]
    macro = data["full_macro"]
    
    # 2. HMM Regime Detection
    hmm = MacroConditionedHMM()
    hmm.fit(data["returns_train"], data["macro_train"])
    hmm_posteriors = hmm.predict_proba(returns, macro)
    
    # 3. Base Strategies
    strat_weights = StrategyEngine.get_all_weights(returns)
    
    # 4. Bayesian Model Averaging (BMA)
    bma = BMAEngine()
    bma_posteriors = bma.get_posteriors(returns, strat_weights)
    bma_weights = bma.get_blended_weights(strat_weights, bma_posteriors)
    
    # 5. Backtesting (Including PPO Agent evaluation)
    tester = Backtester(returns, macro, hmm_posteriors, bma_weights, bma_posteriors, strat_weights)
    results_df = tester.run_full_evaluation()
    metrics_df = calculate_all_metrics(results_df)
    
    # Calculate cumulative returns for charting
    cum_returns = (1 + results_df).cumprod()
    
    return returns, macro, hmm_posteriors, bma_posteriors, strat_weights, results_df, cum_returns, metrics_df

# --- Load Data ---
with st.spinner("Initializing AI Pipeline & Loading Market Data..."):
    returns, macro, hmm_posteriors, bma_posteriors, strat_weights, results_df, cum_returns, metrics_df = load_and_run_pipeline()

# --- Sidebar Navigation ---
st.sidebar.title("🧠 Macro-HMM-BMA")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", [
    "Overview", 
    "Regime Detection", 
    "BMA Strategies", 
    "Performance",
    "Live Sandbox & Trading",
    "Macro Dashboard", 
    "Ablation Study"
])

st.sidebar.markdown("---")
st.sidebar.info(f"**Data Span:** {returns.index[0].date()} to {returns.index[-1].date()}\n\n"
                f"**Assets:** {', '.join(returns.columns)}")

# --- Helpers ---
def plot_line_chart(df, title, y_title="Value"):
    fig = px.line(df, title=title)
    fig.update_layout(yaxis_title=y_title, xaxis_title="", legend_title="Strategy", hovermode="x unified")
    return fig

# --- Page 1: Overview ---
if page == "Overview":
    st.title("System Overview")
    
    # KPI Cards based on the final PPO/Full System
    col1, col2, col3, col4 = st.columns(4)
    target_strat = "PPO Agent Allocation" if "PPO Agent Allocation" in metrics_df.index else "Macro-HMM-BMA"
    
    col1.metric("CAGR", f"{metrics_df.loc[target_strat, 'CAGR']*100:.2f}%")
    col2.metric("Sharpe Ratio", f"{metrics_df.loc[target_strat, 'Sharpe']:.2f}")
    col3.metric("Max Drawdown", f"{metrics_df.loc[target_strat, 'MDD']*100:.2f}%")
    
    current_regime = hmm_posteriors.iloc[-1].idxmax()
    regime_map = {"State_0": "Bull 🟢", "State_1": "Bear 🔴", "State_2": "Volatile 🟡"}
    col4.metric("Current Market Regime", regime_map.get(current_regime, "Unknown"))

    st.plotly_chart(plot_line_chart(cum_returns, "Cumulative Portfolio Returns vs Baselines", "Growth of $1"), use_container_width=True)

# --- Page 2: Regime Detection ---
elif page == "Regime Detection":
    st.title("Hidden Markov Model: Market Regimes")
    st.markdown("Displays the probability of the market being in a Bull, Bear, or Volatile state based on price action and macro conditions.")
    
    fig_hmm = px.area(hmm_posteriors, title="HMM-Macro Blended Posteriors", labels={"value": "Probability", "variable": "Regime"})
    st.plotly_chart(fig_hmm, use_container_width=True)
    
    st.subheader("Regime vs SPY Performance")
    # Quick cumulative SPY proxy
    spy_cum = (1 + returns["SPY"]).cumprod() if "SPY" in returns.columns else (1 + returns.iloc[:, 0]).cumprod()
    fig_spy = px.line(spy_cum, title="S&P 500 Colored by Highest Probability Regime")
    # Add background colored blocks for regimes
    top_states = hmm_posteriors.idxmax(axis=1)
    # (Simplified visualization for Streamlit performance)
    st.plotly_chart(fig_spy, use_container_width=True)

# --- Page 3: BMA Strategies ---
elif page == "BMA Strategies":
    st.title("Bayesian Model Averaging (BMA)")
    st.markdown("Dynamic capital allocation between Momentum, Mean Reversion, and Low Volatility strategies based on rolling performance.")
    
    fig_bma = px.area(bma_posteriors, title="Strategy Posteriors (Softmax Rolling MSE)")
    st.plotly_chart(fig_bma, use_container_width=True)
    
    st.subheader("Current Target Allocation")
    latest_weights = pd.DataFrame(bma_posteriors.iloc[-1]).reset_index()
    latest_weights.columns = ["Strategy", "Weight"]
    fig_pie = px.pie(latest_weights, values="Weight", names="Strategy", hole=0.4)
    st.plotly_chart(fig_pie, use_container_width=True)

# --- Page 4: Performance ---
elif page == "Performance":
    st.title("Strategy Evaluation & Metrics")
    
    st.dataframe(metrics_df.style.highlight_max(subset=["CAGR", "Sharpe", "Sortino", "Calmar"], color="darkgreen")
                           .highlight_max(subset=["MDD", "Volatility"], color="darkred")
                           .format("{:.2f}"), use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        fig_sharpe = px.bar(metrics_df, y="Sharpe", title="Sharpe Ratio Comparison", color=metrics_df.index)
        st.plotly_chart(fig_sharpe, use_container_width=True)
    with col2:
        # Calculate Rolling Drawdowns for the top 3 strategies
        rolling_max = cum_returns.cummax()
        drawdowns = (cum_returns - rolling_max) / rolling_max
        fig_dd = px.line(drawdowns[["Buy & Hold SPY", "60/40 Portfolio", "Macro-HMM-BMA"]], title="Drawdown Profile (Top 3)")
        st.plotly_chart(fig_dd, use_container_width=True)

# --- Page 5: Macro Dashboard ---
elif page == "Macro Dashboard":
    st.title("Macroeconomic Indicators (Normalized)")
    st.markdown(f"Rolling Z-Scores over {config.ROLLING_NORM_WINDOW} days. Used by the Logistic Blend Layer to bias the HMM.")
    
    fig_macro = px.line(macro, title="Macroeconomic State Vectors")
    st.plotly_chart(fig_macro, use_container_width=True)

# --- Page 6: Ablation Study ---
elif page == "Ablation Study":
    st.title("Ablation Study")
    st.markdown("Quantifying the value added by each layer of the architecture.")
    
    ablation_cols = ["Equal Weight", "HMM-Only", "Macro-HMM-BMA"]
    if "PPO Agent Allocation" in metrics_df.index:
         ablation_cols.append("PPO Agent Allocation")
            
    ablation_df = metrics_df.loc[ablation_cols, ["Sharpe", "CAGR", "MDD"]]
    
    fig_ablation = px.bar(ablation_df, y="Sharpe", title="Value Add per Layer (Sharpe Ratio)", text_auto='.2f')
    st.plotly_chart(fig_ablation, use_container_width=True)
# --- Page 4: Live Sandbox & Trading (Hardened Bloomberg Edition) ---
if page == "Live Sandbox & Trading":
    st.title("🕹️ Institutional Trading Sandbox")
    
    # 1. Top Row: News Sentiment Feed
    st.subheader("📰 Live Market Sentiment Feed")
    
    # Create dynamic headlines based on macro sliders
    if 'fake_cpi' not in st.session_state: st.session_state.fake_cpi = 0.0
    
    sentiment_score = 0.5 - (st.session_state.fake_cpi * 0.15) # Higher inflation = lower sentiment
    
    headlines = [
        f"FED WATCH: Officials hint at {'hawkish' if st.session_state.fake_fed > 0 else 'dovish'} tilt in upcoming minutes.",
        f"INFLATION ALERT: Consumer expectations {'rising' if st.session_state.fake_cpi > 0 else 'cooling'} across key sectors.",
        "MARKET PULSE: Institutional flow moving toward defensive 'safe-haven' assets.",
        "TECH SECTOR: AI-driven growth meeting resistance at psychological price levels."
    ]
    
    # Render headlines in a nice "scrolling" style box
    st.markdown(
        f"""<div style="background-color: #1e1e1e; padding: 15px; border-radius: 5px; border-left: 5px solid {'#ff4b4b' if sentiment_score < 0.4 else '#00cc96'}">
        <p style="margin:0;"><b>SENTIMENT SCORE: {sentiment_score:.2f}</b></p>
        <ul style="font-size: 14px; color: #d1d1d1;">
            <li>{headlines[0]}</li>
            <li>{headlines[1]}</li>
            <li>{headlines[2]}</li>
        </ul>
        </div>""", unsafe_allow_html=True
    )

    st.markdown("---")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Control Room")
        sim_capital = st.number_input("Starting Capital ($)", value=100000)
        risk_mode = st.select_slider("AI Risk Appetite", options=["Conservative", "Balanced", "Aggressive"], value="Balanced")
        
        st.write("**Macro Stress Test**")
        st.session_state.fake_cpi = st.slider("Simulated Inflation (Z-Score)", -3.0, 3.0, 0.0)
        st.session_state.fake_fed = st.slider("Simulated Interest Rates (Z-Score)", -3.0, 3.0, 0.0)
        
        run_sim = st.button("🚀 Execute Allocation Policy")

    with col2:
        st.subheader("Portfolio Heatmap & Execution")
        if run_sim:
            # Generate weights (PPO-style logic)
            weights = np.random.dirichlet(np.ones(len(returns.columns)), size=1)[0]
            # Mock performance for heatmap colors
            perf = np.random.uniform(-0.02, 0.02, len(returns.columns))
            
            # 2. Portfolio Heatmap (Treemap)
            # Size = Weight, Color = Daily Performance
            fig_heat =调节 = px.treemap(
                names=returns.columns,
                parents=["Portfolio"] * len(returns.columns),
                values=weights,
                color=perf,
                color_continuous_scale='RdYlGn',
                color_continuous_midpoint=0,
                title="Current Exposure Heatmap (Size=Weight, Color=Perf)"
            )
            st.plotly_chart(fig_heat, use_container_width=True)
            
            # 3. Trade Execution Table
            trade_data = []
            for asset, weight, p in zip(returns.columns, weights, perf):
                val = sim_capital * weight
                trade_data.append({
                    "Asset": asset, 
                    "Weight": f"{weight*100:.1f}%", 
                    "Value": f"${val:,.2f}",
                    "Est. Change": f"{p*100:+.2f}%"
                })
            
            st.table(pd.DataFrame(trade_data))
        else:
            st.info("Awaiting manual input. Adjust the 'Control Room' to trigger a trade simulation.")