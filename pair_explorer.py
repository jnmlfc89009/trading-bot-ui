"""
Pairs Trading Pro Explorer
--------------------------
An interactive Streamlit dashboard for researching and validating 
statistical relationships between stock pairs.

Features:
- Live data fetching via yfinance
- Z-Score and Cointegration analysis
- RSI Momentum verification
- Secure input sanitization
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint
import re

# --- UI CONFIGURATION ---
st.set_page_config(page_title="Pairs Trading Pro", layout="wide")

# --- PROMINENT LEGAL DISCLAIMER ---
st.warning("⚠️ **LEGAL DISCLAIMER:** This tool is for **educational and informational purposes only**. Algorithmic trading involves a high risk of losing money. The creator of this tool is not a financial advisor. Past performance does not guarantee future results. **Use this tool at your own risk.**")

st.title("📈 Pairs Trading Pro Explorer")
st.markdown("Enter two tickers to analyze their statistical relationship and identify mean-reversion signals.")

# --- SIDEBAR: USER INPUTS ---
st.sidebar.header("Pair Configuration")

def sanitize_ticker(text):
    """Removes special characters to prevent script injection and ensures uppercase."""
    return re.sub(r'[^A-Z0-9.\-]', '', text.upper())

t1_raw = st.sidebar.text_input("Ticker 1", value="LOW")
t2_raw = st.sidebar.text_input("Ticker 2", value="HD")

t1 = sanitize_ticker(t1_raw)
t2 = sanitize_ticker(t2_raw)
time_period = st.sidebar.selectbox("Lookback Period", options=["3mo", "6mo", "1y", "2y"], index=1)

# --- ANALYTICS ENGINE ---
def calculate_rsi(series, periods=14):
    """Calculates the Relative Strength Index (RSI) for momentum checks."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

if st.sidebar.button("Run Deep Analysis"):
    with st.spinner(f"Fetching market data for {t1} and {t2}..."):
        # 1. DATA INGESTION
        data = yf.download([t1, t2], period=time_period, progress=False)
        
        if not data.empty and len(data.columns) >= 2:
            prices = data['Close'].dropna()
            
            # 2. PRICE HIERARCHY LOGIC
            # Automatically identifies which stock is the 'primary' based on price
            p1_last = prices[t1].iloc[-1]
            p2_last = prices[t2].iloc[-1]
            
            if p1_last > p2_last:
                high_ticker, low_ticker = t1, t2
                high_price, low_price = p1_last, p2_last
            else:
                high_ticker, low_ticker = t2, t1
                high_price, low_price = p2_last, p1_last
            
            # 3. STATISTICAL CALCULATIONS
            # Hedge Ratio: Used to equalize the dollar value of both positions
            ratio = high_price / low_price
            # Spread: The difference between the two stocks over time
            spread = prices[high_ticker] - (ratio * prices[low_ticker])
            # Z-Score: Standardizes the spread to find statistical outliers
            z_score = (spread - spread.mean()) / spread.std()
            curr_z = z_score.iloc[-1]
            
            # Correlation: Measures how similarly they move
            correlation = prices[t1].corr(prices[t2])
            # Cointegration: Measures if they stay tethered over the long term
            score, pvalue, _ = coint(prices[t1], prices[t2])
            
            # RSI: Measures if the stocks are currently overextended
            rsi1 = calculate_rsi(prices[t1]).iloc[-1]
            rsi2 = calculate_rsi(prices[t2]).iloc[-1]

            # --- UI: DASHBOARD ---
            st.subheader("System Health & Live Metrics")
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            with m_col1:
                st.metric(f"💎 High ({high_ticker})", f"${high_price:.2f}")
            with m_col2:
                st.metric(f"🪙 Low ({low_ticker})", f"${low_price:.2f}")
            with m_col3:
                st.metric("🔗 Correlation", f"{correlation:.2%}")
            with m_col4:
                st.metric("🎯 Current Z", f"{curr_z:.2f}")

            # --- UI: VISUALIZATION ---
            st.subheader("Statistical Divergence Analysis")
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            
            # Top Plot: Normalized path starting at 100
            norm1 = (prices[t1] / prices[t1].iloc[0]) * 100
            norm2 = (prices[t2] / prices[t2].iloc[0]) * 100
            ax1.plot(norm1, label=f"{t1} (Normalized)", color='#2ecc71', linewidth=2)
            ax1.plot(norm2, label=f"{t2} (Normalized)", color='#3498db', linewidth=2)
            ax1.set_title("Price Correlation (Historical Convergence)", fontsize=14)
            ax1.legend()
            ax1.grid(alpha=0.3)

            # Bottom Plot: Z-Score divergence
            ax2.plot(z_score, color='#9b59b6', linewidth=2)
            ax2.axhline(2.0, color='#e74c3c', linestyle='--', label="Sell Threshold")
            ax2.axhline(-2.0, color='#27ae60', linestyle='--', label="Buy Threshold")
            ax2.axhline(0, color='black', linestyle=':', alpha=0.5)
            ax2.set_title("Z-Score (Divergence Level)", fontsize=14)
            ax2.fill_between(z_score.index, z_score, 0, where=(z_score >= 0), color='#9b59b6', alpha=0.1)
            ax2.fill_between(z_score.index, z_score, 0, where=(z_score < 0), color='#9b59b6', alpha=0.1)
            plt.tight_layout()
            st.pyplot(fig)
            
            # --- UI: RISK ASSESSMENT ---
            st.divider()
            st.subheader("🕵️ Deep Risk & Suitability Assessment")
            r_col1, r_col2 = st.columns(2)
            
            with r_col1:
                st.markdown("#### 📐 Statistical Foundation")
                corr_pass = correlation > 0.85
                coint_pass = pvalue < 0.05
                
                if corr_pass:
                    st.success(f"**Correlation:** {correlation:.2%} (Strong relationship)")
                else:
                    st.warning(f"**Correlation:** {correlation:.2%} (Weak relationship - Risky)")
                
                if coint_pass:
                    st.success(f"**Cointegration:** P-Value {pvalue:.4f} (Significant)")
                else:
                    st.error(f"**Cointegration:** P-Value {pvalue:.4f} (Not Significant)")

            with r_col2:
                st.markdown("#### ⚡ Individual Stock Momentum")
                rsi_neutral = (30 < rsi1 < 70) and (30 < rsi2 < 70)
                st.write(f"**{t1} RSI:** {rsi1:.2f}")
                st.write(f"**{t2} RSI:** {rsi2:.2f}")
                
                if rsi_neutral:
                    st.success("✅ Momentum is neutral. Safe for statistical entry.")
                else:
                    st.warning("⚠️ Momentum Alert: One stock is reaching an extreme (Overbought/Oversold).")

            # --- UI: FINAL INTEGRATED TRADING VERDICT ---
            st.markdown("### 🏁 Final Execution Signal")
            
            z_signal = abs(curr_z) >= 2.0
            
            if z_signal and coint_pass and corr_pass and rsi_neutral:
                if curr_z >= 2.0:
                    st.error(f"🚀 **STRONG SELL SIGNAL:** {high_ticker} is statistically overvalued vs {low_ticker}. Entry recommended.")
                else:
                    st.success(f"🚀 **STRONG BUY SIGNAL:** {high_ticker} is statistically undervalued vs {low_ticker}. Entry recommended.")
            
            elif z_signal:
                reasons = []
                if not coint_pass: reasons.append("Weak Cointegration")
                if not corr_pass: reasons.append("Low Correlation")
                if not rsi_neutral: reasons.append("Extreme RSI")
                st.warning(f"⚠️ **CAUTION:** Z-Score is at signal level ({curr_z:.2f}), but entry is discouraged due to: {', '.join(reasons)}.")
            
            elif not z_signal and coint_pass and corr_pass:
                st.info("⚖️ **WAITING:** Relationship is strong, but stocks are currently at equilibrium. No entry yet.")
            
            else:
                st.info("⚖️ **NO SIGNAL:** No reliable statistical divergence detected.")
                
            st.caption("Disclaimer: This tool provides mathematical analysis only. All trading involves capital risk.")

        else:
            st.error("Error: Could not retrieve data. Please ensure the tickers are correct and include suffixes for non-US stocks (e.g. .SI).")
