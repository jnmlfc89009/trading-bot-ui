import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint

st.set_page_config(page_title="Pairs Trading Pro", layout="wide")

# --- PROMINENT LEGAL DISCLAIMER ---
st.warning("⚠️ **LEGAL DISCLAIMER:** This tool is for **educational and informational purposes only**. Algorithmic trading involves a high risk of losing money. The creator of this tool is not a financial advisor. Past performance does not guarantee future results. **Use this tool at your own risk.**")

st.title("📈 Pairs Trading Pro Explorer")
st.markdown("Enter two tickers to analyze their statistical relationship and identify mean-reversion signals.")

# --- Sidebar Inputs ---
st.sidebar.header("Pair Configuration")

# SECURITY: Input sanitization to prevent injection and ensure clean API calls
def sanitize_ticker(text):
    import re
    return re.sub(r'[^A-Z0-9.\-]', '', text.upper())

t1_raw = st.sidebar.text_input("Ticker 1", value="LOW")
t2_raw = st.sidebar.text_input("Ticker 2", value="HD")

t1 = sanitize_ticker(t1_raw)
t2 = sanitize_ticker(t2_raw)
time_period = st.sidebar.selectbox("Lookback Period", options=["3mo", "6mo", "1y", "2y"], index=1)

def calculate_rsi(series, periods=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

if st.sidebar.button("Run Deep Analysis"):
    with st.spinner(f"Fetching market data for {t1} and {t2}..."):
        data = yf.download([t1, t2], period=time_period, progress=False)
        
        if not data.empty and len(data.columns) >= 2:
            prices = data['Close'].dropna()
            
            # --- Identify Higher vs Lower Price ---
            p1_last = prices[t1].iloc[-1]
            p2_last = prices[t2].iloc[-1]
            
            if p1_last > p2_last:
                high_ticker, low_ticker = t1, t2
                high_price, low_price = p1_last, p2_last
            else:
                high_ticker, low_ticker = t2, t1
                high_price, low_price = p2_last, p1_last
            
            # --- Math Engine ---
            ratio = high_price / low_price
            spread = prices[high_ticker] - (ratio * prices[low_ticker])
            z_score = (spread - spread.mean()) / spread.std()
            curr_z = z_score.iloc[-1]
            
            correlation = prices[t1].corr(prices[t2])
            score, pvalue, _ = coint(prices[t1], prices[t2])
            
            rsi1 = calculate_rsi(prices[t1]).iloc[-1]
            rsi2 = calculate_rsi(prices[t2]).iloc[-1]

            # --- Metrics Dashboard ---
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

            # --- Visualization ---
            st.subheader("Statistical Divergence Analysis")
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            norm1 = (prices[t1] / prices[t1].iloc[0]) * 100
            norm2 = (prices[t2] / prices[t2].iloc[0]) * 100
            ax1.plot(norm1, label=f"{t1} (Normalized)", color='#2ecc71', linewidth=2)
            ax1.plot(norm2, label=f"{t2} (Normalized)", color='#3498db', linewidth=2)
            ax1.set_title("Price Correlation (Starting at 100)", fontsize=14)
            ax1.legend()
            ax1.grid(alpha=0.3)
            ax2.plot(z_score, color='#9b59b6', linewidth=2)
            ax2.axhline(2.0, color='#e74c3c', linestyle='--', label="Sell Threshold")
            ax2.axhline(-2.0, color='#27ae60', linestyle='--', label="Buy Threshold")
            ax2.axhline(0, color='black', linestyle=':', alpha=0.5)
            ax2.set_title("Z-Score (Divergence Level)", fontsize=14)
            ax2.fill_between(z_score.index, z_score, 0, where=(z_score >= 0), color='#9b59b6', alpha=0.1)
            ax2.fill_between(z_score.index, z_score, 0, where=(z_score < 0), color='#9b59b6', alpha=0.1)
            plt.tight_layout()
            st.pyplot(fig)
            
            # --- COMPREHENSIVE RISK ASSESSMENT ---
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

            # --- FINAL INTEGRATED TRADING VERDICT ---
            st.markdown("### 🏁 Final Execution Signal")
            
            z_signal = abs(curr_z) >= 2.0
            
            # The "Informed" Logic: All 3 pillars must be green for a Strong Signal
            if z_signal and coint_pass and corr_pass and rsi_neutral:
                if curr_z >= 2.0:
                    st.error(f"🚀 **STRONG SELL SIGNAL:** All conditions met. {high_ticker} is extremely overvalued vs {low_ticker}. Probabilistic high-conviction trade.")
                else:
                    st.success(f"🚀 **STRONG BUY SIGNAL:** All conditions met. {high_ticker} is extremely undervalued vs {low_ticker}. Probabilistic high-conviction trade.")
            
            elif z_signal:
                # Z-Score is triggered, but other factors are failing
                reasons = []
                if not coint_pass: reasons.append("Weak Cointegration (P-Value > 0.05)")
                if not corr_pass: reasons.append("Low Correlation")
                if not rsi_neutral: reasons.append("Extreme RSI (Momentum Risk)")
                
                st.warning(f"⚠️ **CAUTION:** Z-Score is at a signal level ({curr_z:.2f}), but the trade is **NOT** recommended due to: {', '.join(reasons)}.")
            
            elif not z_signal and coint_pass and corr_pass:
                st.info("⚖️ **WAITING:** The relationship is statistically perfect, but the stocks are currently at equilibrium. No entry signal yet.")
            
            else:
                st.info("⚖️ **NO SIGNAL:** No statistical divergence detected.")
                
            st.caption("Disclaimer: This tool provides mathematical analysis only. All trading involves capital risk.")

        else:
            st.error("Error: Could not retrieve data. Please ensure the tickers are correct.")
