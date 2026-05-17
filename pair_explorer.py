"""
Pairs Trading Pro Explorer (Global Edition)
------------------------------------------
An interactive Streamlit dashboard for researching and validating 
statistical relationships between stock pairs across different global exchanges.

Now features: Automatic Currency Normalization for international pairs.
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

st.warning("⚠️ **LEGAL DISCLAIMER:** This tool is for **educational and informational purposes only**. Algorithmic trading involves a high risk of losing money. Use this tool at your own risk.")

st.title("📈 Global Pairs Trading Explorer")
st.markdown("Analyze stock pairs across any exchange. The tool automatically adjusts for currency differences.")

# --- SIDEBAR: USER INPUTS ---
st.sidebar.header("Pair Configuration")

def sanitize_ticker(text):
    return re.sub(r'[^A-Z0-9.\-]', '', text.upper())

t1_raw = st.sidebar.text_input("Ticker 1 (e.g. AAPL)", value="AAPL")
t2_raw = st.sidebar.text_input("Ticker 2 (e.g. 005930.KS)", value="005930.KS") # Samsung

t1 = sanitize_ticker(t1_raw)
t2 = sanitize_ticker(t2_raw)
time_period = st.sidebar.selectbox("Lookback Period", options=["6mo", "1y", "2y"], index=1)

def calculate_rsi(series, periods=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

if st.sidebar.button("Run Global Analysis"):
    with st.spinner("Fetching data and normalizing currencies..."):
        # 1. FETCH METADATA (CURRENCY)
        tick1 = yf.Ticker(t1)
        tick2 = yf.Ticker(t2)
        
        # Default to USD if currency metadata is missing
        curr1 = tick1.info.get('currency', 'USD')
        curr2 = tick2.info.get('currency', 'USD')
        
        # 2. DOWNLOAD HISTORICAL DATA
        data = yf.download([t1, t2], period=time_period, progress=False)
        
        if not data.empty and len(data.columns) >= 2:
            prices = data['Close'].dropna()
            
            # --- CURRENCY NORMALIZATION ENGINE ---
            if curr1 != curr2:
                st.info(f"💱 Currency Mismatch Detected: {curr1} vs {curr2}. Normalizing to {curr1}...")
                
                # Fetch exchange rate (e.g., KRWUSD=X)
                fx_ticker = f"{curr2}{curr1}=X"
                fx_data = yf.download(fx_ticker, period=time_period, progress=False)
                
                if not fx_data.empty:
                    fx_rates = fx_data['Close'].iloc[:, 0] # Get first column as series
                    # Align dates and forward fill if FX market is closed on stock holidays
                    prices = prices.join(fx_rates, how='left').ffill().dropna()
                    fx_col = prices.columns[-1]
                    
                    # Convert Ticker 2 prices to Ticker 1's currency
                    prices[t2] = prices[t2] * prices[fx_col]
                    # Clean up the dataframe
                    prices = prices[[t1, t2]]
                else:
                    st.error(f"Could not fetch exchange rate {fx_ticker}. Results may be skewed.")

            # 3. ANALYTICS ENGINE (Same as before, but on normalized prices)
            p1_last = prices[t1].iloc[-1]
            p2_last = prices[t2].iloc[-1]
            
            high_ticker, low_ticker = (t1, t2) if p1_last > p2_last else (t2, t1)
            high_price, low_price = max(p1_last, p2_last), min(p1_last, p2_last)
            
            ratio = high_price / low_price
            spread = prices[high_ticker] - (ratio * prices[low_ticker])
            z_score = (spread - spread.mean()) / spread.std()
            curr_z = z_score.iloc[-1]
            correlation = prices[t1].corr(prices[t2])
            _, pvalue, _ = coint(prices[t1], prices[t2])
            rsi1, rsi2 = calculate_rsi(prices[t1]).iloc[-1], calculate_rsi(prices[t2]).iloc[-1]

            # --- UI: DASHBOARD ---
            st.subheader(f"System Health ({curr1} Normalized)")
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            with m_col1: st.metric(f"💎 High ({high_ticker})", f"{curr1} {high_price:.2f}")
            with m_col2: st.metric(f"🪙 Low ({low_ticker})", f"{curr1} {low_price:.2f}")
            with m_col3: st.metric("🔗 Correlation", f"{correlation:.2%}")
            with m_col4: st.metric("🎯 Current Z", f"{curr_z:.2f}")

            # --- UI: VISUALIZATION ---
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            norm1 = (prices[t1] / prices[t1].iloc[0]) * 100
            norm2 = (prices[t2] / prices[t2].iloc[0]) * 100
            ax1.plot(norm1, label=f"{t1}", color='#2ecc71', linewidth=2)
            ax1.plot(norm2, label=f"{t2}", color='#3498db', linewidth=2)
            ax1.set_title(f"Currency-Adjusted Correlation (Starting at 100 in {curr1})")
            ax1.legend(); ax1.grid(alpha=0.3)
            ax2.plot(z_score, color='#9b59b6', linewidth=2)
            ax2.axhline(2.0, color='#e74c3c', linestyle='--'); ax2.axhline(-2.0, color='#27ae60', linestyle='--')
            ax2.axhline(0, color='black', linestyle=':', alpha=0.5)
            ax2.set_title("Z-Score Divergence (FX Adjusted)")
            plt.tight_layout()
            st.pyplot(fig)
            
            # --- UI: RISK ASSESSMENT & VERDICT ---
            st.divider()
            corr_pass = correlation > 0.85
            coint_pass = pvalue < 0.05
            rsi_neutral = (30 < rsi1 < 70) and (30 < rsi2 < 70)

            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.markdown("#### 📐 Statistical Foundation")
                if corr_pass: st.success(f"**Correlation:** {correlation:.2%}")
                else: st.warning(f"**Correlation:** {correlation:.2%}")
                if coint_pass: st.success(f"**Cointegration:** P-Value {pvalue:.4f}")
                else: st.error(f"**Cointegration:** P-Value {pvalue:.4f}")

            with res_col2:
                st.markdown("#### ⚡ Momentum Check")
                st.write(f"{t1} RSI: {rsi1:.2f} | {t2} RSI: {rsi2:.2f}")
                if rsi_neutral: st.success("✅ Momentum is neutral.")
                else: st.warning("⚠️ Momentum Alert: Extreme RSI detected.")

            st.markdown("### 🏁 Final Execution Signal")
            z_signal = abs(curr_z) >= 2.0
            if z_signal and coint_pass and corr_pass and rsi_neutral:
                st.success(f"🚀 **STRONG SIGNAL:** {high_ticker} is statistically overextended. Entry recommended.")
            elif z_signal:
                st.warning("⚠️ **CAUTION:** Z-Score is triggered, but FX risk or weak cointegration detected.")
            else:
                st.info("⚖️ **NO SIGNAL:** No reliable divergence detected.")
        else:
            st.error("Error: Could not retrieve data. Check tickers and suffixes (e.g. .KS for Korea).")
