"""
Pairs Trading Pro Explorer (Global Edition)
------------------------------------------
An interactive Streamlit dashboard for researching and validating 
statistical relationships between stock pairs across different global exchanges.

Now features: 
- Automatic Currency Normalization for international pairs.
- Math engine fully aligned with Production Bot (OLS hedge ratio, rolling Z-score, ADF test, Log Returns).
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import re

# --- UI CONFIGURATION ---
st.set_page_config(page_title="Pairs Trading Pro", layout="wide")

st.warning("⚠️ **LEGAL DISCLAIMER:** This tool is for **educational and informational purposes only**. Algorithmic trading involves a high risk of losing money. Use this tool at your own risk.")

st.title("📈 Global Pairs Trading Explorer")
st.markdown("Analyze stock pairs across any exchange. The tool automatically adjusts for currency differences and uses the exact math engine as the production trading bot.")

# --- SIDEBAR: USER INPUTS ---
st.sidebar.header("Pair Configuration")

def sanitize_ticker(text):
    return re.sub(r'[^A-Z0-9.\-]', '', text.upper())

t1_raw = st.sidebar.text_input("Ticker 1 (e.g. AAPL)", value="DUK")
t2_raw = st.sidebar.text_input("Ticker 2 (e.g. SO)", value="SO")

t1 = sanitize_ticker(t1_raw)
t2 = sanitize_ticker(t2_raw)
time_period = st.sidebar.selectbox("Lookback Period", options=["6mo", "1y", "2y", "5y"], index=1)
rolling_window = st.sidebar.selectbox("Rolling Z-Score Window", options=[30, 60, 90, 120, 252], index=1)

def calculate_rsi(series, periods=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

if st.sidebar.button("Run Global Analysis"):
    with st.spinner("Fetching data and analyzing pair..."):
        # 1. FETCH METADATA (CURRENCY)
        tick1 = yf.Ticker(t1)
        tick2 = yf.Ticker(t2)
        
        curr1 = tick1.info.get('currency', 'USD')
        curr2 = tick2.info.get('currency', 'USD')
        
        # 2. DOWNLOAD HISTORICAL DATA
        data = yf.download([t1, t2], period=time_period, progress=False)
        
        if not data.empty and len(data.columns) >= 2:
            # Handle multi-level columns from yfinance
            if isinstance(data.columns, pd.MultiIndex):
                try:
                    prices = data['Close']
                except KeyError:
                    # In newer yfinance versions, it might just be the ticker names or "Adj Close"
                    prices = data.xs('Close', axis=1, level=0) if 'Close' in data.columns.get_level_values(0) else data
            else:
                prices = data

            prices = prices.dropna()
            
            if t1 not in prices.columns or t2 not in prices.columns:
                st.error("Error: Could not retrieve valid data for one or both tickers.")
                st.stop()
            
            # --- CURRENCY NORMALIZATION ENGINE ---
            is_fx_trade = curr1 != curr2
            if is_fx_trade:
                st.info(f"💱 Currency Mismatch Detected: {curr1} vs {curr2}. Normalizing to {curr1}...")
                fx_ticker = f"{curr2}{curr1}=X"
                fx_data = yf.download(fx_ticker, period=time_period, progress=False)
                
                if not fx_data.empty:
                    fx_rates = fx_data['Close'] if not isinstance(fx_data.columns, pd.MultiIndex) else fx_data['Close'].iloc[:, 0]
                    if isinstance(fx_rates, pd.DataFrame):
                        fx_rates = fx_rates.iloc[:, 0]
                        
                    prices = prices.join(fx_rates.rename('FX'), how='left').ffill().dropna()
                    prices[t2] = prices[t2] * prices['FX']
                    prices = prices[[t1, t2]]
                else:
                    st.error(f"Could not fetch exchange rate {fx_ticker}. Results may be skewed.")

            # 3. ANALYTICS ENGINE (Matched to Production Bot)
            p1_last = prices[t1].iloc[-1]
            p2_last = prices[t2].iloc[-1]
            
            # Log-return correlation
            log_returns = np.log(prices).diff().dropna()
            correlation = log_returns[t1].corr(log_returns[t2])
            
            # OLS Hedge Ratio on log prices
            log_1 = np.log(prices[t1])
            log_2 = np.log(prices[t2])
            hedge_ratio = np.polyfit(log_2, log_1, 1)[0]
            
            # Log-price spread
            spread = log_1 - hedge_ratio * log_2
            
            # ADF Cointegration Test
            adf_result = adfuller(spread.dropna(), autolag='AIC')
            pvalue = adfuller_result = adf_result[1]
            adf_stat = adf_result[0]
            
            # Rolling Z-score
            rolling_mean = spread.rolling(window=rolling_window).mean()
            rolling_std = spread.rolling(window=rolling_window).std()
            z_score = (spread - rolling_mean) / rolling_std
            curr_z = z_score.dropna().iloc[-1]
            
            rsi1, rsi2 = calculate_rsi(prices[t1]).iloc[-1], calculate_rsi(prices[t2]).iloc[-1]

            # --- UI: DASHBOARD ---
            st.subheader(f"System Health ({curr1} Normalized)")
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            with m_col1: st.metric(f"💎 {t1} Price", f"{curr1} {p1_last:.2f}")
            with m_col2: st.metric(f"🪙 {t2} Price", f"{curr1} {p2_last:.2f}")
            with m_col3: st.metric("🔗 Log-Return Corr", f"{correlation:.2%}")
            with m_col4: st.metric("🎯 Current Z", f"{curr_z:+.2f}")

            # --- UI: VISUALIZATION ---
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            norm1 = (prices[t1] / prices[t1].iloc[0]) * 100
            norm2 = (prices[t2] / prices[t2].iloc[0]) * 100
            ax1.plot(norm1, label=f"{t1}", color='#2ecc71', linewidth=2)
            ax1.plot(norm2, label=f"{t2}", color='#3498db', linewidth=2)
            ax1.set_title(f"Currency-Adjusted Growth (Starting at 100 in {curr1})")
            ax1.legend(); ax1.grid(alpha=0.3)
            
            ax2.plot(z_score, color='#9b59b6', linewidth=2)
            ax2.axhline(2.0, color='#e74c3c', linestyle='--'); ax2.axhline(-2.0, color='#27ae60', linestyle='--')
            ax2.axhline(0, color='black', linestyle=':', alpha=0.5)
            ax2.set_title(f"Rolling Z-Score Divergence ({rolling_window}-Day Window, Log Spread)")
            plt.tight_layout()
            st.pyplot(fig)
            
            # --- UI: RISK ASSESSMENT & VERDICT ---
            st.divider()
            corr_pass = correlation >= 0.70
            
            # 3-tier ADF cointegration logic
            coint_status = "✅ Strong" if pvalue < 0.05 else ("⚠️ Weak" if pvalue < 0.10 else "❌ Fail")
            coint_pass = pvalue < 0.10
            
            rsi_neutral = (30 < rsi1 < 70) and (30 < rsi2 < 70)

            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.markdown("#### 📐 Statistical Foundation")
                if corr_pass: st.success(f"**Log-Return Correlation:** {correlation:.2%} (Min 70%)")
                else: st.error(f"**Log-Return Correlation:** {correlation:.2%} (Min 70%)")
                
                if pvalue < 0.05:
                    st.success(f"**Cointegration (ADF):** P-Value {pvalue:.4f} ({coint_status})")
                elif pvalue < 0.10:
                    st.warning(f"**Cointegration (ADF):** P-Value {pvalue:.4f} ({coint_status})")
                else:
                    st.error(f"**Cointegration (ADF):** P-Value {pvalue:.4f} ({coint_status})")
                    
                st.info(f"**OLS Hedge Ratio (β):** {hedge_ratio:.3f}")

            with res_col2:
                st.markdown("#### ⚡ Momentum Check")
                st.write(f"{t1} RSI: {rsi1:.2f} | {t2} RSI: {rsi2:.2f}")
                if rsi_neutral: st.success("✅ Momentum is neutral.")
                else: st.warning("⚠️ Momentum Alert: Extreme RSI detected.")

            st.markdown("### 🏁 Final Execution Signal")
            z_signal = abs(curr_z) >= 2.0
            
            if not coint_pass or not corr_pass:
                st.error("❌ **NOT A VALID PAIR:** Fails core statistical checks (Correlation or Cointegration). Do not trade.")
            elif z_signal:
                direction = f"SELL {t1} / BUY {t2}" if curr_z > 0 else f"BUY {t1} / SELL {t2}"
                if pvalue < 0.05 and rsi_neutral:
                    st.success(f"🚀 **STRONG SIGNAL ({direction}):** Divergence is statistically overextended. Entry recommended.")
                else:
                    reasons = []
                    if pvalue >= 0.05: reasons.append("Weak Cointegration (Borderline p-value)")
                    if not rsi_neutral: reasons.append("Extreme RSI momentum")
                    if is_fx_trade: reasons.append("High FX Volatility Risk")
                    st.warning(f"⚠️ **CAUTION ({direction}):** Signal detected, but proceed carefully due to: {', '.join(reasons)}.")
            else:
                st.info("⚖️ **NO SIGNAL:** Spread is within normal bounds. No reliable divergence detected.")
        else:
            st.error("Error: Could not retrieve data. Check tickers and suffixes (e.g. .KS for Korea).")
