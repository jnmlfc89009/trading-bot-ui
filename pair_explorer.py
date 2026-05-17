import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Pairs Trading Pro", layout="wide")

st.title("📈 Pairs Trading Pro Explorer")
st.markdown("Enter two tickers to analyze their correlation and Z-Score divergence.")

# --- Sidebar Inputs ---
st.sidebar.header("Pair Configuration")
t1 = st.sidebar.text_input("Ticker 1", value="LOW").upper()
t2 = st.sidebar.text_input("Ticker 2", value="HD").upper()
time_period = st.sidebar.selectbox("Lookback Period", options=["3mo", "6mo", "1y", "2y"], index=1)

if st.sidebar.button("Run Deep Analysis"):
    with st.spinner(f"Fetching market data for {t1} and {t2}..."):
        # Download data
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
            
            # --- Metrics Dashboard ---
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(f"💎 Higher Priced ({high_ticker})", f"${high_price:.2f}")
            with col2:
                st.metric(f"🪙 Lower Priced ({low_ticker})", f"${low_price:.2f}")
            
            # --- Math Engine ---
            ratio = high_price / low_price
            spread = prices[high_ticker] - (ratio * prices[low_ticker])
            z_score = (spread - spread.mean()) / spread.std()
            curr_z = z_score.iloc[-1]
            
            with col3:
                st.metric("🎯 Current Z-Score", f"{curr_z:.2f}", delta_color="inverse")

            # --- Visualization (The "Jupyter" View) ---
            st.subheader("Statistical Divergence Analysis")
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            
            # Top Chart: Normalized Prices
            norm1 = (prices[t1] / prices[t1].iloc[0]) * 100
            norm2 = (prices[t2] / prices[t2].iloc[0]) * 100
            ax1.plot(norm1, label=f"{t1} (Normalized)", color='#2ecc71', linewidth=2)
            ax1.plot(norm2, label=f"{t2} (Normalized)", color='#3498db', linewidth=2)
            ax1.set_title("Price Correlation (Starting at 100)", fontsize=14)
            ax1.legend()
            ax1.grid(alpha=0.3)

            # Bottom Chart: Z-Score
            ax2.plot(z_score, color='#9b59b6', linewidth=2)
            ax2.axhline(2.0, color='#e74c3c', linestyle='--', label="Sell Threshold")
            ax2.axhline(-2.0, color='#27ae60', linestyle='--', label="Buy Threshold")
            ax2.axhline(0, color='black', linestyle=':', alpha=0.5)
            ax2.set_title("Z-Score (Divergence Level)", fontsize=14)
            ax2.fill_between(z_score.index, z_score, 0, where=(z_score >= 0), color='#9b59b6', alpha=0.1)
            ax2.fill_between(z_score.index, z_score, 0, where=(z_score < 0), color='#9b59b6', alpha=0.1)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # --- Trading Signal Instruction ---
            if curr_z >= 2.0:
                st.error(f"🚨 SIGNAL: {high_ticker} is overvalued relative to {low_ticker}. SELL {high_ticker} / BUY {low_ticker}.")
            elif curr_z <= -2.0:
                st.success(f"🚨 SIGNAL: {high_ticker} is undervalued relative to {low_ticker}. BUY {high_ticker} / SELL {low_ticker}.")
            else:
                st.info("⚖️ Neutral: The pair is currently within normal statistical boundaries.")
                
        else:
            st.error("Error: Could not retrieve data. Please ensure the tickers are correct.")
