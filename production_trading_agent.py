"""
Production Trading Agent (Telegram Edition)
------------------------------------------
An automated statistical arbitrage agent designed to run as a scheduled 
cloud task (GitHub Actions). 

Workflow:
1. Ingests pre-approved stock pairs from trading_pairs.py.
2. Performs Z-Score divergence math on 1-year historical data.
3. Calculates optimal position sizes via the Kelly Criterion.
4. Dispatches actionable signals directly to a Telegram mobile client.
"""

import yfinance as yf
import numpy as np
import pandas as pd
import asyncio
from telegram import Bot
import os
from dotenv import load_dotenv

# --- MODULAR IMPORTS ---
# Load your private stock database from the local directory
try:
    from trading_pairs import APPROVED_PAIRS
except ImportError:
    print("❌ Critical Error: trading_pairs.py not found.")
    APPROVED_PAIRS = {}

# --- SECURITY & ENVIRONMENT ---
load_dotenv() # Load local .env file if it exists
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# --- CONFIGURATION ---
TOTAL_ACCOUNT_EQUITY = 20000  # Total USD capital scale for sizing calculations

# =====================================================================
# RISK MANAGEMENT ENGINE
# =====================================================================
def calculate_kelly_position_size(win_rate, win_loss_ratio, account_equity):
    """
    Computes the optimal dollar amount to risk based on historical win probability.
    Uses a 'Half-Kelly' buffer to protect against extreme volatility.
    """
    # Standard Kelly Formula: f* = p - (1-p)/b
    kelly_fraction = win_rate - ((1 - win_rate) / win_loss_ratio)
    
    # Safety Floor: Never risk more than 25% of equity, and never risk negative amounts
    safe_kelly = max(0, min(kelly_fraction / 2, 0.25)) 
    
    return account_equity * safe_kelly

# =====================================================================
# NOTIFICATION SYSTEM
# =====================================================================
async def send_telegram_notification(pair_name, z_score, instruction):
    """Constructs and dispatches a formatted alert message to your phone."""
    message = (
        f"🤖 **TRADING AGENT SIGNAL**\n\n"
        f"**Pair:** {pair_name}\n"
        f"**Z-Score:** {z_score:.2f} σ\n\n"
        f"🚨 **ACTION:**\n{instruction}\n\n"
        f"🔗 [Open Research Dashboard](https://trading-bot-ui.streamlit.app/)"
    )
    
    try:
        bot = Bot(token=TELEGRAM_TOKEN)
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode='Markdown')
        print(f"✅ Success: Signal dispatched for {pair_name}")
    except Exception as e:
        print(f"❌ Telegram Error: {e}")

# =====================================================================
# CORE EXECUTION PIPELINE
# =====================================================================
async def run_market_scan():
    """Main loop: Iterates through stock pairs and identifies divergence opportunities."""
    print(f"🤖 Starting scan for {len(APPROVED_PAIRS)} pairs...")
    
    # Calculate target trade size based on risk parameters
    # Assumes 60% win rate and 2.0 profit/loss profile
    trade_size_usd = calculate_kelly_position_size(0.60, 2.00, TOTAL_ACCOUNT_EQUITY)
    
    for pair_id, details in APPROVED_PAIRS.items():
        ticker_a, ticker_b = details['ticker_a'], details['ticker_b']
        
        # 1. Fetch 1 year of daily closing prices
        data = yf.download([ticker_a, ticker_b], period="1y", progress=False)
        if data.empty or len(data.columns) < 2:
            print(f"⚠️ Warning: Skipping {pair_id} (Data not found)")
            continue
        
        prices = data['Close'].dropna()
        p_a, p_b = prices[ticker_a].iloc[-1], prices[ticker_b].iloc[-1]
        
        # 2. Statistical Analysis
        # Create a price-neutral spread and convert to Z-Score
        hedge_ratio = p_a / p_b
        spread = prices[ticker_a] - (hedge_ratio * prices[ticker_b])
        z_score_series = (spread - spread.mean()) / spread.std()
        current_z = z_score_series.iloc[-1]
        
        # 3. Position Sizing
        # Split trade size equally between the Long and Short legs
        shares_a = max(1, round((trade_size_usd / 2) / p_a, 2))
        shares_b = max(1, round((trade_size_usd / 2) / p_b, 2))
        
        # 4. Signal Generation Logic (+/- 2.0 Standard Deviations)
        if current_z >= 2.0:
            rec = f"🟢 SELL {shares_a} {ticker_a} / BUY {shares_b} {ticker_b}"
            await send_telegram_notification(details['name'], current_z, rec)
        elif current_z <= -2.0:
            rec = f"🟢 BUY {shares_a} {ticker_a} / SELL {shares_b} {ticker_b}"
            await send_telegram_notification(details['name'], current_z, rec)
        else:
            print(f"🛑 {details['name']}: No signal (Z={current_z:.2f})")

if __name__ == "__main__":
    # Launch the asynchronous loop
    asyncio.run(run_market_scan())
