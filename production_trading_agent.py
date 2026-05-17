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
5. Sends 'Health Check' reports on Mondays and Fridays.
"""

import yfinance as yf
import pandas as pd
import asyncio
from telegram import Bot
import os
from datetime import datetime
from dotenv import load_dotenv

# --- BUG FIX: DISABLE YFINANCE CACHE ---
# This prevents the "database is locked" error on shared runners like GitHub Actions
import yfinance.cache as yfc
yfc.disable()

# --- MODULAR IMPORTS ---
try:
    from trading_pairs import APPROVED_PAIRS
except ImportError:
    print("❌ Critical Error: trading_pairs.py not found.")
    APPROVED_PAIRS = {}

# --- SECURITY & ENVIRONMENT ---
load_dotenv() 
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# --- CONFIGURATION ---
TOTAL_ACCOUNT_EQUITY = 20000  

# =====================================================================
# RISK MANAGEMENT ENGINE
# =====================================================================
def calculate_kelly_position_size(win_rate, win_loss_ratio, account_equity):
    """Computes optimal dollar amount to risk with a Half-Kelly buffer."""
    kelly_fraction = win_rate - ((1 - win_rate) / win_loss_ratio)
    safe_kelly = max(0, min(kelly_fraction / 2, 0.25)) 
    return account_equity * safe_kelly

# =====================================================================
# NOTIFICATION SYSTEM
# =====================================================================
async def send_telegram_notification(message):
    """Dispatches a formatted alert message to your phone."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("❌ Error: Telegram credentials missing.")
        return
    try:
        bot = Bot(token=TELEGRAM_TOKEN)
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode='Markdown')
        print("✅ Success: Telegram message dispatched.")
    except Exception as e:
        print(f"❌ Telegram Error: {e}")

# =====================================================================
# CORE EXECUTION PIPELINE
# =====================================================================
async def run_market_scan():
    """Main loop: Iterates through stock pairs and identifies signals."""
    today = datetime.now()
    day_of_week = today.weekday() # 0 = Monday, 4 = Friday
    is_health_check_day = day_of_week in [0, 4]
    
    print(f"🤖 Starting scan for {len(APPROVED_PAIRS)} pairs...")
    trade_size_usd = calculate_kelly_position_size(0.60, 2.00, TOTAL_ACCOUNT_EQUITY)
    
    signals_detected = []
    summary_report = []

    for pair_id, details in APPROVED_PAIRS.items():
        ticker_a, ticker_b = details['ticker_a'], details['ticker_b']
        
        # 1. DOWNLOAD DATA (With no caching to avoid locks)
        data = yf.download([ticker_a, ticker_b], period="1y", progress=False)
        
        # 2. SAFETY CHECK: Ensure data exists before doing math
        if data.empty or len(data.columns) < 2:
            print(f"⚠️ Warning: Could not download complete data for {pair_id}. Skipping.")
            continue
            
        prices = data['Close'].dropna()
        
        # 3. SECOND SAFETY CHECK: Ensure prices aren't empty after dropping NaNs
        if prices.empty:
            print(f"⚠️ Warning: No overlapping dates for {pair_id}. Skipping.")
            continue
            
        p_a, p_b = prices[ticker_a].iloc[-1], prices[ticker_b].iloc[-1]
        
        hedge_ratio = p_a / p_b
        spread = prices[ticker_a] - (hedge_ratio * prices[ticker_b])
        z_score_series = (spread - spread.mean()) / spread.std()
        current_z = z_score_series.iloc[-1]
        
        shares_a = max(1, round((trade_size_usd / 2) / p_a, 2))
        shares_b = max(1, round((trade_size_usd / 2) / p_b, 2))
        
        summary_report.append(f"• {details['name']}: Z={current_z:.2f}")

        if current_z >= 2.0:
            rec = f"🟢 SELL {shares_a} {ticker_a} / BUY {shares_b} {ticker_b}"
            signals_detected.append((details['name'], current_z, rec))
        elif current_z <= -2.0:
            rec = f"🟢 BUY {shares_a} {ticker_a} / SELL {shares_b} {ticker_b}"
            signals_detected.append((details['name'], current_z, rec))

    # --- SENDING ALERTS ---
    for name, z, rec in signals_detected:
        msg = (
            f"🚀 **TRADING SIGNAL DETECTED**\n\n"
            f"**Pair:** {name}\n"
            f"**Z-Score:** {z:.2f} σ\n\n"
            f"🚨 **ACTION:**\n{rec}\n\n"
            f"🔗 [Open Research Dashboard](https://trading-bot-ui.streamlit.app/)"
        )
        await send_telegram_notification(msg)

    if is_health_check_day and not signals_detected:
        report_type = "WEEKLY OPEN" if day_of_week == 0 else "WEEKLY CLOSE"
        report_body = "\n".join(summary_report)
        health_msg = (
            f"🩺 **AGENT HEALTH CHECK: {report_type}**\n\n"
            f"The bot is running correctly. No trading signals detected today.\n\n"
            f"**Current Pair Status:**\n{report_body}\n\n"
            f"🔗 [Open Research Dashboard](https://trading-bot-ui.streamlit.app/)"
        )
        await send_telegram_notification(health_msg)

if __name__ == "__main__":
    asyncio.run(run_market_scan())
