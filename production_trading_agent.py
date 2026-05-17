import yfinance as yf
import numpy as np
import pandas as pd
import asyncio
from telegram import Bot
import os
from dotenv import load_dotenv

# Load credentials (Works for both local .env and GitHub Secrets)
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# =====================================================================
# 1. CORE SYSTEM CONFIGURATION
# =====================================================================
TOTAL_ACCOUNT_EQUITY = 20000 

# High-Conviction Pairs researched today
APPROVED_PAIRS = {
    "US_UTILITIES": {
        "ticker_a": "DUK", "ticker_b": "SO", "name": "Duke Energy vs Southern Co"
    },
    "US_PAYMENTS": {
        "ticker_a": "V", "ticker_b": "MA", "name": "Visa vs Mastercard"
    }
}

# =====================================================================
# 2. RISK & POSITION SIZING ENGINE
# =====================================================================
def calculate_kelly_position_size(win_rate, win_loss_ratio, account_equity):
    """Optimizes position sizing with a Half-Kelly safety buffer."""
    kelly_fraction = win_rate - ((1 - win_rate) / win_loss_ratio)
    safe_kelly = max(0, min(kelly_fraction / 2, 0.25)) # Cap at 25%
    return account_equity * safe_kelly

# =====================================================================
# 3. TELEGRAM NOTIFICATION SYSTEM
# =====================================================================
async def send_telegram_notification(pair_name, z_score, instruction):
    """Sends trading signals to Telegram."""
    message = (
        f"🤖 **TRADING AGENT SIGNAL**\n\n"
        f"**Pair:** {pair_name}\n"
        f"**Z-Score:** {z_score:.2f} σ\n\n"
        f"🚨 **ACTION:**\n{instruction}"
    )
    try:
        bot = Bot(token=TELEGRAM_TOKEN)
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode='Markdown')
        print(f"📩 Signal sent for {pair_name}")
    except Exception as e:
        print(f"❌ Telegram Error: {e}")

# =====================================================================
# 4. EXECUTION PIPELINE
# =====================================================================
async def run_market_scan():
    print("🤖 Scanning markets...")
    trade_size_usd = calculate_kelly_position_size(0.60, 2.00, TOTAL_ACCOUNT_EQUITY)
    
    for pair_id, details in APPROVED_PAIRS.items():
        t_a, t_b = details['ticker_a'], details['ticker_b']
        
        # Download 1 year of data for stable math
        data = yf.download([t_a, t_b], period="1y", progress=False)
        if data.empty: continue
        
        prices = data['Close'].dropna()
        latest_a, latest_b = prices[t_a].iloc[-1], prices[t_b].iloc[-1]
        
        # Math Engine
        ratio = latest_a / latest_b
        spread = prices[t_a] - (ratio * prices[t_b])
        z_score = (spread - spread.mean()) / spread.std()
        curr_z = z_score.iloc[-1]
        
        # Sizing
        shares_a = max(1, round((trade_size_usd/2) / latest_a, 2))
        shares_b = max(1, round((trade_size_usd/2) / latest_b, 2))
        
        if curr_z >= 2.0:
            msg = f"🟢 SELL {shares_a} {t_a} / BUY {shares_b} {t_b}"
            await send_telegram_notification(details['name'], curr_z, msg)
        elif curr_z <= -2.0:
            msg = f"🟢 BUY {shares_a} {t_a} / SELL {shares_b} {t_b}"
            await send_telegram_notification(details['name'], curr_z, msg)
        else:
            print(f"🛑 {details['name']} Neutral ({curr_z:.2f})")

if __name__ == "__main__":
    asyncio.run(run_market_scan())
