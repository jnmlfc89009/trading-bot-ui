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

import pandas as pd
import requests
import asyncio
import time
import random
from telegram import Bot
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

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
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

# --- CONFIGURATION ---
TOTAL_ACCOUNT_EQUITY = 20000

# Alpha Vantage free tier: 5 requests/min → wait 13s between each call to be safe
AV_RATE_LIMIT_DELAY = 13

# Remap yfinance-style SGX tickers (.SI) → Alpha Vantage format (.SES)
TICKER_MAP = {
    "D05.SI": "D05.SES",
    "O39.SI": "O39.SES",
}

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
    
    print(f"🤖 Starting scan for {len(APPROVED_PAIRS)} pairs in the database...")
    
    # Sizing
    trade_size_usd = calculate_kelly_position_size(0.60, 2.00, TOTAL_ACCOUNT_EQUITY)
    
    signals_detected = []
    summary_report = []

# =====================================================================
# DATA ENGINE — Alpha Vantage
# =====================================================================
def fetch_close_prices(ticker, max_retries=3):
    """
    Fetches 1 year of daily Close prices from Alpha Vantage.
    Automatically remaps SGX tickers from .SI to .SES format.
    Respects the 5 calls/min free tier limit via AV_RATE_LIMIT_DELAY.
    """
    av_ticker = TICKER_MAP.get(ticker, ticker)  # Remap SGX tickers if needed
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": av_ticker,
        "outputsize": "full",        # Full history; we'll slice to 1 year
        "datatype": "json",
        "apikey": ALPHA_VANTAGE_API_KEY,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            # Detect API-level errors (e.g. invalid key, rate limit message)
            if "Note" in data:
                print(f"  ⚠️  Alpha Vantage rate limit hit for {av_ticker}. Waiting 60s...")
                time.sleep(60)
                continue
            if "Error Message" in data:
                print(f"  ❌ Alpha Vantage error for {av_ticker}: {data['Error Message']}")
                return None
            if "Time Series (Daily)" not in data:
                print(f"  ⚠️  Unexpected response for {av_ticker}: {list(data.keys())}")
                return None

            # Parse into a clean Close price Series
            ts = data["Time Series (Daily)"]
            series = pd.Series(
                {pd.Timestamp(date): float(vals["4. close"]) for date, vals in ts.items()}
            ).sort_index()

            # Slice to last 1 year
            one_year_ago = pd.Timestamp.now().normalize() - pd.DateOffset(years=1)
            series = series[series.index >= one_year_ago]

            if len(series) < 30:
                print(f"  ⚠️  Insufficient data returned for {av_ticker} ({len(series)} rows).")
                return None

            print(f"  ✅ {av_ticker}: {len(series)} days of data fetched.")
            return series

        except Exception as e:
            print(f"  ↩️  Error fetching {av_ticker} (attempt {attempt + 1}/{max_retries}): {e}")
            sleep_time = (2 ** (attempt + 1)) + random.uniform(0.5, 2.0)
            print(f"  ⏳ Waiting {sleep_time:.1f}s before retry...")
            time.sleep(sleep_time)

    print(f"  ❌ Failed to fetch {av_ticker} after {max_retries} attempts.")
    return None

    for pair_id, details in APPROVED_PAIRS.items():
        ticker_a, ticker_b = details['ticker_a'], details['ticker_b']
        
        # 1. DOWNLOAD DATA via Alpha Vantage (respects rate limit between each call)
        try:
            print(f"📥 Fetching {ticker_a}...")
            close_a = fetch_close_prices(ticker_a)
            time.sleep(AV_RATE_LIMIT_DELAY)  # Respect 5 calls/min free tier limit

            print(f"📥 Fetching {ticker_b}...")
            close_b = fetch_close_prices(ticker_b)
            time.sleep(AV_RATE_LIMIT_DELAY)

            if close_a is None or close_b is None:
                print(f"⚠️ Warning: Could not fetch data for {pair_id}. Skipping.")
                continue

            # Align both series on common trading dates
            prices = pd.concat([close_a, close_b], axis=1, keys=[ticker_a, ticker_b]).dropna()

            if prices.empty or len(prices) < 30:
                print(f"⚠️ Warning: Insufficient overlapping history for {pair_id}. Skipping.")
                continue
                
            p_a, p_b = prices[ticker_a].iloc[-1], prices[ticker_b].iloc[-1]
            
            # 2. MATH
            hedge_ratio = p_a / p_b
            spread = prices[ticker_a] - (hedge_ratio * prices[ticker_b])
            z_score_series = (spread - spread.mean()) / spread.std()
            current_z = z_score_series.iloc[-1]
            
            # 3. SIZING
            shares_a = max(1, round((trade_size_usd / 2) / p_a, 2))
            shares_b = max(1, round((trade_size_usd / 2) / p_b, 2))
            
            summary_report.append(f"• {details['name']}: Z={current_z:.2f}")

            # 4. SIGNAL LOGIC
            if current_z >= 2.0:
                rec = f"🟢 SELL {shares_a} {ticker_a} / BUY {shares_b} {ticker_b}"
                signals_detected.append((details['name'], current_z, rec))
            elif current_z <= -2.0:
                rec = f"🟢 BUY {shares_a} {ticker_a} / SELL {shares_b} {ticker_b}"
                signals_detected.append((details['name'], current_z, rec))
                
        except Exception as e:
            print(f"⚠️ Error processing {pair_id}: {e}")
            continue

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
            f"The bot is running correctly. No signals detected.\n\n"
            f"**Pair Status:**\n{report_body}\n\n"
            f"🔗 [Open Research Dashboard](https://trading-bot-ui.streamlit.app/)"
        )
        await send_telegram_notification(health_msg)

if __name__ == "__main__":
    asyncio.run(run_market_scan())
