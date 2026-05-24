"""
Production Trading Agent (Telegram Edition)
------------------------------------------
An automated statistical arbitrage agent designed to run as a scheduled
cloud task (GitHub Actions).

Workflow:
1. Ingests pre-approved stock pairs from trading_pairs.py.
2. Validates each pair with log-return correlation (min 0.70).
3. Runs an ADF cointegration test on the spread residuals.
   - p < 0.05  → ✅ Cointegrated   : full signal processing
   - p < 0.10  → ⚠️ Weak           : signal fires with caution flag
   - p >= 0.10 → ❌ Not cointegrated: no signal, sends PAIR REVIEW ALERT
4. Computes an OLS-derived hedge ratio on log prices for a stationary spread.
5. Applies a 60-day rolling Z-score for adaptive, drift-free signals.
6. Calculates optimal position sizes via the Kelly Criterion.
7. Dispatches rich indicator alerts directly to a Telegram mobile client.
8. Sends 'Health Check' reports on Mondays and Fridays.
"""

import pandas as pd
import numpy as np
import requests
import asyncio
import time
import random
from telegram import Bot
from statsmodels.tsa.stattools import adfuller
import os
from datetime import datetime
from dotenv import load_dotenv

# --- MODULAR IMPORTS ---
try:
    from trading_pairs import APPROVED_PAIRS
except ImportError:
    print("❌ Critical Error: trading_pairs.py not found.")
    APPROVED_PAIRS = {}

# --- SECURITY & ENVIRONMENT ---
load_dotenv()
TELEGRAM_TOKEN        = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID      = os.getenv("TELEGRAM_CHAT_ID")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

# --- CONFIGURATION ---
TOTAL_ACCOUNT_EQUITY = 20000

# Alpha Vantage free tier: 5 requests/min → wait 13s between each call to be safe
AV_RATE_LIMIT_DELAY = 13

# Pairs trading signal parameters
DEFAULT_WINDOW  = 60    # Fallback rolling window if not specified in pair
MIN_CORRELATION = 0.70  # Min log-return correlation to trust the pair
Z_ENTRY         = 2.0   # Minimum Z-score to trigger a signal
Z_STRONG        = 2.5   # Strong signal threshold
Z_EXTREME       = 3.0   # Extreme signal threshold

# ADF cointegration thresholds (p-value from Augmented Dickey-Fuller test)
ADF_STRONG  = 0.05   # p < 0.05  → ✅ Cointegrated
ADF_WEAK    = 0.10   # p < 0.10  → ⚠️ Weakly cointegrated (signal with caution)
                     # p >= 0.10 → ❌ Not cointegrated (PAIR REVIEW alert only)

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
# DATA ENGINE — Alpha Vantage
# =====================================================================
def fetch_close_prices(ticker, max_retries=3):
    """
    Fetches 1 year of daily Close prices from Alpha Vantage.
    Automatically remaps SGX tickers from .SI to .SES format.
    Respects the 5 calls/min free tier limit via AV_RATE_LIMIT_DELAY.
    """
    av_ticker = TICKER_MAP.get(ticker, ticker)
    url = "https://www.alphavantage.co/query"
    params = {
        "function":   "TIME_SERIES_DAILY",
        "symbol":     av_ticker,
        "outputsize": "full",
        "datatype":   "json",
        "apikey":     ALPHA_VANTAGE_API_KEY,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            if "Note" in data:
                print(f"  ⚠️  Alpha Vantage rate limit hit for {av_ticker}. Waiting 60s...")
                time.sleep(60)
                continue
            if "Information" in data and "rate limit" in data["Information"].lower():
                print(f"  ❌ Alpha Vantage daily rate limit hit: {data['Information']}")
                return None
            if "Error Message" in data:
                print(f"  ❌ Alpha Vantage error for {av_ticker}: {data['Error Message']}")
                return None
            if "Time Series (Daily)" not in data:
                print(f"  ⚠️  Unexpected response for {av_ticker}: {list(data.keys())}")
                return None

            ts = data["Time Series (Daily)"]
            series = pd.Series(
                {pd.Timestamp(date): float(vals["4. close"]) for date, vals in ts.items()}
            ).sort_index()

            one_year_ago = pd.Timestamp.now().normalize() - pd.DateOffset(years=1)
            series = series[series.index >= one_year_ago]

            if len(series) < 150: # Ensures enough data for any typical rolling window (max ~120)
                print(f"  ⚠️  Insufficient data for {av_ticker} ({len(series)} rows).")
                return None

            print(f"  ✅ {av_ticker}: {len(series)} days fetched.")
            return series

        except Exception as e:
            print(f"  ↩️  Error fetching {av_ticker} (attempt {attempt + 1}/{max_retries}): {e}")
            sleep_time = (2 ** (attempt + 1)) + random.uniform(0.5, 2.0)
            print(f"  ⏳ Waiting {sleep_time:.1f}s before retry...")
            time.sleep(sleep_time)

    print(f"  ❌ Failed to fetch {av_ticker} after {max_retries} attempts.")
    return None

# =====================================================================
# SIGNAL ANALYSIS ENGINE
# =====================================================================
def cointegration_label(adf_pvalue):
    """
    Returns a status emoji + label + trading permission based on ADF p-value.
    - p < 0.05  → ✅ Cointegrated      → proceed normally
    - p < 0.10  → ⚠️ Weakly cointegrated → proceed with caution flag
    - p >= 0.10 → ❌ Not cointegrated   → block signal, send review alert
    """
    if adf_pvalue < ADF_STRONG:
        return "✅ Cointegrated", "proceed"
    elif adf_pvalue < ADF_WEAK:
        return "⚠️ Weakly cointegrated", "caution"
    else:
        return "❌ Not cointegrated", "review"

def analyse_pair(ticker_a, ticker_b, close_a, close_b, window):
    """
    Full statistical analysis for a pair using log prices, ADF test,
    and rolling Z-score.

    Returns a dict of all computed indicators, or None if the pair is invalid.

    Steps:
      1. Align price series on common trading dates.
      2. Log-return correlation filter (min 0.70).
      3. OLS regression on log prices → statistically derived hedge ratio.
      4. Compute log-price spread.
      5. ADF test on the spread residuals → cointegration verdict.
      6. 60-day rolling Z-score for an adaptive, drift-free signal.
    """
    # 1. Align on common dates
    prices = pd.concat([close_a, close_b], axis=1, keys=[ticker_a, ticker_b]).dropna()
    if len(prices) < window + 10:
        print(f"  ⚠️  Insufficient overlapping data ({len(prices)} rows).")
        return None

    # 2. Log-return correlation filter
    log_returns = np.log(prices).diff().dropna()
    correlation = log_returns[ticker_a].corr(log_returns[ticker_b])
    if correlation < MIN_CORRELATION:
        print(f"  ⚠️  Correlation too low ({correlation:.2f} < {MIN_CORRELATION}). Skipping.")
        return None

    # 3. OLS hedge ratio on log prices (β from regressing log_a ~ β·log_b + α)
    log_a       = np.log(prices[ticker_a])
    log_b       = np.log(prices[ticker_b])
    hedge_ratio = np.polyfit(log_b, log_a, 1)[0]

    # 4. Log-price spread (percentage-based, more stationary than dollar spread)
    spread = log_a - hedge_ratio * log_b

    # 5. ADF test on spread residuals to formally confirm cointegration
    #    H0: spread has a unit root (non-stationary / NOT cointegrated)
    #    Low p-value → reject H0 → spread IS stationary → pairs ARE cointegrated
    adf_result  = adfuller(spread, autolag='AIC')
    adf_pvalue  = adf_result[1]
    adf_stat    = adf_result[0]
    coint_label, coint_status = cointegration_label(adf_pvalue)
    print(f"  🧪 ADF stat={adf_stat:.3f} | p={adf_pvalue:.4f} | {coint_label}")

    # 6. Rolling Z-score (adaptive — avoids stale full-year mean drift)
    rolling_mean = spread.rolling(window).mean()
    rolling_std  = spread.rolling(window).std()
    z_series     = (spread - rolling_mean) / rolling_std
    current_z    = z_series.dropna().iloc[-1]

    # Latest prices for position sizing
    p_a = prices[ticker_a].iloc[-1]
    p_b = prices[ticker_b].iloc[-1]

    return {
        "current_z":     current_z,
        "correlation":   correlation,
        "hedge_ratio":   hedge_ratio,
        "adf_pvalue":    adf_pvalue,
        "adf_stat":      adf_stat,
        "coint_label":   coint_label,
        "coint_status":  coint_status,   # "proceed" | "caution" | "review"
        "p_a":           p_a,
        "p_b":           p_b,
        "spread_mean":   rolling_mean.iloc[-1],
        "spread_std":    rolling_std.iloc[-1],
        "window":        window,
    }

def signal_strength_label(z):
    """Returns an emoji + label based on Z-score magnitude."""
    abs_z = abs(z)
    if abs_z >= Z_EXTREME:
        return "🔴 EXTREME"
    elif abs_z >= Z_STRONG:
        return "🟠 STRONG"
    else:
        return "🟡 MODERATE"

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

def build_signal_message(name, ticker_a, ticker_b, stats, action, trade_size_usd):
    """Builds a rich Telegram signal message with all key indicators."""
    z            = stats["current_z"]
    corr         = stats["correlation"]
    hr           = stats["hedge_ratio"]
    p_a          = stats["p_a"]
    p_b          = stats["p_b"]
    adf_pvalue   = stats["adf_pvalue"]
    coint_label  = stats["coint_label"]
    coint_status = stats["coint_status"]
    window       = stats.get("window", 60)
    strength     = signal_strength_label(z)

    direction = "Spread *ABOVE* mean → A overvalued vs B" if z > 0 else "Spread *BELOW* mean → A undervalued vs B"
    shares_a  = max(1, round((trade_size_usd / 2) / p_a, 2))
    # Leg B is hedge-ratio adjusted so both legs correctly offset the spread
    shares_b  = max(1, round((trade_size_usd / 2) * hr / p_b, 2))

    # Caution footer if weakly cointegrated
    caution_note = (
        "\n⚠️ *Caution:* ADF p-value is borderline (0.05–0.10).\n"
        "Mean-reversion is less certain. Consider reduced sizing.\n"
    ) if coint_status == "caution" else ""

    return (
        f"🚨 *TRADING SIGNAL DETECTED*\n"
        f"{'─' * 28}\n\n"
        f"📌 *Pair:* {name}\n"
        f"🔗 *Tickers:* `{ticker_a}` ↔ `{ticker_b}`\n\n"
        f"📊 *Signal Indicators*\n"
        f"  • Z-Score:       `{z:+.2f} σ`  {strength}\n"
        f"  • Correlation:   `{corr:.2f}` ✅\n"
        f"  • Hedge Ratio β: `{hr:.3f}`\n"
        f"  • Window:        `{window}-day rolling`\n"
        f"  • {direction}\n\n"
        f"🧪 *Cointegration (ADF Test)*\n"
        f"  • Status:  {coint_label}\n"
        f"  • p-value: `{adf_pvalue:.4f}` (threshold: 0.05)\n\n"
        f"💡 *Interpretation:*\n"
        f"  A divergence of `{abs(z):.2f}σ` from the {window}-day mean\n"
        f"  has historically reverted. The wider the gap,\n"
        f"  the stronger the mean-reversion case.\n"
        f"{caution_note}\n"
        f"🚀 *Recommended Action:*\n"
        f"  `{action}`\n\n"
        f"💰 *Position Size:* ~${trade_size_usd / 2:,.0f} per leg\n"
        f"  ({shares_a} shares of {ticker_a} @ ${p_a:.2f})\n"
        f"  ({shares_b} shares of {ticker_b} @ ${p_b:.2f})\n\n"
        f"🔗 [Open Research Dashboard](https://trading-bot-ui.streamlit.app/)"
    )

def build_pair_review_message(name, ticker_a, ticker_b, stats):
    """
    Builds a PAIR REVIEW NEEDED alert when ADF test shows the pair
    is no longer cointegrated. No trading signal is sent for this pair.
    """
    corr       = stats["correlation"]
    hr         = stats["hedge_ratio"]
    adf_pvalue = stats["adf_pvalue"]
    adf_stat   = stats["adf_stat"]

    return (
        f"🔄 *PAIR REVIEW NEEDED*\n"
        f"{'─' * 28}\n\n"
        f"📌 *Pair:* {name}\n"
        f"🔗 *Tickers:* `{ticker_a}` ↔ `{ticker_b}`\n\n"
        f"🧪 *Cointegration Test Failed*\n"
        f"  • ADF stat:  `{adf_stat:.3f}`\n"
        f"  • p-value:   `{adf_pvalue:.4f}` ❌ (need < 0.05)\n"
        f"  • Verdict:   Spread is *not mean-reverting*\n\n"
        f"📉 *What This Means:*\n"
        f"  The spread between `{ticker_a}` and `{ticker_b}` no longer\n"
        f"  behaves as a stationary series. Z-score signals on this\n"
        f"  pair are unreliable — the gap may not close.\n\n"
        f"  Correlation is still `{corr:.2f}` and β=`{hr:.3f}`, meaning\n"
        f"  the stocks still move together *directionally*, but the\n"
        f"  spread has drifted beyond recoverable bounds.\n\n"
        f"✅ *Recommended Actions:*\n"
        f"  1\\. Close any open positions on this pair\n"
        f"  2\\. Research a replacement with similar sector exposure\n"
        f"  3\\. Update `trading\\_pairs.py` with the new pair\n\n"
        f"  _This pair will continue to be monitored daily._\n\n"
        f"🔗 [Open Research Dashboard](https://trading-bot-ui.streamlit.app/)"
    )

def build_health_message(report_type, summary_report):
    """Builds a health check message with per-pair indicator summary."""
    report_body = "\n".join(summary_report) if summary_report else "  No pairs processed."
    return (
        f"🩺 *AGENT HEALTH CHECK: {report_type}*\n"
        f"{'─' * 28}\n\n"
        f"✅ Bot running correctly — no signals today.\n\n"
        f"📋 *Pair Indicator Summary:*\n"
        f"{report_body}\n\n"
        f"🔗 [Open Research Dashboard](https://trading-bot-ui.streamlit.app/)"
    )

# =====================================================================
# CORE EXECUTION PIPELINE
# =====================================================================
async def run_market_scan():
    """Main loop: Iterates through stock pairs and identifies signals."""
    today       = datetime.now()
    day_of_week = today.weekday()          # 0 = Monday, 4 = Friday
    is_health_check_day = day_of_week in [0, 4]

    print(f"🤖 Starting scan for {len(APPROVED_PAIRS)} pairs...")

    trade_size_usd   = calculate_kelly_position_size(0.60, 2.00, TOTAL_ACCOUNT_EQUITY)
    signals_detected = []
    review_alerts    = []
    summary_report   = []

    for pair_id, details in APPROVED_PAIRS.items():
        ticker_a = details['ticker_a']
        ticker_b = details['ticker_b']
        name     = details['name']
        window   = details.get('window', DEFAULT_WINDOW)

        print(f"\n🔎 Analysing: {name} ({ticker_a} / {ticker_b})")

        try:
            # --- FETCH DATA ---
            print(f"  📥 Fetching {ticker_a}...")
            close_a = fetch_close_prices(ticker_a)
            time.sleep(AV_RATE_LIMIT_DELAY)

            print(f"  📥 Fetching {ticker_b}...")
            close_b = fetch_close_prices(ticker_b)
            time.sleep(AV_RATE_LIMIT_DELAY)

            if close_a is None or close_b is None:
                print(f"  ⚠️  Data unavailable for {pair_id}. Skipping.")
                summary_report.append(f"  ❌ {name}: Data unavailable")
                continue

            # --- ANALYSE PAIR (correlation + ADF + rolling Z-score) ---
            stats = analyse_pair(ticker_a, ticker_b, close_a, close_b, window)
            if stats is None:
                summary_report.append(f"  ⚠️  {name}: Failed quality check")
                continue

            z            = stats["current_z"]
            corr         = stats["correlation"]
            hr           = stats["hedge_ratio"]
            adf_pvalue   = stats["adf_pvalue"]
            coint_label  = stats["coint_label"]
            coint_status = stats["coint_status"]
            strength     = signal_strength_label(z)

            print(f"  📈 Z={z:+.2f} | Corr={corr:.2f} | β={hr:.3f} | ADF p={adf_pvalue:.4f} | {coint_label}")

            # Build health check summary row for this pair (always)
            summary_report.append(
                f"  • *{name}*\n"
                f"    Z=`{z:+.2f}σ` | Corr=`{corr:.2f}` | β=`{hr:.3f}`\n"
                f"    ADF p=`{adf_pvalue:.4f}` | {coint_label} | {strength}"
            )

            # --- COINTEGRATION GATE ---
            if coint_status == "review":
                # Pair has broken down — queue a review alert, skip signal
                print(f"  ❌ Pair not cointegrated. Queuing review alert.")
                review_alerts.append(
                    build_pair_review_message(name, ticker_a, ticker_b, stats)
                )
                continue

            # --- SIGNAL LOGIC (proceed or caution) ---
            if z >= Z_ENTRY:
                action = f"SELL {ticker_a} / BUY {ticker_b}"
                msg = build_signal_message(
                    name, ticker_a, ticker_b, stats, action, trade_size_usd
                )
                signals_detected.append(msg)
            elif z <= -Z_ENTRY:
                action = f"BUY {ticker_a} / SELL {ticker_b}"
                msg = build_signal_message(
                    name, ticker_a, ticker_b, stats, action, trade_size_usd
                )
                signals_detected.append(msg)

        except Exception as e:
            print(f"  ⚠️  Error processing {pair_id}: {e}")
            summary_report.append(f"  ❌ {name}: Processing error")
            continue

    # --- SEND PAIR REVIEW ALERTS (always, when triggered) ---
    for msg in review_alerts:
        await send_telegram_notification(msg)

    # --- SEND SIGNAL ALERTS ---
    for msg in signals_detected:
        await send_telegram_notification(msg)

    # --- SEND HEALTH CHECK (Mon / Fri, no trading signals) ---
    if is_health_check_day and not signals_detected:
        report_type = "WEEKLY OPEN" if day_of_week == 0 else "WEEKLY CLOSE"
        health_msg  = build_health_message(report_type, summary_report)
        await send_telegram_notification(health_msg)

    print(
        f"\n✅ Scan complete. "
        f"{len(signals_detected)} signal(s), "
        f"{len(review_alerts)} pair review alert(s) sent."
    )

if __name__ == "__main__":
    asyncio.run(run_market_scan())
